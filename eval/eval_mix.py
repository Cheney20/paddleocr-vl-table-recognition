#!/usr/bin/env python3
"""Mixed-pool DP evaluation for PaddleOCR-VL table recognition.

* K vLLM servers per GPU x G GPUs (DP = number of GPUs, K workers each).
* All cohorts are merged into one pool and interleaved, so every worker sees a
  mix of datasets instead of finishing one dataset before the next.
* Samples are pre-assigned to workers with greedy LPT on image pixel count, so
  each worker's total image resolution is about the same.
* Scoring overlaps inference: finished predictions are batched and scored by a
  ProcessPool (TEDS is ~n^3, so it needs many CPU workers) while inference
  continues. TEDS == 1 implies TEDS-S == 1.

Usage:
  eval_mix.py --model DIR --tag TAG --cohorts gate_pub_icdar gate_val_eval10k ...
"""
import argparse
import base64
import heapq
import json
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

V5 = "/autodl-fs/data/povl_v5"
sys.path.insert(0, "/autodl-fs/data/povl_optimization/protocol")
import canonical_teds as ct  # noqa: E402

VLLM = os.environ.get("POVL_VLLM", "/tmp/cpy/bin/vllm")
PROMPT = "Table Recognition:"


def log(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def score_one_raw(job):
    """消融路径: 不抽表格, 直接把模型原始输出丢进 TEDS。"""
    sid, pred, gt, gt_fmt = job
    out = {"sample_id": sid}
    if not isinstance(pred, str) or not pred.strip():
        out.update(teds=0.0, teds_s=0.0, status="empty")
        return out
    try:
        res = ct.score_pair(pred, gt, "otsl", gt_fmt)
    except Exception as e:
        out.update(teds=0.0, teds_s=0.0, status=f"score_exc:{type(e).__name__}")
        return out
    if res.get("status") == "metric_oversize":
        out.update(teds=0.0, teds_s=0.0, status="pred_oversize")
        return out
    if res.get("status") == "gt_error":
        out.update(teds=None, teds_s=None, status="gt_error")
        return out
    t, s = res["teds"], res["teds_structure"]
    if t is not None and t >= 1.0:
        s = 1.0
    out.update(teds=t, teds_s=s, status=res["status"])
    # 空 cell 表按用户定的口径以 TEDS 为准, 常规表以 TEDS-S 为准;
    # 汇报时按 primary_metric 分开, 不要把两种口径混在一张表里排序
    for k in ("gt_fill_ratio", "gt_structure_only", "primary_metric", "primary"):
        if k in res:
            out[k] = res[k]
    return out


def score_one(job):
    sid, pred, gt, gt_fmt = job
    out = {"sample_id": sid}
    if not isinstance(pred, str) or not pred.strip():
        out.update(teds=0.0, teds_s=0.0, status="empty")
        return out
    # 抽表格本体再算分: 模型被喂整页图时可能把页眉/正文一起吐出来, 那些不该进 TEDS。
    # prose_frac 同时是诊断量 —— 它直接说明模型是在读表还是在复述整页。
    #
    # fmt=None 自动识别方言。这里**不能**像以前那样硬编码 "otsl": 模型有时会吐
    # HTML, 那样 extract_table 会返回空 -> 直接判 0 分(实测自检 243/300 失败)。
    # 识别之后再把预测与 GT 统一归一到 canonical HTML, 两边同方言再比,
    # 消除 OTSL/HTML 的口径错配。
    try:
        pred_t, info = ct.extract_table(pred, None)
    except Exception as e:
        out.update(teds=0.0, teds_s=0.0, status=f"extract_exc:{type(e).__name__}")
        return out
    out.update(extracted_format=info["format"], prose_frac=info["prose_frac"])
    if not pred_t:
        out.update(teds=0.0, teds_s=0.0, status=f"no_table:{info['format']}")
        return out
    try:
        pred_html = ct.cells_to_canonical_html(ct.parse_table(pred_t, info["format"]))
        gt_html = ct.cells_to_canonical_html(ct.parse_table(gt, gt_fmt))
        res = ct.score_pair(pred_html, gt_html, "html", "html")
    except Exception as e:  # never let one pathological sample kill a batch
        out.update(teds=0.0, teds_s=0.0, status=f"score_exc:{type(e).__name__}")
        return out
    st = res.get("status")
    if st == "metric_oversize":
        try:
            g = ct.count_tree_nodes(ct.parse_table(gt_html, "html"))
        except Exception:
            g = 10 ** 9
        if g <= 4000:  # GT is scoreable; the prediction blew up -> model failure
            out.update(teds=0.0, teds_s=0.0, status="pred_oversize")
        else:
            out.update(teds=None, teds_s=None, status="gt_oversize")
        return out
    if st == "gt_error":
        out.update(teds=None, teds_s=None, status="gt_error")
        return out
    t, s = res["teds"], res["teds_structure"]
    if t is not None and t >= 1.0:
        s = 1.0
    out.update(teds=t, teds_s=s, status=st,
               merge_f1=res.get("merged_rect_f1"))
    return out


# ----------------------------------------------------------------------------
# servers
# ----------------------------------------------------------------------------
def gpu_free_bytes(gpu: int) -> int:
    """查某张卡的空闲显存; 查不到就返回一个保守估计。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu),
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=20)
        return int(out.decode().strip().splitlines()[0]) * 1024 ** 2
    except Exception:
        return 40 * 10 ** 9


def plan_workers(gpus, per_gpu_hint: int, util: float, reserve: float = 0.10):
    """按每卡空闲显存决定每卡几个 vLLM worker。

    规则(用户 2026-10-18): 每卡 2-4 个 worker, 按 gpu_memory_utilization 算,
    且**worker 往大来** —— 显存够就顶满 4 个, 不够再往下退, 低于 2 就退到 1。
    真正的 KV 预算在 start_servers 里按 util/per_gpu 再分一次。
    """
    need = 4 * 10 ** 9          # 每个 engine 的模型权重 + 激活 + 一点 KV
    plan = []
    for g in gpus:
        free = gpu_free_bytes(g)
        budget = int(free * (1 - reserve))
        n = max(1, min(per_gpu_hint, budget // need))
        plan.append((g, n, free))
    return plan


def start_servers(model, gpus, per_gpu, base_port, gmu, max_len, logdir, served):
    procs = []
    for gi, g in enumerate(gpus):
        for w in range(per_gpu):
            port = base_port + gi * per_gpu + w
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                       VLLM_ATTENTION_BACKEND="TRITON_ATTN",
                       VLLM_USE_FLASHINFER_SAMPLER="0",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            cmd = [VLLM, "serve", model, "--port", str(port),
                   "--served-model-name", served,
                   "--max-model-len", str(max_len),
                   *(["--kv-cache-memory", str(int(gmu)),
                      "--gpu-memory-utilization", str(round(0.9 / per_gpu, 3))] if gmu > 1 else
                     ["--gpu-memory-utilization", str(gmu)]),
                   "--max-num-batched-tokens", "16384",
                   "--max-num-seqs", "128",
                   "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"]
            if os.environ.get("POVL_TRC", "1") == "1":
                cmd.append("--trust-remote-code")
            fh = open(f"{logdir}/vllm_g{g}_w{w}.log", "w")
            p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            procs.append((p, fh, port))
            # stagger start on the same GPU so memory profiling does not collide
            time.sleep(2)
        time.sleep(1)
    deadline = time.time() + 1800
    for p, _, port in procs:
        while True:
            if p.poll() is not None:
                stop_servers(procs)
                raise RuntimeError(f"vLLM :{port} died; see {logdir}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                break
            except Exception:
                if time.time() > deadline:
                    stop_servers(procs)
                    raise RuntimeError(f"vLLM :{port} timeout")
                time.sleep(3)
    log(f"{len(procs)} vLLM servers ready")
    return procs


def stop_servers(procs):
    for p, _, _ in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    t = time.time() + 30
    while time.time() < t and any(p.poll() is None for p, _, _ in procs):
        time.sleep(0.5)
    for p, fh, _ in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        fh.close()


# ----------------------------------------------------------------------------
# load balancing
# ----------------------------------------------------------------------------
def load_from_packs(pack_root, cohorts=None):
    """从 parquet shards 读评测行。

    与 cohort jsonl 路径产出同构的 row dict, 区别只有:
      image  = (shard, row_group, idx) 定位符, 由 _infer 惰性解析
      gt/gt_format 取自 gt_format_native 指向的那一份
        (OTSL->HTML 是有损方向, 见 eval_pack.otsl_roundtrip_ok)
    """
    import eval_pack_reader as epr
    epr.PACK_ROOT = pack_root
    rows = []
    for shard in epr.shard_files(pack_root):
        for rg, idx, meta, sid in epr.index_rows(shard, pack_root):
            if not meta.get("has_image"):
                continue  # 缺图行无法推理
            if cohorts and meta.get("cohort") not in cohorts:
                continue
            r = dict(meta)
            r["_cohort"] = meta.get("cohort")
            r["_shard"] = shard
            if not r.get("sample_id"):
                # 很多源的 sample_id 是空的; 合成一个稳定 id,
                # 跨 pack join(跨度子集等)要靠它
                r["sample_id"] = f"{os.path.basename(shard)}#{rg}#{idx}"
            r["sample_id"] = sid
            r["image"] = (shard, rg, idx)
            native = meta.get("gt_format_native") or "otsl"
            r["gt"] = meta["gt_otsl"] if native == "otsl" else meta["gt_html"]
            r["gt_format"] = native
            r["_px"] = meta["img_w"] * meta["img_h"]   # 宽高已是列, 无需解码
            rows.append(r)
    return rows


def image_pixels(path, row=None):
    """像素量, 供 lpt_assign 做负载均衡。

    parquet 行直接用 img_w/img_h 列, 不解码。
    注意: 这里**不能**像以前那样 except: return 0 —— 静默的 0 会让 LPT 权重
    全零、负载均衡无声失效, 而且不会报任何错。解不开就抛。
    """
    if isinstance(path, tuple):
        if row is not None and row.get("img_w"):
            return row["img_w"] * row["img_h"]
        return 1  # 缺图行(缺图的行本就该被过滤掉)
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return w * h


def lpt_assign(rows, n_workers):
    """Greedy LPT on pixel count; return per-worker lists (each mixed-shuffled)."""
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["_px"])
    heap = [(0, w) for w in range(n_workers)]
    buckets = [[] for _ in range(n_workers)]
    for i in order:
        load, w = heapq.heappop(heap)
        buckets[w].append(rows[i])
        heapq.heappush(heap, (load + rows[i]["_px"], w))
    rng = random.Random(0)
    for b in buckets:
        rng.shuffle(b)  # mix datasets and sizes within each worker
    return buckets


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------
def shard_work_units(rows, n_workers):
    """把 rows 按 shard 聚成"工作单元", 按像素量降序。

    动态调度(而不是静态切桶)的前提: 每个单元必须能独立完成。
    单元按成本降序进队列, worker 领完一个跑完再去领下一个 ——
    领用本身是独占的(Queue.get), 所以不会出现两个 worker 抢同一片。
    """
    by = {}
    for r in rows:
        by.setdefault(r["_shard"], []).append(r)
    units = [(sum(x["_px"] for x in rs), sh, rs) for sh, rs in by.items()]
    units.sort(key=lambda t: -t[0])          # 贵的先领, 避免长尾
    log(f"  工作单元: {len(units)} 个 shard (按像素量降序), workers={n_workers}")
    return units


def infer(port, row, max_tokens, served, timeout):
    try:
        return _infer(port, row, max_tokens, served, timeout)
    except Exception as e:
        return {"prediction": "", "finish_reason": "error", "error": f"{type(e).__name__}: {e}"[:300]}


def _infer(port, row, max_tokens, served, timeout):
    # row["image"] 可能是散图路径, 也可能是 parquet 定位符 (shard, rg, idx)。
    # 定位符必须在这里才解析成字节 —— 见 eval_pack_reader 的模块说明。
    loc = row["image"]
    if isinstance(loc, tuple):
        import eval_pack_reader as epr
        raw = epr.image_bytes(*loc)
        mime = "image/png" if row.get("img_format", "").startswith("png") else "image/jpeg"
    else:
        mime = "image/png" if loc.lower().endswith(".png") else "image/jpeg"
        raw = open(loc, "rb").read()
    b64 = base64.b64encode(raw).decode()
    body = {"model": served, "temperature": 0, "max_tokens": max_tokens,
            "skip_special_tokens": False,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": PROMPT}]}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    for attempt in range(3):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            ch = r["choices"][0]
            return {"prediction": ch["message"]["content"] or "",
                    "finish_reason": ch.get("finish_reason"),
                    "completion_tokens": r.get("usage", {}).get("completion_tokens"),
                    "latency": round(time.time() - t0, 2)}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return {"prediction": "", "finish_reason": "error", "error": err[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--gmu", type=float, default=0.0, help="0 = auto (0.92/per_gpu)")
    ap.add_argument("--conc-per-worker", type=int, default=24)
    ap.add_argument("--score-workers", type=int, default=96)
    ap.add_argument("--score-batch", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=20000)
    ap.add_argument("--base-port", type=int, default=18000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--apply-blocklist", action="store_true",
                    help="在脚本内再过滤一次表过长/GT不可用的样本"
                         "(默认不过滤, 请直接用 eval_packs_filtered)")
    ap.add_argument("--span-subset", default=SPAN_SUBSET_DEFAULT,
                    help="15k 跨行跨列子集目录; 只做身份标记, 不额外推理")
    ap.add_argument("--packs", default="",
                    help="parquet shards 根目录; 非空则从 shard 读, 忽略 cohort jsonl")
    a = ap.parse_args()

    outdir = f"{V5}/eval/runs/{a.tag}"
    logdir = f"{outdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    global SPAN
    SPAN = load_span_subset(a.span_subset) if a.span_subset else {}
    if SPAN:
        log(f"跨度子集: {len(SPAN)} 个 sample_id 会在汇总时分档")

    pred_path = f"/dev/shm/povl_v5/preds/{a.tag}.jsonl"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    score_path = f"{outdir}/scores.jsonl"

    rows, cohort_of = [], {}
    if a.packs:
        rows = load_from_packs(a.packs, a.cohorts)
    else:
        for c in a.cohorts:
            for line in open(f"{V5}/eval/cohorts/{c}.jsonl"):
                r = json.loads(line)
                r["_cohort"] = c
                rows.append(r)

    # 默认不做名单过滤 —— 用 data/eval_packs_filtered 那份已经把表过长 / GT 不可用
    # 的样本剔掉的 parquet 即可, 过滤发生在数据侧而不是代码侧, 可审计。
    # 只有在直接喂未过滤的包/cohort 时才需要在这里兜一层。
    if a.apply_blocklist:
        import eval_blocklist
        bl = eval_blocklist.Blocklist.load()
        rows, dropped = bl.filter_rows(rows)
        log(f"排除名单 {bl.summary()}, 本次剔除 {dict(dropped)} -> 剩 {len(rows)}")

    if a.limit:
        random.Random(1).shuffle(rows)
        rows = rows[:a.limit]
    # resume: skip already scored samples
    done = set()
    if os.path.exists(score_path):
        for line in open(score_path):
            try:
                d = json.loads(line)
                done.add((d["cohort"], d["sample_id"]))
            except Exception:
                pass
    rows = [r for r in rows if (r["_cohort"], r["sample_id"]) not in done]
    log(f"{len(rows)} samples to run ({len(done)} already scored), cohorts={a.cohorts}")
    if not rows:
        summarize(score_path, outdir, a)
        return
    if a.packs:
        # 宽高已经是 parquet 列, 不用再全量 stat 一遍
        for r in rows:
            r["_px"] = r["img_w"] * r["img_h"]
    else:
        with ThreadPoolExecutor(64) as ex:
            for r, px in zip(rows, ex.map(lambda r: image_pixels(r["image"]), rows)):
                r["_px"] = px
    served = "m"
    # OOM / startup failure -> fall back to fewer vLLM workers per GPU
    while True:
        plan = plan_workers(a.gpus, a.per_gpu, a.gmu or 0.9)
        log("  GPU 规划: " + ", ".join(
            f"gpu{g}->{n}worker(空闲{free/1e9:.0f}G)" for g, n, free in plan))
        per_gpu = max(n for _, n, _ in plan)
        n_workers = sum(n for _, n, _ in plan)
        # KV 预算: 按每卡空闲显存给每个 engine, 避免并发探测互相干扰
        gmu = a.gmu if a.gmu > 1 else int(
            min(free for _, _, free in plan) * 0.55 / per_gpu)
        log(f"  每卡 {per_gpu} worker, 共 {n_workers} 个, "
            f"每 engine KV 预算 {gmu/1e9:.1f} GB")
        try:
            procs = start_servers(a.model, a.gpus, per_gpu, a.base_port, gmu,
                                  a.max_model_len, logdir, served)
            break
        except RuntimeError as e:
            if per_gpu <= 1:
                raise
            log(f"server start failed ({e}); per-gpu {per_gpu} -> {per_gpu - 1}")
            a.per_gpu -= 1
            a.gmu = 0.0
            time.sleep(10)
    n_workers = len(a.gpus) * a.per_gpu
    # 动态调度: 成本降序的 shard 队列, worker 独占领取(见 worker_loop)
    shard_q = queue.Queue()
    for cost, shard, srows in shard_work_units(rows, n_workers):
        shard_q.put((cost, shard, srows))
    done_shards = []
    loads = [0]
    log(f"LPT: {n_workers} workers ({len(a.gpus)} GPU x {a.per_gpu}), "
        f"pixel load min/max = {min(loads)/1e6:.1f}M/{max(loads)/1e6:.1f}M")
    ports = [port for _, _, port in procs]

    done_q = queue.Queue()
    total = len(rows)
    t_start = time.time()

    def worker_loop(widx):
        """一个 worker 一个 vLLM 服务, 循环从共享队列领取未处理的 shard。

        队列是独占领取的: 一个 shard 被某个 worker 取走后, 其它 worker
        再取就拿不到同一片。worker 跑完手上的 shard 会立刻去看有没有
        别人没领的, 有就接着跑 —— 长尾 worker 不会早退, 空转的 worker
        也不会闲着。
        """
        port = ports[widx]
        while True:
            try:
                _, shard, srows = shard_q.get_nowait()
            except queue.Empty:
                break
            t0 = time.time()
            with ThreadPoolExecutor(a.conc_per_worker) as ex:
                futs = {ex.submit(infer, port, r, a.max_tokens, served,
                                  a.timeout): r for r in srows}
                for f in futs:
                    res = f.result()
                    done_q.put((futs[f], res))
            done_shards.append((shard, len(srows), time.time() - t0))
            log(f"  [w{widx}] shard {shard} 完成 "
                f"({len(srows)} 行, {time.time()-t0:.0f}s), "
                f"剩余 {shard_q.qsize()}")

    threads = [threading.Thread(target=worker_loop, args=(i,), daemon=True)
               for i in range(n_workers)]
    for t in threads:
        t.start()

    def _score_init():
        """打分进程降优先级。

        TEDS 是纯 CPU 密集的编辑距离, 推理侧是 GPU 喂数, 两边抢同一批核。
        打分让位: 推理跑满的时候打分慢一点无所谓, 反过来会拖慢喂数、
        把 GPU 饿着。
        """
        try:
            os.nice(19)
        except Exception:
            pass
        # 打分池最多占一半的核, 剩下的留给推理
        try:
            c = len(os.sched_getaffinity(0))
            os.sched_setaffinity(0, set(range(0, max(1, c // 2))))
        except Exception:
            pass

    pool = ProcessPoolExecutor(a.score_workers, initializer=_score_init)
    pending = []
    n_pred = n_scored = 0
    pf = open(pred_path, "a")
    sf = open(score_path, "a")
    batch = []

    def flush_scores(block=False):
        nonlocal n_scored, pending
        keep = []
        for fut, meta in pending:
            if block or fut.done():
                for res, r in zip(fut.result(), meta):
                    res.update(cohort=r["_cohort"], dataset=r.get("dataset"),
                               origin=r.get("origin"),
                               has_numeric=r.get("has_numeric"),
                               res_bucket=r.get("res_bucket"),
                               metrics=r.get("metrics"),
                               n_merge=r.get("n_merge"),
                               difficulty=SPAN.get(r.get("sample_id") or ""),
                               n_spans=r.get("n_spans"),
                               finish_reason=r["_fr"], ctoks=r["_ct"])
                    sf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    n_scored += 1
                sf.flush()
            else:
                keep.append((fut, meta))
        pending = keep

    def submit(b):
        jobs = [(r["sample_id"], r["_pred"], r["gt"], r["gt_format"]) for r in b]
        # chunksize=1 so one huge table cannot pin a whole chunk
        fut = pool.submit(_score_batch, jobs)
        pending.append((fut, b))

    alive = lambda: any(t.is_alive() for t in threads)
    last = 0
    while alive() or not done_q.empty():
        try:
            r, res = done_q.get(timeout=1)
        except queue.Empty:
            flush_scores()
            continue
        dead = [port for p, _, port in procs if p.poll() is not None]
        if dead:
            raise RuntimeError(f"vLLM died: {dead}")
        rec = {"sample_id": r["sample_id"], "cohort": r["_cohort"], **res}
        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_pred += 1
        r["_pred"], r["_fr"], r["_ct"] = res["prediction"], res.get("finish_reason"), res.get("completion_tokens")
        batch.append(r)
        if len(batch) >= max(1, a.score_batch // 16):
            submit(batch)
            batch = []
        flush_scores()
        if time.time() - last > 30:
            el = time.time() - t_start
            log(f"pred {n_pred}/{total} ({n_pred/el:.1f}/s)  scored {n_scored}  pending_batches {len(pending)}")
            last = time.time()
    pf.close()
    if batch:
        submit(batch)
    stop_servers(procs)
    log(f"inference done in {time.time()-t_start:.0f}s; draining scoring")
    flush_scores(block=True)
    pool.shutdown()
    sf.close()
    log(f"all done in {time.time()-t_start:.0f}s")
    summarize(score_path, outdir, a)


def _score_batch(jobs):
    return [score_one(j) for j in jobs]


# ----------------------------------------------------------------------------
# 跨行跨列子集 (15k): 不重复推理, 只在跑全量时把命中的行标出来再分层汇报
# ----------------------------------------------------------------------------
SPAN_SUBSET_DEFAULT = "/autodl-tmp/span15k"
SPAN = {}          # sample_id -> difficulty, 由 --span-subset 填充


def load_span_subset(path):
    """-> {sample_id: difficulty}。子集是 shard_* 的子集, 推理照常跑一遍,
    这里只做身份标记, 汇报时按 difficulty 聚合。"""
    import glob as _glob
    out = {}
    for f in sorted(_glob.glob(os.path.join(path, "*.parquet"))):
        t = pq.read_table(f, columns=["sample_id", "difficulty"])
        for sid, d in zip(t.column("sample_id").to_pylist(),
                          t.column("difficulty").to_pylist()):
            out.setdefault(sid, d)
    return out



def summarize(score_path, outdir, a):
    agg = defaultdict(lambda: defaultdict(list))

    def add(k, d):
        if d["teds"] is None:
            agg[k]["na"].append(1)
            return
        agg[k]["teds"].append(d["teds"])
        agg[k]["teds_s"].append(d["teds_s"])
        if d.get("finish_reason") == "length":
            agg[k]["trunc"].append(1)

    for line in open(score_path):
        d = json.loads(line)
        c = d["cohort"]
        add(c, d)
        if c.startswith("cx_") and d.get("dataset"):
            add(f"{c}/{d['dataset']}", d)
        # parquet 带的新维度: 自建/公开, 有无数值, 分辨率档
        # 跨行跨列三档: 15k 子集的行会被额外聚合一次
        if d.get("difficulty"):
            agg[f"_span/{d['difficulty']}"]["teds"].append(
                d["teds"] if d["teds"] is not None else 0)
            agg[f"_span/{d['difficulty']}"]["teds_s"].append(
                d["teds_s"] if d["teds_s"] is not None else 0)
            if d["teds"] is None:
                agg[f"_span/{d['difficulty']}"]["na"].append(1)
            if d.get("benchmark"):
                agg[f"_span/{d['difficulty']}/{d['benchmark']}"]["teds"].append(
                    d["teds"] if d["teds"] is not None else 0)
                agg[f"_span/{d['difficulty']}/{d['benchmark']}"]["teds_s"].append(
                    d["teds_s"] if d["teds_s"] is not None else 0)
                if d["teds"] is None:
                    agg[f"_span/{d['difficulty']}/{d['benchmark']}"]["na"].append(1)
        for dim in ("origin", "has_numeric", "res_bucket"):
            v = d.get(dim)
            if v is not None:
                agg[f"_by/{dim}/{v}"]["teds"].append(d["teds"] if d["teds"] is not None else 0)
                agg[f"_by/{dim}/{v}"]["teds_s"].append(d["teds_s"] if d["teds_s"] is not None else 0)
                if d["teds"] is None:
                    agg[f"_by/{dim}/{v}"]["na"].append(1)
    out = {}
    for c, v in sorted(agg.items()):
        n = len(v["teds"])
        if not n:
            continue
        out[c] = {"n": n, "teds": round(sum(v["teds"]) / n, 5),
                  "teds_s": round(sum(v["teds_s"]) / n, 5),
                  "truncated": len(v["trunc"]), "na": len(v["na"])}
    json.dump(out, open(f"{outdir}/summary.json", "w"), indent=1)
    for c, v in out.items():
        print(f"{c:40s} n={v['n']:6d} TEDS={v['teds']:.4f} TEDS-S={v['teds_s']:.4f} trunc={v['truncated']}")
    # mustard / gate_mustard 的 GT 是纯结构(全空 td), 只有 TEDS-S 有意义;
    # 它的 TEDS 列天然偏低, 不要当成模型问题读。
    for c in out:
        if "mustard" in c and not c.startswith("_by/"):
            print(f"  注: {c} 为纯结构集, 请只看 TEDS-S")


if __name__ == "__main__":
    main()
