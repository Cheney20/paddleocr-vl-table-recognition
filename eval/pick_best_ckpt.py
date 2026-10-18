#!/usr/bin/env python3
"""Pick the best checkpoint from the full eval runs.

Criterion (user 2026-10-18):
  1. highest primary metric on the full 184k set
     primary = TEDS for structure-only tables, TEDS-S otherwise
  2. tie-break: number of sources that beat orig

Prints a table and writes best.json; deleting the losers is a separate,
explicit step -- this script never removes anything.
"""
import collections
import glob
import json
import os
import statistics as st

RUNS = "/autodl-fs/data/povl_v5/eval/runs"
ORIG = "orig_full"


def load(tag):
    p = f"{RUNS}/{tag}/scores.jsonl"
    if not os.path.exists(p):
        return None
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def primary(d):
    """汇报口径 = TEDS-S。

    纯结构表(gt_structure_only)的 TEDS 根本不计算, 是 null;
    其余行两个都算, 但汇报用 TEDS-S(和 canonical_teds.primary_metric 一致)。
    """
    if d.get("primary") is not None:
        return d["primary"]
    return d.get("teds_s")


def summarise(rows):
    per_src = collections.defaultdict(list)
    span = collections.defaultdict(list)
    tot = []
    for d in rows:
        v = primary(d)
        if v is None:
            continue
        src = d.get("benchmark") or d.get("dataset") or d.get("cohort") or "?"
        per_src[src].append(v)
        tot.append(v)
        if d.get("difficulty"):
            span[d["difficulty"]].append(d.get("teds_s"))
    return {
        "n": len(tot),
        "primary": st.mean(tot) if tot else None,
        # 纯结构表的 teds 是 null, 必须过滤掉而不是当 0
        "teds": st.mean([d["teds"] for d in rows if d.get("teds") is not None]) if any(d.get("teds") is not None for d in rows) else None,
        "teds_s": st.mean([d["teds_s"] for d in rows if d.get("teds_s") is not None]) if any(d.get("teds_s") is not None for d in rows) else None,
        "per_source": {k: st.mean(v) for k, v in per_src.items()},
        "span_teds_s": {k: (st.mean(v) if v else None) for k, v in span.items()},
    }


# eval/runs/ 里有 111 个历史 run(不同数据/不同模型), 只认本轮的:
# orig_full 基线 + sft_v7b 的 checkpoint。
PREFIXES = ("orig_full", "v0-")


def main():
    import re
    tags = []
    for p in glob.glob(f"{RUNS}/*/scores.jsonl"):
        tag = os.path.basename(os.path.dirname(p))
        if tag == ORIG or tag.startswith(PREFIXES):
            tags.append(tag)
    tags.sort()
    print(f"本轮 run {len(tags)} 个 (已跳过 {len(glob.glob(f'{RUNS}/*/scores.jsonl'))-len(tags)} 个历史 run)\n")

    orig = load(ORIG)
    orig_sum = summarise(orig) if orig else None

    out = {}
    for t in tags:
        rows = load(t)
        if rows:
            out[t] = summarise(rows)

    ranked = sorted(out.items(), key=lambda kv: -(kv[1]["primary"] or 0))

    hdr = f"{'tag':28s}{'n':>8s}{'primary':>9s}{'TEDS':>8s}{'TEDS-S':>8s}" \
          f"{'beat_orig':>11s}{'sp/simple':>11s}{'sp/medium':>10s}{'sp/hard':>9s}"
    print(hdr); print("-" * len(hdr))
    for tag, s in ranked:
        beat = "-"
        if orig_sum and tag != ORIG:
            n_beat = sum(1 for k, v in s["per_source"].items()
                         if orig_sum["per_source"].get(k) is not None
                         and v > orig_sum["per_source"][k])
            beat = f"{n_beat}/{len(s['per_source'])}"
        sp = s["span_teds_s"]
        f = lambda x: f"{x:.4f}" if x is not None else "-"
        print(f"{tag:28s}{s['n']:8d}{s['primary']:9.4f}"
              f"{f(s['teds']):>8s}{f(s['teds_s']):>8s}{beat:>11s}"
              f"{f(sp.get('simple')):>11s}{f(sp.get('medium')):>10s}"
              f"{f(sp.get('hard')):>9s}")

    if orig_sum:
        print("\n=== 逐来源对比 (top1 候选 vs orig) ===")
        best_tag, best = ranked[0]
        print(f"{'source':22s}{'orig':>9s}{best_tag[:16]:>18s}{'Δ':>9s}")
        for k in sorted(best["per_source"]):
            o = orig_sum["per_source"].get(k)
            v = best["per_source"][k]
            if o is None:
                continue
            print(f"{k:22s}{o:9.4f}{v:18.4f}{v-o:+9.4f}")

    if ranked:
        best_tag = ranked[0][0]
        json.dump({"best": best_tag,
                   "ranking": [{"tag": t, "primary": s["primary"],
                                "n": s["n"]} for t, s in ranked]},
                  open("/autodl-tps/best.json", "w") if False else
                  open("/root/best_ckpt.json", "w"), indent=2, ensure_ascii=False)
        print(f"\nBEST = {best_tag}  (primary {ranked[0][1]['primary']:.4f})")
        print("已写 /root/best_ckpt.json —— 删除其余 ckpt 需要你确认后再执行")


if __name__ == "__main__":
    main()
