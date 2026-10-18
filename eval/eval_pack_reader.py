#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 data/eval_packs/ 下的 parquet shards。

与 src/pack_reader.py(训练侧 zip shards)是同一套思路: 单一实现, 评测侧
import 它而不是各自重写。

**为什么必须惰性取图**
eval_mix.py 把整个 worker bucket 的 row dict 捕获进闭包一次性 submit, 所以
row 里绝不能放解码后的图片字节 —— pub_pt1m(93,656) + pub_synth(43,433) 同时驻留
会吃掉几十 GB。这里 row["image"] 存的是 (shard, row_group, idx) 定位符,
在真正要推理的那一刻才解析成 bytes。

绝不要在请求路径上 `pq.read_table(shard).to_pylist()`: 那是每个请求重解码整个
row group, 768 路并发下会直接打爆。这里按 (线程, shard, row_group) 缓存已解码的
row group, LRU 淘汰。
"""
import os
import threading
from collections import OrderedDict

import pyarrow.parquet as pq

PACK_ROOT = os.environ.get("POVL_EVAL_PACK_ROOT",
                           "/autodl-fs/data/povl_v5/data/eval_packs")

# ==============================================================================
# Schema 适配
# ==============================================================================
# 两套 pack schema 都在用：
#
#   旧 (eval_packs)          新 (reshard, benchmarks_v2)
#   ---------------------    --------------------------------------
#   image                    bytes
#   cohort                   benchmark
#   dataset                  source
#   gt_otsl                  otsl
#   gt_html                  html
#   gt_format_native         (缺 -> 由 otsl/html 推断)
#   img_w / img_h            width / height
#   megapixels               (缺 -> 算)
#   has_image                (缺 -> bytes 非空)
#   n_rows / n_cols          rows / cols
#   img_format / metrics /   (缺 -> None)
#   origin / group / ...
#
# 下面按实际列名做一次探测，之后统一走 *_COLS / *_ALIAS 查表，
# 上层 (eval_mix.load_from_packs) 不需要知道自己读的是哪套。
_IMAGE_COL = None
_ALIAS: dict = {}


def detect_schema(shard: str) -> None:
    """探测一次 pack 的列名，写入模块级查表。thread-safe 靠 GIL 下的赋值语义。"""
    global _IMAGE_COL, _ALIAS
    cols = set(pq.ParquetFile(os.path.join(PACK_ROOT, shard)).schema_arrow.names)
    old = {
        "image": "image", "cohort": "cohort", "dataset": "dataset",
        "gt_otsl": "gt_otsl", "gt_html": "gt_html",
        "gt_format_native": "gt_format_native", "img_w": "img_w",
        "img_h": "img_h", "megapixels": "megapixels", "has_image": "has_image",
        "img_format": "img_format", "metrics": "metrics",
        "n_rows": "n_rows", "n_cols": "n_cols", "n_cells": "n_cells",
        "has_numeric": "has_numeric", "res_bucket": "res_bucket",
        "otsl_roundtrip_ok": "otsl_roundtrip_ok", "gt_conflict": "gt_conflict",
        "origin": "origin", "group": "group",
    }
    new = {
        "image": "bytes", "cohort": "benchmark", "dataset": "source",
        "gt_otsl": "otsl", "gt_html": "html",
        "img_w": "width", "img_h": "height",
        "n_rows": "rows", "n_cols": "cols",
    }
    if "image" in cols:
        _ALIAS, _IMAGE_COL = old, "image"
    else:
        _ALIAS, _IMAGE_COL = new, "bytes"


def alias(name: str) -> str:
    """旧字段名 -> 实际列名；该字段在这套 schema 里不存在时原样返回。"""
    return _ALIAS.get(name, name)


_local = threading.local()


def _cache() -> "OrderedDict":
    c = getattr(_local, "cache", None)
    if c is None:
        c = _local.cache = OrderedDict()
        _local.files = {}
    return c


def _files():
    f = getattr(_local, "files", None)
    if f is None:
        f = _local.files = {}
    return f


def image_bytes(shard: str, row_group: int, idx: int, max_cached: int = 2) -> bytes:
    """取回一张图的原始编码字节。thread-safe。"""
    key = (shard, row_group)
    cache = _cache()
    tbl = cache.get(key)
    if tbl is None:
        pf = _files().get(shard)
        if pf is None:
            pf = _files()[shard] = pq.ParquetFile(os.path.join(PACK_ROOT, shard))
        # 只取 image 一列, 不解码其它列
        tbl = pf.read_row_group(row_group, columns=[_IMAGE_COL])
        cache[key] = tbl
        while len(cache) > max_cached:
            cache.popitem(last=False)
    return tbl.column("image")[idx].as_py()


def shard_files(root: str = None):
    root = root or PACK_ROOT
    return sorted(f for f in os.listdir(root)
                  if f.startswith("shard_") and f.endswith(".parquet"))


def load_manifest(root: str = None) -> dict:
    import json
    root = root or PACK_ROOT
    p = os.path.join(root, "MANIFEST.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def iter_rows(shard: str, root: str = None, columns=None):
    """顺序遍历一个 shard 的全部行(建索引 / 统计用, 不要走推理热路径)。"""
    root = root or PACK_ROOT
    pf = pq.ParquetFile(os.path.join(root, shard))
    for rg in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(rg, columns=columns).to_pylist():
            yield rg, r


def index_rows(shard: str, root: str = None):
    """-> [(row_group, idx, meta_dict)], meta 不含 image 字节。

    评测启动时调用一次, 把定位符 + 元数据(gt/w/h/...)装进内存; 图片字节始终惰性。
    列名走 alias() 适配, 只读这套 schema 真实存在的列。
    """
    detect_schema(shard)
    want = ["sample_id", "cohort", "dataset", "origin", "group",
            "gt_otsl", "gt_html", "gt_format_native", "metrics",
            "img_format", "has_image", "img_w", "img_h", "megapixels",
            "res_bucket", "has_numeric", "n_rows", "n_cols", "n_cells",
            "otsl_roundtrip_ok", "gt_conflict", "structure_only"]
    cols = [alias(c) for c in want]
    have = set(pq.ParquetFile(os.path.join(root or PACK_ROOT, shard))
               .schema_arrow.names)
    cols = [c for c in dict.fromkeys(cols) if c in have]

    out = []
    for rg, r in iter_rows(shard, root, columns=cols):
        # 归一到旧字段名, 上层就不用分叉
        back = {v: k for k, v in _ALIAS.items()}
        m = {back.get(k, k): v for k, v in r.items() if k != "sample_id"}
        w, h = m.get("img_w") or 0, m.get("img_h") or 0
        m["img_w"], m["img_h"] = w, h
        m.setdefault("megapixels", round(w * h / 1e6, 4))
        m.setdefault("has_image", True)
        if not m.get("gt_format_native"):
            m["gt_format_native"] = "otsl" if m.get("gt_otsl") else "html"
        out.append((rg, len(out), m, r.get("sample_id") or ""))
    return out