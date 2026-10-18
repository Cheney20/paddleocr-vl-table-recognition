#!/bin/bash
# 训练结束后: 评测 orig 基线 + 每个 checkpoint, 选 primary 指标最高者
#
# 判据(用户 2026-10-18): 全量 primary 指标最高者胜;
#   第二判据 = 超过 orig 的来源数
# primary = 空 cell 表 TEDS, 其余 TEDS-S (canonical_teds.primary_metric)
set -Eeuo pipefail

V5=/autodl-fs/data/povl_v5
PACKS=/autodl-tmp/reshard
PY=/root/miniconda3/bin/python3
export PYTHONPATH=/autodl-fs/data/envs/vllm_pkgs
export POVL_VLLM=$(command -v vllm || echo /autodl-fs/data/envs/vllm_env/bin/vllm)
export HF_HOME=/autodl-fs/data/hf_cache
export TMPDIR=/autodl-fs/data/tmp
mkdir -p "$HF_HOME" "$TMPDIR" "$V5/eval/runs"

per_gpu=4
score_workers=32

echo "=== 冒烟: 32 条, 1 卡 1 worker ==="
cd "$V5"
timeout 1800 $PY src/eval_mix.py \
    --model /autodl-fs/data/PaddleOCR-VL-1.6 \
    --tag smoke --packs "$PACKS" --cohorts default \
    --gpus 0 --per-gpu 1 --gmu 0.15 --score-workers 16 \
    --span-subset /autodl-tmp/span15k --limit 32 \
    2>&1 | tail -20

echo "=== orig 基线 (全量) ==="
$PY src/eval_mix.py --model /autodl-fs/data/PaddleOCR-VL-1.6 \
    --tag orig_full --packs "$PACKS" --cohorts default \
    --gpus 0 1 2 3 --per-gpu $per_gpu --score-workers $score_workers \
    --span-subset /autodl-tmp/span15k \
    2>&1 | tail -30

echo "=== 各 checkpoint (全量) ==="
for ck in $(ls -d "$V5"/ckpt/sft_v7b/v0-*/checkpoint-* | sort -t- -k2 -n); do
    name=$(basename "$(dirname "$ck")")_$(basename "$ck")
    echo "--- $name ---"
    $PY src/eval_mix.py --model "$ck" \
        --tag "$name" --packs "$PACKS" --cohorts default \
        --gpus 0 1 2 3 --per-gpu $per_gpu --score-workers $score_workers \
        --span-subset /autodl-tmp/span15k \
        2>&1 | tail -30
done

echo "=== 汇总 ==="
$PY /root/pick_best_ckpt.py
