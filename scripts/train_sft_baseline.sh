#!/usr/bin/env bash
# Fine-tune NanoVLM SFT baseline on MiniGrid action prediction
set -euo pipefail

cd "$(dirname "$0")/.."

python nanovlm/main.py \
    --config nanovlm/configs.yaml \
    --mode action \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-4 \
    --dataset nanovlm/data/minigrid_sft/dataset.jsonl \
    --output-dir nanovlm/checkpoints/sft_baseline
