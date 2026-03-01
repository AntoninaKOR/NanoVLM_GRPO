#!/usr/bin/env bash
# Quick test script for model training with minimal settings
set -euo pipefail

cd "$(dirname "$0")/.."

python -m nanovlm.main \
    --dataset nanovlm/data/minigrid_small/dataset.jsonl \
    --mode action \
    --epochs 2 \
    --batch-size 1 \
    --lr 1e-4 \
    --output-dir nanovlm/checkpoints/test_training

echo "Test training complete. Results saved to nanovlm/checkpoints/test_training"
