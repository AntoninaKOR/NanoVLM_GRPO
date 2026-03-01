#!/usr/bin/env bash
# Quick test script for GRPO training with minimal settings
set -euo pipefail

cd "$(dirname "$0")/.."

# Check if checkpoint exists, otherwise train SFT first
if [ ! -f "nanovlm/checkpoints/test_training/best_model/config.json" ]; then
    echo "SFT checkpoint not found. Training SFT baseline first..."
    python -m nanovlm.main \
        --dataset nanovlm/data/minigrid_small/dataset.jsonl \
        --mode action \
        --epochs 2 \
        --batch-size 1 \
        --lr 1e-4 \
        --output-dir nanovlm/checkpoints/test_training
fi

echo "Training GRPO model..."
python -m nanovlm.main \
    --method grpo \
    --checkpoint nanovlm/checkpoints/test_training/best_model \
    --mode action \
    --batch-size 1 \
    --lr 1e-4 \
    --num-episodes 4 \
    --num-updates 2 \
    --num-trajectory-batch 2 \
    --kl-beta 0.1 \
    --entropy-weight 0.01 \
    --output-dir nanovlm/checkpoints/test_grpo

echo "GRPO training complete. Results saved to nanovlm/checkpoints/test_grpo"
