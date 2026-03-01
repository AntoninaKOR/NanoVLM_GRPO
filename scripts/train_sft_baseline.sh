#!/usr/bin/env bash
# SFT (Supervised Fine-Tuning) baseline training
# Stage 1: Train NanoVLM to predict actions from observations using imitation learning
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Stage 1: SFT Baseline Training"
echo "=========================================="
echo ""
echo "This script trains NanoVLM to predict actions"
echo "using supervised learning on expert trajectories."
echo ""

python -m nanovlm.main \
    --method sft \
    --dataset nanovlm/data/minigrid_small/dataset.jsonl \
    --val-split 0.2 \
    --mode action \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --epochs 10 \
    --batch-size 8 \
    --lr 5e-5 \
    --warmup-steps 100 \
    --output-dir nanovlm/checkpoints/sft_baseline \
    --save-interval 1 \
    --seed 42 \
    --device auto

echo ""
echo "=========================================="
echo "SFT Training Complete!"
echo "=========================================="
echo "Checkpoint saved to: nanovlm/checkpoints/sft_baseline/best_model"
echo "Metrics saved to: nanovlm/checkpoints/sft_baseline/metrics.json"
echo ""
echo "Next steps:"
echo "  1. For GRPO (action mode): bash scripts/train_grpo_action.sh"
echo "  2. For GRPO (text+action): bash scripts/train_grpo_text_action.sh"
