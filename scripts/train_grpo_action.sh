#!/usr/bin/env bash
# GRPO Training with Direct Action Output
# Stage 2a: Train NanoVLM using GRPO to directly predict actions in environment
set -euo pipefail

cd "$(dirname "$0")/.."

# Check if SFT baseline exists
if [ ! -d "nanovlm/checkpoints/sft_baseline/best_model" ]; then
    echo "ERROR: SFT baseline not found at nanovlm/checkpoints/sft_baseline/best_model"
    echo "Please run: bash scripts/train_sft_baseline.sh"
    exit 1
fi

echo "=========================================="
echo "Stage 2a: GRPO Training (Action Mode)"
echo "=========================================="
echo ""
echo "This uses RL to fine-tune the SFT model."
echo "Policy directly outputs actions."
echo ""

python -m nanovlm.main \
    --method grpo \
    --mode action \
    --checkpoint nanovlm/checkpoints/sft_baseline/best_model \
    --num-episodes 50 \
    --num-updates 100 \
    --num-trajectory-batch 32 \
    --batch-size 4 \
    --lr 1e-5 \
    --kl-beta 0.1 \
    --entropy-weight 0.01 \
    --output-dir nanovlm/checkpoints/grpo_action \
    --save-interval 10 \
    --seed 42 \
    --device auto

echo ""
echo "=========================================="
echo "GRPO (Action) Training Complete!"
echo "=========================================="
echo "Checkpoint saved to: nanovlm/checkpoints/grpo_action"
echo "Metrics saved to: nanovlm/checkpoints/grpo_action/metrics_all.json"
echo ""
echo "Next step:"
echo "  For GRPO (text+action): bash scripts/train_grpo_text_action.sh"
