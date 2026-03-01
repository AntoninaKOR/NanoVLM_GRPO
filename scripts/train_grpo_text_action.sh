#!/usr/bin/env bash
# GRPO Training with Text+Action Output
# Stage 2b: Train NanoVLM using GRPO to generate reasoning text + actions
set -euo pipefail

cd "$(dirname "$0")/.."

# Check if SFT baseline exists
if [ ! -d "nanovlm/checkpoints/sft_baseline/best_model" ]; then
    echo "ERROR: SFT baseline not found at nanovlm/checkpoints/sft_baseline/best_model"
    echo "Please run: bash scripts/train_sft_baseline.sh"
    exit 1
fi

echo "=========================================="
echo "Stage 2b: GRPO Training (Text+Action Mode)"
echo "=========================================="
echo ""
echo "This uses RL with text reasoning."
echo "Policy generates: [state description] + [action]"
echo "Text mode: plan (generates action plans)"
echo ""

python -m nanovlm.main \
    --method grpo \
    --mode text_action \
    --checkpoint nanovlm/checkpoints/sft_baseline/best_model \
    --num-episodes 50 \
    --num-updates 100 \
    --num-trajectory-batch 32 \
    --batch-size 4 \
    --lr 1e-5 \
    --kl-beta 0.1 \
    --entropy-weight 0.01 \
    --text-mode plan \
    --output-dir nanovlm/checkpoints/grpo_text_action \
    --save-interval 10 \
    --seed 42 \
    --device auto

echo ""
echo "=========================================="
echo "GRPO (Text+Action) Training Complete!"
echo "=========================================="
echo "Checkpoint saved to: nanovlm/checkpoints/grpo_text_action"
echo "Metrics saved to: nanovlm/checkpoints/grpo_text_action/metrics_all.json"
