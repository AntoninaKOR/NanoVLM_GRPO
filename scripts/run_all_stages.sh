#!/usr/bin/env bash
# Master training pipeline: SFT baseline → GRPO (action) → GRPO (text+action) → Evaluation
# Runs all training and evaluation stages sequentially
set -euo pipefail

cd "$(dirname "$0")/.."

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     NanoVLM GRPO Training Pipeline - All Stages            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "This script runs:"
echo "  1. SFT Baseline Training"
echo "  2. GRPO Training (Direct Action)"
echo "  3. GRPO Training (Text+Action)"
echo "  4. Comprehensive Evaluation"
echo ""
echo "Total training time: ~4-6 hours (depending on GPU)"
echo ""

# Allow user to skip stages
SKIP_SFT=${SKIP_SFT:-0}
SKIP_GRPO_ACTION=${SKIP_GRPO_ACTION:-0}
SKIP_GRPO_TEXT=${SKIP_GRPO_TEXT:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# Stage 1: SFT Baseline
if [ $SKIP_SFT -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "STAGE 1: Training SFT Baseline"
    echo "════════════════════════════════════════════════════════════"
    bash scripts/train_sft_baseline.sh
    echo ""
else
    echo "[SKIPPED] Stage 1: SFT Baseline"
    echo ""
fi

# Stage 2a: GRPO Action
if [ $SKIP_GRPO_ACTION -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "STAGE 2a: Training GRPO (Direct Action)"
    echo "════════════════════════════════════════════════════════════"
    bash scripts/train_grpo_action.sh
    echo ""
else
    echo "[SKIPPED] Stage 2a: GRPO (Action)"
    echo ""
fi

# Stage 2b: GRPO Text+Action
if [ $SKIP_GRPO_TEXT -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "STAGE 2b: Training GRPO (Text+Action)"
    echo "════════════════════════════════════════════════════════════"
    bash scripts/train_grpo_text_action.sh
    echo ""
else
    echo "[SKIPPED] Stage 2b: GRPO (Text+Action)"
    echo ""
fi

# Stage 3: Comprehensive Evaluation
if [ $SKIP_EVAL -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "STAGE 3: Comprehensive Evaluation"
    echo "════════════════════════════════════════════════════════════"
    bash scripts/eval_grpo.sh
    echo ""
else
    echo "[SKIPPED] Stage 3: Evaluation"
    echo ""
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              All Stages Complete! 🎉                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Training Results:"
echo "  SFT Baseline:        nanovlm/checkpoints/sft_baseline/"
echo "  GRPO (Action):       nanovlm/checkpoints/grpo_action/"
echo "  GRPO (Text+Action):  nanovlm/checkpoints/grpo_text_action/"
echo "  Evaluation Results:  nanovlm/eval_results/"
echo ""
echo "View evaluation results:"
echo "  cat nanovlm/eval_results/evaluation_results.json"
echo ""
echo "View episode visualizations:"
echo "  ls -la nanovlm/eval_results/*_episodes/"
echo ""
