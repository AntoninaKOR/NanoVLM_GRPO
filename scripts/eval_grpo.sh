#!/usr/bin/env bash
# Evaluate and Compare All Trained Models
# Final stage: Comprehensive comparison of SFT vs GRPO (action) vs GRPO (text+action)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Model Comparison & Evaluation"
echo "=========================================="
echo ""
echo "Evaluating three trained models:"
echo "  1. SFT Baseline (imitation learning)"
echo "  2. GRPO (direct action output)"
echo "  3. GRPO (text+action reasoning)"
echo ""

# Check if models exist
if [ ! -d "nanovlm/checkpoints/sft_baseline/best_model" ]; then
    echo "ERROR: SFT baseline not found. Run: bash scripts/train_sft_baseline.sh"
    exit 1
fi

echo "Checkpoints found:"
ls -1d nanovlm/checkpoints/*/best_model 2>/dev/null || echo "  (some models not found - that's ok)"

echo ""
echo "Running comprehensive evaluation..."
echo ""

python -m nanovlm.main \
    --method eval \
    --checkpoints \
        nanovlm/checkpoints/sft_baseline/best_model \
        nanovlm/checkpoints/grpo_action/best_model \
        nanovlm/checkpoints/grpo_text_action/best_model \
    --checkpoint-names \
        "SFT_Baseline" \
        "GRPO_Action" \
        "GRPO_TextAction" \
    --num-episodes 100 \
    --output-dir nanovlm/eval_results \
    --seed 42 \
    --device auto

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: nanovlm/eval_results/evaluation_results.json"
echo "Episode GIFs saved to: nanovlm/eval_results/*_episodes/"
echo ""
echo "To view results:"
echo "  cat nanovlm/eval_results/evaluation_results.json"
