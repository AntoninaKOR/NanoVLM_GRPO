#!/bin/bash
# Complete GRPO training pipeline

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GRPO_EPISODES="${GRPO_EPISODES:-50}"
GRPO_UPDATES="${GRPO_UPDATES:-100}"
GRPO_LR="${GRPO_LR:-1e-5}"
GRPO_KL="${GRPO_KL:-0.1}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}GRPO Complete Training Pipeline${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"

# Step 1: Check SFT baseline
echo -e "\n${YELLOW}Step 1: Checking SFT baseline${NC}"
if [ -d "nanovlm/checkpoints/sft_baseline" ]; then
    echo -e "${GREEN}✓ SFT baseline found${NC}"
    SFT_READY=1
else
    echo -e "${RED}✗ SFT baseline not found${NC}"
    echo "Please train SFT baseline first:"
    echo "  python nanovlm/main.py --mode action --output-dir nanovlm/checkpoints/sft_baseline"
    SFT_READY=0
fi

if [ $SFT_READY -eq 0 ]; then
    exit 1
fi

# Step 2: Train Direct Action GRPO
echo -e "\n${YELLOW}Step 2: Training Direct Action GRPO${NC}"
echo -e "Configuration:"
echo -e "  Episodes per update: $GRPO_EPISODES"
echo -e "  Number of updates: $GRPO_UPDATES"
echo -e "  Learning rate: $GRPO_LR"
echo -e "  KL coefficient: $GRPO_KL"

CHECKPOINT_DIR="nanovlm/checkpoints/sft_baseline" \
OUTPUT_DIR="nanovlm/checkpoints/grpo_action" \
NUM_EPISODES="$GRPO_EPISODES" \
NUM_UPDATES="$GRPO_UPDATES" \
LR="$GRPO_LR" \
KL_COEFF="$GRPO_KL" \
DEVICE="$DEVICE" \
SEED="$SEED" \
bash scripts/train_grpo_action.sh

echo -e "${GREEN}✓ Direct Action GRPO training complete${NC}"

# Step 3: Train Text+Action GRPO (plan mode)
echo -e "\n${YELLOW}Step 3: Training Text+Action GRPO (plan mode)${NC}"
echo -e "Configuration:"
echo -e "  Episodes per update: $GRPO_EPISODES"
echo -e "  Number of updates: $GRPO_UPDATES"
echo -e "  Learning rate: $GRPO_LR"
echo -e "  KL coefficient: $GRPO_KL"
echo -e "  Text mode: plan"

CHECKPOINT_DIR="nanovlm/checkpoints/sft_baseline" \
OUTPUT_DIR="nanovlm/checkpoints/grpo_text_action" \
NUM_EPISODES="$GRPO_EPISODES" \
NUM_UPDATES="$GRPO_UPDATES" \
LR="$GRPO_LR" \
KL_COEFF="$GRPO_KL" \
TEXT_MODE="plan" \
DEVICE="$DEVICE" \
SEED="$SEED" \
bash scripts/train_grpo_text_action.sh

echo -e "${GREEN}✓ Text+Action GRPO training complete${NC}"

# Step 4: Evaluate all models
echo -e "\n${YELLOW}Step 4: Evaluating and comparing models${NC}"
echo -e "Configuration:"
echo -e "  Evaluation episodes: $EVAL_EPISODES"

SFT_CHECKPOINT="nanovlm/checkpoints/sft_baseline" \
ACTION_CHECKPOINT="nanovlm/checkpoints/grpo_action" \
TEXT_ACTION_CHECKPOINT="nanovlm/checkpoints/grpo_text_action" \
OUTPUT_DIR="nanovlm/eval_results" \
NUM_EPISODES="$EVAL_EPISODES" \
TRAINING_CURVES="nanovlm/checkpoints/grpo_action/metrics_all.json" \
DEVICE="$DEVICE" \
SEED="$SEED" \
bash scripts/eval_grpo.sh

echo -e "${GREEN}✓ Evaluation complete${NC}"

# Summary
echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Pipeline Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Trained models:"
echo -e "  1. SFT Baseline: ${YELLOW}nanovlm/checkpoints/sft_baseline${NC}"
echo -e "  2. Direct Action GRPO: ${YELLOW}nanovlm/checkpoints/grpo_action${NC}"
echo -e "  3. Text+Action GRPO: ${YELLOW}nanovlm/checkpoints/grpo_text_action${NC}"
echo ""
echo -e "Results:"
echo -e "  - Evaluation metrics: ${YELLOW}nanovlm/eval_results/evaluation_results.json${NC}"
echo -e "  - Comparison plots: ${YELLOW}nanovlm/eval_results/model_comparison.png${NC}"
echo -e "  - Training curves: ${YELLOW}nanovlm/eval_results/training_curves.png${NC}"
echo ""
echo "Next steps:"
echo "  1. View results: open nanovlm/eval_results/"
echo "  2. Experiment with hyperparameters"
echo "  3. Try different text modes: state, reasoning"
echo "  4. Analyze failure cases in checkpoints"
echo ""
