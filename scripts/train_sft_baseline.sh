#!/bin/bash

set -e

echo "=========================================="
echo "Training NanoVLM SFT Baseline"
echo "Using simple prompt (no state description)"
echo "=========================================="

DATASET=${1:-"nanovlm/data/minigrid/dataset.jsonl"}
EPOCHS=${2:-5}
BATCH_SIZE=${3:-8}
LR=${4:-1e-4}
OUTPUT_DIR=${5:-"nanovlm/checkpoints/sft_baseline"}

echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Run training
python -m nanovlm.main \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --output-dir "$OUTPUT_DIR" 

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Learning curves saved to: $OUTPUT_DIR/learning_curves.png"
echo "GIFs saved to: $OUTPUT_DIR/gifs_epoch_*/"
echo "=========================================="
