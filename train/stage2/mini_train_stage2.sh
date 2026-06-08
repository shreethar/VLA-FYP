#!/bin/bash
# mini_train_stage2.sh
# --------------------
# Runs a mini training session on 15% of the dataset for exactly 100 steps:
#   - 50 steps Warmup (Verbalizer LM loss)
#   - 50 steps Phase 2 (Frozen Verbalizer DPO loss)
#
# Usage: ./mini_train_stage2.sh

set -e

# Replace with the actual path to your Stage 1 checkpoint
STAGE1_CKPT="shreethar/stage1_unsloth"

echo "============================================================"
echo " Starting ThinkFlow-VLA Stage 2 Mini Training"
echo " Dataset Subset  : 15%"
echo " Total Steps     : 100"
echo " Warmup Steps    : 50"
echo "============================================================"

# Create the output directory for this test run
OUTPUT_DIR="checkpoints/stage2_mini"
mkdir -p "$OUTPUT_DIR"

python training/train_stage2.py \
    --stage1_ckpt "$STAGE1_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --split "test" \
    --subset_ratio 1.0 \
    --total_steps 100 \
    --warmup_steps 50 \
    --save_steps 50 \
    --wandb_run "stage2-mini-test" \
    --batch_size 4 \
    --num_workers 2

echo "Mini training complete! Checkpoints saved to $OUTPUT_DIR"
