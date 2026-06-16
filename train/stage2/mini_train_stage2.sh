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

# Ensure user is logged into Weights & Biases before starting
wandb login

echo "============================================================"
echo " Starting ThinkFlow-VLA Stage 2 Training"
echo " Dataset Subset  : 100%"
echo " Total Steps     : 6000"
echo " Warmup Steps    : 4000"
echo "============================================================"

# Create the output directory for this test run
OUTPUT_DIR="checkpoints/stage2_1"
mkdir -p "$OUTPUT_DIR"

python training/train_stage2.py \
    --stage1_ckpt "$STAGE1_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --split "test" \
    --subset_ratio 1.0 \
    --total_steps 6000 \
    --warmup_steps 4000 \
    --save_steps 100 \
    --wandb_run "stage2-mini-train" \
    --batch_size 2 \
    --grad_accum_steps 1 \
    --log_steps 1 \
    --max_seq_len 4096 \
    --num_workers 2 2>&1 | tee "$OUTPUT_DIR/training_log_full.txt"

echo "Mini training complete! Checkpoints saved to $OUTPUT_DIR"
