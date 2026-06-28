#!/bin/bash
# mini_train_stage2.sh
# --------------------
# Runs the Decoupled Stage 2 Pipeline:
#   Phase A: Teacher-Only (Generates offline rollouts to disk)
#   Phase B: Student-Offline (Trains Student + Verbalizer on offline rollouts)
#
# Usage: 
#   ./mini_train_stage2.sh teacher   # Runs only the teacher
#   ./mini_train_stage2.sh student   # Runs only the student
#   ./mini_train_stage2.sh all       # Runs both sequentially

set -e

MODE=${1:-"all"}

# Replace with the actual path to your Stage 1 checkpoint
STAGE1_CKPT="shreethar/stage1_unsloth"
OUTPUT_DIR="checkpoints/stage2_decoupled_mini"
OFFLINE_DATA_DIR="$OUTPUT_DIR/offline_data"

# Ensure user is logged into Weights & Biases before starting
wandb login || echo "Proceeding without WandB login..."

echo "============================================================"
echo " Starting ThinkFlow-VLA Stage 2 Training (Decoupled)"
echo " Mode            : $MODE"
echo " Output Dir      : $OUTPUT_DIR"
echo " Total Steps     : 675"
echo " Warmup Steps    : 450"
echo " Batch Size      : 12 (Accumulated 4x = 48 effective)"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

if [[ "$MODE" == "teacher" || "$MODE" == "all" ]]; then
    echo "------------------------------------------------------------"
    echo " PHASE A: TEACHER DATA GENERATION"
    echo "------------------------------------------------------------"
    # We run the teacher to populate the offline data directory.
    # The teacher uses chunking (grpo_backward_batch_size) for speed.
    python3 training/train_stage2.py \
        --stage1_ckpt "$STAGE1_CKPT" \
        --output_dir "$OUTPUT_DIR" \
        --split "test" \
        --subset_ratio 1.0 \
        --total_steps 675 \
        --warmup_steps 450 \
        --save_steps 20 \
        --wandb_project "VLA-FYP-Teacher" \
        --wandb_run "stage2-mini-teacher" \
        --batch_size 12 \
        --G 8 \
        --grpo_backward_batch_size 2 \
        --grad_accum_steps 4 \
        --log_steps 1 \
        --max_seq_len 2560 \
        --offload_ref_model False \
        --num_workers 2 \
        --mode "teacher_only" \
        --offline_data_dir "$OFFLINE_DATA_DIR" 2>&1 | tee "$OUTPUT_DIR/teacher_log.txt"
fi

if [[ "$MODE" == "student" || "$MODE" == "all" ]]; then
    echo "------------------------------------------------------------"
    echo " PHASE B: STUDENT + VERBALIZER TRAINING"
    echo "------------------------------------------------------------"
    # We run the student to train on the generated offline data.
    python3 training/train_stage2.py \
        --stage1_ckpt "$STAGE1_CKPT" \
        --output_dir "$OUTPUT_DIR" \
        --total_steps 600 \
        --warmup_steps 400 \
        --save_steps 20 \
        --wandb_project "VLA-FYP" \
        --wandb_run "stage2-mini-student" \
        --batch_size 8 \
        --G 8 \
        --grad_accum_steps 4 \
        --log_steps 1 \
        --max_seq_len 2048 \
        --mode "student_offline" \
        --offline_data_dir "$OFFLINE_DATA_DIR" 2>&1 | tee "$OUTPUT_DIR/student_log.txt"
fi

echo "============================================================"
echo " Decoupled training phase(s) complete!"
echo "============================================================"
