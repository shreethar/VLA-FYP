"""
train_stage2.py
---------------
Main training loop for ThinkFlow-VLA Stage 2:
    Teacher-Student Distillation + GRPO + Spatial Forcing

Execution order per step (mirrors Algorithm 1):
    A. Pre-step:   Extract frozen extractor reference features (once per batch)
    B. Teacher:    GRPO rollouts → score → update → extract h_T
                   → populates RolloutBuffer
    C. Student:
       Warm-up (0–3000):
           C1. Verbalizer LM loss (z detached) → verbalizer_optimizer.step()
           C2. L_distill + L_ans + L_spatial   → student_optimizer.step()
       Frozen (3000–4500):
           C1. L_verb + L_distill + L_ans + L_spatial → student_optimizer.step()
    D. Logging + checkpointing

Three separate optimizers:
    teacher_optimizer   : AdamW, LR 1e-4, Teacher LoRA params only
    student_optimizer   : AdamW, LR 2e-4, Student LoRA + spatial_tokens + spatial_mlp
    verbalizer_optimizer: AdamW, LR 1e-4, Verbalizer CA blocks + LoRA (frozen at step 3000)

Checkpointing saves all three models independently every save_steps steps
and always at the final step.
"""

import os
import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from models.latent_student  import LatentStudent
from models.verbalizer       import Verbalizer
from training.grpo_teacher   import GRPOTeacher, RolloutBuffer
from training.student_losses import StudentLossComputer, build_student_loss_computer
from tokenizer_setup         import load_answer_token_id


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Stage2Config:
    # Paths
    base_model_name:     str  = "Qwen/Qwen3.5-4B"
    verbalizer_name:     str  = "Qwen/Qwen3.5-0.8B"
    stage1_ckpt_dir:     str  = "checkpoints/stage1"           # shared Teacher/Student init
    output_dir:          str  = "checkpoints/stage2"

    # Training schedule
    total_steps:         int  = 4500
    warmup_steps:        int  = 3000                            # Verbalizer LM warm-up
    lr_warmup_steps:     int  = 200                             # LR scheduler warm-up
    save_steps:          int  = 500
    log_steps:           int  = 10
    grad_clip:           float = 1.0

    # Optimizers
    teacher_lr:          float = 1e-4
    student_lr:          float = 2e-4
    verbalizer_lr:       float = 1e-4
    weight_decay:        float = 0.01

    # LoRA
    lora_rank:           int  = 64
    lora_alpha:          int  = 128
    verbalizer_lora_rank: int = 32

    # GRPO
    G:                   int  = 5
    gen_temperature:     float = 0.9
    gen_max_new_tokens:  int  = 512
    kl_coef:             float = 0.0

    # Architecture
    M:                   int  = 6     # reasoning latents
    K:                   int  = 5     # spatial tokens / waypoints

    # Loss weights
    lambda_distill:      float = 1.0
    lambda_ans:          float = 1.0

    # Misc
    seed:                int  = 42
    bf16:                bool = True
    grad_log_steps:      int  = 100   # how often to log gradient norms


# ---------------------------------------------------------------------------
# Optimizer builders
# ---------------------------------------------------------------------------

def build_teacher_optimizer(teacher: GRPOTeacher, cfg: Stage2Config):
    params = [p for p in teacher.vlm.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.teacher_lr, weight_decay=cfg.weight_decay)


def build_student_optimizer(student: LatentStudent, cfg: Stage2Config):
    param_groups = [
        # LoRA params — standard LR
        {
            "params": [
                p for n, p in student.vlm.named_parameters()
                if p.requires_grad and "lora_" in n
            ],
            "lr": cfg.student_lr,
        },
        # Spatial tokens + SpatialMLP — same LR
        {
            "params": list(student.spatial_mlp.parameters()) + [student.spatial_tokens],
            "lr": cfg.student_lr,
        },
    ]
    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def build_verbalizer_optimizer(verbalizer: Verbalizer, cfg: Stage2Config):
    param_groups = [
        # CA blocks — full LR
        {
            "params": list(verbalizer.ca_blocks.parameters()),
            "lr": cfg.verbalizer_lr,
        },
        # LoRA on base LM — same LR
        {
            "params": [
                p for n, p in verbalizer.lm.named_parameters()
                if p.requires_grad and "lora_" in n
            ],
            "lr": cfg.verbalizer_lr,
        },
    ]
    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


# ---------------------------------------------------------------------------
# Scheduler builder (cosine with warm-up, shared pattern for all three)
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, cfg: Stage2Config, total_steps: int):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=total_steps,
    )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    step: int,
    teacher: GRPOTeacher,
    student: LatentStudent,
    verbalizer: Verbalizer,
    teacher_opt, student_opt, verbalizer_opt,
    teacher_sched, student_sched, verbalizer_sched,
    output_dir: str,
):
    ckpt_dir = os.path.join(output_dir, f"step_{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save LoRA adapters only (not frozen base weights) — saves disk space
    teacher.vlm.save_pretrained(os.path.join(ckpt_dir, "teacher_lora"))
    student.vlm.save_pretrained(os.path.join(ckpt_dir, "student_lora"))
    verbalizer.lm.save_pretrained(os.path.join(ckpt_dir, "verbalizer_lora"))

    # Save non-LoRA trainable components
    torch.save(
        {
            "step": step,
            # Student extras
            "spatial_tokens": student.spatial_tokens.data,
            "spatial_mlp":    student.spatial_mlp.state_dict(),
            # Verbalizer CA blocks
            "ca_blocks":      verbalizer.ca_blocks.state_dict(),
            # Optimizers
            "teacher_opt":    teacher_opt.state_dict(),
            "student_opt":    student_opt.state_dict(),
            "verbalizer_opt": verbalizer_opt.state_dict(),
            # Schedulers
            "teacher_sched":    teacher_sched.state_dict(),
            "student_sched":    student_sched.state_dict(),
            "verbalizer_sched": verbalizer_sched.state_dict(),
        },
        os.path.join(ckpt_dir, "training_state.pt"),
    )
    logger.info(f"Checkpoint saved → {ckpt_dir}")


def load_checkpoint(
    ckpt_dir: str,
    teacher: GRPOTeacher,
    student: LatentStudent,
    verbalizer: Verbalizer,
    teacher_opt, student_opt, verbalizer_opt,
    teacher_sched, student_sched, verbalizer_sched,
    device: torch.device,
) -> int:
    """Returns the step number to resume from."""
    from peft import PeftModel

    state = torch.load(
        os.path.join(ckpt_dir, "training_state.pt"), map_location=device
    )
    step = state["step"]

    # Restore non-LoRA components
    student.spatial_tokens.data.copy_(state["spatial_tokens"])
    student.spatial_mlp.load_state_dict(state["spatial_mlp"])
    verbalizer.ca_blocks.load_state_dict(state["ca_blocks"])

    # Restore optimizer and scheduler states
    teacher_opt.load_state_dict(state["teacher_opt"])
    student_opt.load_state_dict(state["student_opt"])
    verbalizer_opt.load_state_dict(state["verbalizer_opt"])
    teacher_sched.load_state_dict(state["teacher_sched"])
    student_sched.load_state_dict(state["student_sched"])
    verbalizer_sched.load_state_dict(state["verbalizer_sched"])

    logger.info(f"Resumed from checkpoint at step {step}")
    return step


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_stage2(
    cfg: Stage2Config,
    dataloader: DataLoader,
    reward_fns,                    # List[RewardFunction] — injected from rewards/
    reward_weights=None,
    resume_from: Optional[str] = None,
    answer_token_id: int = -1,     # set after tokenizer extension
):
    """
    Full Stage 2 training loop.

    Parameters
    ----------
    dataloader    : yields batches with keys:
                        input_ids, pixel_values, image_grid_thw, attention_mask,
                        gt_waypoints  [batch, K, 2],
                        pixel_values_extractor  [batch, C, H, W]  (for DINOv2/VGGT),
                        ground_truth  dict  (task-specific GT for reward functions)
    reward_fns    : list of RewardFunction callables from rewards/
    reward_weights: per-function weights (uniform if None)
    resume_from   : checkpoint directory path to resume from
    answer_token_id: token id of <ans> special token
    """
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)

    # ------------------------------------------------------------------
    # 2. Build models
    # ------------------------------------------------------------------
    logger.info("Building Teacher …")
    teacher = GRPOTeacher(
        model_name=cfg.base_model_name,
        G=cfg.G,
        answer_token_id=answer_token_id,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        gen_temperature=cfg.gen_temperature,
        gen_max_new_tokens=cfg.gen_max_new_tokens,
        kl_coef=cfg.kl_coef,
    ).to(device)

    logger.info("Building Student …")
    student = LatentStudent(
        model_name=cfg.base_model_name,
        M=cfg.M,
        K=cfg.K,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
    ).to(device)

    logger.info("Building Verbalizer …")
    verbalizer = Verbalizer(
        model_name=cfg.verbalizer_name,
        student_hidden=student.hidden_dim,
        lora_rank=cfg.verbalizer_lora_rank,
        lora_alpha=cfg.verbalizer_lora_rank * 2,
    ).to(device)

    # Load Stage 1 checkpoint into both Teacher and Student
    # (identical init — they diverge from here via their respective objectives)
    if cfg.stage1_ckpt_dir and os.path.isdir(cfg.stage1_ckpt_dir):
        logger.info(f"Loading Stage 1 checkpoint: {cfg.stage1_ckpt_dir}")
        from peft import set_peft_model_state_dict
        import safetensors.torch as st

        s1_state = st.load_file(
            os.path.join(cfg.stage1_ckpt_dir, "adapter_model.safetensors")
        )
        set_peft_model_state_dict(teacher.vlm, s1_state)
        set_peft_model_state_dict(student.vlm, s1_state)
        logger.info("Stage 1 weights loaded into Teacher and Student.")

    # ------------------------------------------------------------------
    # 3. Loss computer
    # ------------------------------------------------------------------
    loss_computer = build_student_loss_computer(
        warmup_steps=cfg.warmup_steps,
        lambda_distill=cfg.lambda_distill,
        lambda_ans=cfg.lambda_ans,
    )

    # ------------------------------------------------------------------
    # 4. Optimizers + schedulers
    # ------------------------------------------------------------------
    teacher_opt    = build_teacher_optimizer(teacher, cfg)
    student_opt    = build_student_optimizer(student, cfg)
    verbalizer_opt = build_verbalizer_optimizer(verbalizer, cfg)

    teacher_sched    = build_scheduler(teacher_opt,    cfg, cfg.total_steps)
    student_sched    = build_scheduler(student_opt,    cfg, cfg.total_steps)
    verbalizer_sched = build_scheduler(verbalizer_opt, cfg, cfg.warmup_steps)
    # Verbalizer scheduler only runs for warmup_steps; frozen after that

    # ------------------------------------------------------------------
    # 5. Resume from checkpoint if requested
    # ------------------------------------------------------------------
    start_step = 0
    if resume_from and os.path.isdir(resume_from):
        start_step = load_checkpoint(
            resume_from,
            teacher, student, verbalizer,
            teacher_opt, student_opt, verbalizer_opt,
            teacher_sched, student_sched, verbalizer_sched,
            device,
        )
        # Re-apply freeze state if resuming past warm-up
        if start_step >= cfg.warmup_steps and not verbalizer.is_frozen():
            verbalizer.freeze_for_student_training()
            logger.info("Verbalizer re-frozen after checkpoint load.")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    teacher.train()
    student.train()
    verbalizer.train()

    # Infinite dataloader iterator
    data_iter = iter(dataloader)

    for step in range(start_step, cfg.total_steps):

        # ------ Get next batch (cycle dataloader) ---------------------
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        input_ids      = batch["input_ids"].to(device)
        pixel_values   = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_waypoints   = batch["gt_waypoints"].to(device)          # [batch, K, 2]
        ground_truth   = batch["ground_truth"]                      # dict (stays on CPU)

        # ----------------------------------------------------------------
        # B. Teacher GRPO step
        #    Internally: rollouts → score → GRPO backward → step → h_T
        # ----------------------------------------------------------------
        buffer: RolloutBuffer = teacher.training_step(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            ground_truth=ground_truth,
            reward_fns=reward_fns,
            reward_weights=reward_weights,
            optimizer=teacher_opt,
            tokenizer=tokenizer,
            grad_clip=cfg.grad_clip,
        )
        teacher_sched.step()

        # ----------------------------------------------------------------
        # C. Verbalizer freeze transition at warmup_steps
        # ----------------------------------------------------------------
        if step == cfg.warmup_steps and not verbalizer.is_frozen():
            verbalizer.freeze_for_student_training()
            logger.info(f"[Step {step}] Verbalizer frozen — DPO phase begins.")

        # ----------------------------------------------------------------
        # D. Compute all Student (and optionally Verbalizer) losses
        # ----------------------------------------------------------------
        loss_out = loss_computer.compute(
            student=student,
            verbalizer=verbalizer,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            buffer=buffer,
            gt_waypoints=gt_waypoints,
            global_step=step,
        )

        # ----------------------------------------------------------------
        # E. Backward passes — phase dependent
        # ----------------------------------------------------------------
        is_warmup = (step < cfg.warmup_steps)

        if is_warmup:
            # --- E1. Verbalizer backward (LM loss on τ+, z detached) ----
            verbalizer_opt.zero_grad()
            loss_out.lm_loss.backward()
            nn.utils.clip_grad_norm_(verbalizer.parameters(), cfg.grad_clip)
            verbalizer_opt.step()
            verbalizer_sched.step()

            # --- E2. Student backward (distill + ans + spatial) ----------
            student_opt.zero_grad()
            loss_out.student_total.backward()
            nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad],
                cfg.grad_clip,
            )
            student_opt.step()
            student_sched.step()

        else:
            # --- E3. Student backward (verb + distill + ans + spatial) ---
            # Verbalizer is frozen; DPO gradient flows through CA into Student
            student_opt.zero_grad()
            loss_out.student_total.backward()
            nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad],
                cfg.grad_clip,
            )
            student_opt.step()
            student_sched.step()

        # ----------------------------------------------------------------
        # F. Logging
        # ----------------------------------------------------------------
        if step % cfg.log_steps == 0:
            teacher_stats = GRPOTeacher.log_rollout_stats(buffer)
            log_msg = (
                f"Step {step:>5d}/{cfg.total_steps} | "
                f"student={loss_out.metrics['loss/student_total']:.4f} | "
                f"distill={loss_out.metrics['loss/l_distill']:.4f} | "
                f"ans={loss_out.metrics['loss/l_ans']:.4f} | "
                f"reward_mean={teacher_stats['grpo/reward_mean']:.4f} | "
                f"phase={'warmup' if is_warmup else 'frozen'}"
            )
            if is_warmup:
                log_msg += f" | lm={loss_out.metrics.get('loss/lm_loss', 0):.4f}"
            else:
                log_msg += f" | verb={loss_out.metrics.get('loss/l_verb', 0):.4f}"
            logger.info(log_msg)

        # Gradient norm logging (less frequent)
        if step % cfg.grad_log_steps == 0:
            from training.student_losses import StudentLossComputer
            grad_norms = StudentLossComputer.log_student_grad_norms(student)
            logger.info(
                f"  grad_norm/lora={grad_norms.get('grad_norm/lora_total', 0):.4f} | "
                f"  grad_norm/spatial={grad_norms.get('grad_norm/spatial_total', 0):.4f}"
            )

        # ----------------------------------------------------------------
        # G. Checkpointing
        # ----------------------------------------------------------------
        if (step > 0 and step % cfg.save_steps == 0) or step == cfg.total_steps - 1:
            save_checkpoint(
                step=step,
                teacher=teacher,
                student=student,
                verbalizer=verbalizer,
                teacher_opt=teacher_opt,
                student_opt=student_opt,
                verbalizer_opt=verbalizer_opt,
                teacher_sched=teacher_sched,
                student_sched=student_sched,
                verbalizer_sched=verbalizer_sched,
                output_dir=cfg.output_dir,
            )

    logger.info("Stage 2 training complete.")
    return student, teacher, verbalizer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt",   type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/",
                        help="Dir produced by tokenizer_setup.py")
    parser.add_argument("--output_dir",    type=str, default="checkpoints/stage2")
    parser.add_argument("--resume_from",   type=str, default=None)
    parser.add_argument("--total_steps",   type=int, default=4500)
    args = parser.parse_args()

    # Auto-load answer_token_id from tokenizer config
    answer_token_id = load_answer_token_id(args.tokenizer_dir)
    logger.info(f"answer_token_id = {answer_token_id}")

    cfg = Stage2Config(
        stage1_ckpt_dir=args.stage1_ckpt,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
    )

    from rewards.action_reward import ActionAlignedReward, CombinedActionReward
    from rewards.qa_reward     import FormatReward
    reward_fns     = [CombinedActionReward(ActionAlignedReward(), FormatReward())]
    reward_weights = [1.0]   # 0.9/0.1 split baked into CombinedActionReward

    # Dataloader is user-provided — plug your LeRobot StreamingDataset here
    # dataloader = build_stage2_dataloader(cfg)

    # train_stage2(
    #     cfg=cfg,
    #     dataloader=dataloader,
    #     reward_fns=reward_fns,
    #     reward_weights=reward_weights,
    #     resume_from=args.resume_from,
    #     answer_token_id=args.answer_token_id,
    # )