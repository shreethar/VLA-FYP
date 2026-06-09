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

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.latent_student  import LatentStudent
from models.verbalizer       import Verbalizer
from training.updated_grpo_teacher   import GRPOTeacher, RolloutBuffer
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
    base_model_name:     str  = "shreethar/stage1_unsloth"
    verbalizer_name:     str  = "unsloth/Qwen3.5-0.8B"
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

    # WandB
    wandb_project:       str  = "reasonflow-vla"
    wandb_run_name:      str  = "stage2-distillation"
    wandb_tags:          List[str] = field(default_factory=lambda: ["stage2", "grpo", "distillation"])
    wandb_log_steps:     int  = 10   # same as log_steps by default
    use_wandb:           bool = True


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
    # 0. WandB initialisation
    # ------------------------------------------------------------------
    use_wandb = cfg.use_wandb and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            tags=cfg.wandb_tags,
            config={
                # Model
                "base_model":        cfg.base_model_name,
                "verbalizer_model":  cfg.verbalizer_name,
                # Schedule
                "total_steps":       cfg.total_steps,
                "warmup_steps":      cfg.warmup_steps,
                "save_steps":        cfg.save_steps,
                # LR
                "teacher_lr":        cfg.teacher_lr,
                "student_lr":        cfg.student_lr,
                "verbalizer_lr":     cfg.verbalizer_lr,
                "weight_decay":      cfg.weight_decay,
                # LoRA
                "lora_rank":         cfg.lora_rank,
                "lora_alpha":        cfg.lora_alpha,
                "verb_lora_rank":    cfg.verbalizer_lora_rank,
                # GRPO
                "G":                 cfg.G,
                "gen_temperature":   cfg.gen_temperature,
                "gen_max_tokens":    cfg.gen_max_new_tokens,
                "kl_coef":           cfg.kl_coef,
                # Architecture
                "M_latents":         cfg.M,
                "K_spatial":         cfg.K,
                # Loss weights
                "lambda_distill":    cfg.lambda_distill,
                "lambda_ans":        cfg.lambda_ans,
            },
            resume="allow",
        )
        logger.info(f"WandB run: {wandb.run.url}")
    elif cfg.use_wandb and not _WANDB_AVAILABLE:
        logger.warning("WandB requested but not installed. Run: pip install wandb")

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)

    # ------------------------------------------------------------------
    # 2. Build models
    # ------------------------------------------------------------------
    logger.info("Building Teacher …")
    teacher = GRPOTeacher(
        pretrained_model_name_or_path=cfg.base_model_name,
        G=cfg.G,
        answer_token_id=answer_token_id, # This is now the think_end_token_id under the hood
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
        end_think_token_id=answer_token_id,
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
        # F. Logging  (console + WandB)
        # ----------------------------------------------------------------
        if step % cfg.log_steps == 0:
            teacher_stats = GRPOTeacher.log_rollout_stats(buffer)
            m = loss_out.metrics   # shorthand

            log_msg = (
                f"Step {step:>5d}/{cfg.total_steps} | "
                f"student={m['loss/student_total']:.4f} | "
                f"distill={m['loss/l_distill']:.4f} | "
                f"ans={m['loss/l_ans']:.4f} | "
                f"reward_mean={teacher_stats['grpo/reward_mean']:.4f} | "
                f"phase={'warmup' if is_warmup else 'frozen'}"
            )
            if is_warmup:
                log_msg += f" | lm={m.get('loss/lm_loss', 0):.4f}"
            else:
                log_msg += f" | verb={m.get('loss/l_verb', 0):.4f}"
            logger.info(log_msg)

            if use_wandb:
                wandb_payload = {
                    # ── Phase ───────────────────────────────────────────
                    "phase/is_warmup":          float(is_warmup),
                    "phase/step":               step,

                    # ── Student losses ───────────────────────────────────
                    "loss/student_total":        m["loss/student_total"],
                    "loss/l_distill":            m["loss/l_distill"],
                    "loss/l_ans":               m["loss/l_ans"],
                    "loss/lm_loss":             m.get("loss/lm_loss", 0.0),
                    "loss/l_verb":              m.get("loss/l_verb", 0.0),

                    # ── Teacher / GRPO ───────────────────────────────────
                    "teacher/reward_mean":       teacher_stats["grpo/reward_mean"],
                    "teacher/reward_max":        teacher_stats["grpo/reward_max"],
                    "teacher/reward_min":        teacher_stats["grpo/reward_min"],
                    "teacher/reward_std":        teacher_stats["grpo/reward_std"],
                    "teacher/advantage_mean":    teacher_stats["grpo/advantage_mean"],

                    # ── DPO (frozen phase only) ──────────────────────────
                    "dpo/loss":                 m.get("dpo/dpo_loss",      0.0),
                    "dpo/reward_margin":        m.get("dpo/reward_margin", 0.0),
                    "dpo/accuracy":             m.get("dpo/dpo_accuracy",  0.0),
                    "dpo/log_pi_pos":           m.get("dpo/log_pi_pos",    0.0),
                    "dpo/log_pi_neg":           m.get("dpo/log_pi_neg",    0.0),

                    # ── Distillation alignment ───────────────────────────
                    "distill/h_S_norm":         m.get("distill/h_S_norm",    0.0),
                    "distill/h_T_norm":         m.get("distill/h_T_norm",    0.0),
                    "distill/cosine_sim":       m.get("distill/cosine_sim",  0.0),

                    # ── Waypoints ────────────────────────────────────────
                    "waypoints/pred_mean":      m.get("waypoints/pred_mean", 0.0),
                    "waypoints/pred_std":       m.get("waypoints/pred_std",  0.0),

                    # ── Learning rates ───────────────────────────────────
                    "lr/teacher":               teacher_sched.get_last_lr()[0],
                    "lr/student":               student_sched.get_last_lr()[0],
                    "lr/verbalizer":            (verbalizer_sched.get_last_lr()[0]
                                                 if not verbalizer.is_frozen() else 0.0),
                }
                wandb.log(wandb_payload, step=step)

                # ── Rollout Text Logging (every 10 steps) ────────────────
                if step % 10 == 0:
                    try:
                        import wandb
                        table = wandb.Table(columns=["Batch_Idx", "Rollout_Idx", "Reward", "Advantage", "Text"])
                        G_len = buffer.rewards.shape[0]
                        B_len = buffer.rewards.shape[1]
                        for b in range(B_len):
                            for g in range(G_len):
                                table.add_data(
                                    b, 
                                    g, 
                                    float(buffer.rewards[g, b].cpu()), 
                                    float(buffer.advantages[g, b].cpu()), 
                                    buffer.rollout_texts[g][b]
                                )
                        wandb.log({"rollouts/generation_samples": table}, step=step)
                    except Exception as e:
                        logger.warning(f"Failed to log wandb rollout table: {e}")

        # Gradient norm logging (less frequent)
        if step % cfg.grad_log_steps == 0:
            from training.student_losses import StudentLossComputer
            grad_norms = StudentLossComputer.log_student_grad_norms(student)
            logger.info(
                f"  grad_norm/lora={grad_norms.get('grad_norm/lora_total', 0):.4f} | "
                f"  grad_norm/spatial={grad_norms.get('grad_norm/spatial_total', 0):.4f}"
            )
            if use_wandb:
                wandb.log({
                    "grad/lora_total":    grad_norms.get("grad_norm/lora_total",    0.0),
                    "grad/spatial_total": grad_norms.get("grad_norm/spatial_total", 0.0),
                }, step=step)

        # ----------------------------------------------------------------
        # G. Checkpointing
        # ----------------------------------------------------------------
        is_ckpt_step = (step > 0 and step % cfg.save_steps == 0) or step == cfg.total_steps - 1
        if is_ckpt_step:
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
            if use_wandb:
                # Log checkpoint as a WandB artifact so you can restore any step
                ckpt_dir = os.path.join(cfg.output_dir, f"step_{step:06d}")
                artifact = wandb.Artifact(
                    name=f"stage2-ckpt-step{step}",
                    type="model-checkpoint",
                    description=f"Stage 2 checkpoint at step {step}",
                    metadata={"step": step, "phase": "warmup" if is_warmup else "frozen"},
                )
                artifact.add_dir(ckpt_dir)
                wandb.log_artifact(artifact)

    logger.info("Stage 2 training complete.")
    if use_wandb:
        wandb.finish()
    return student, teacher, verbalizer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Stage 2: GRPO Teacher + Latent Student + Verbalizer")
    # Paths
    parser.add_argument("--stage1_ckpt",   type=str, required=True,
                        help="Path to Stage 1 LoRA adapter dir (adapter_model.safetensors)")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/",
                        help="Dir produced by tokenizer_setup.py (contains thinkflow_token_ids.txt)")
    parser.add_argument("--output_dir",    type=str, default="checkpoints/stage2")
    parser.add_argument("--resume_from",   type=str, default=None,
                        help="Checkpoint dir to resume from (e.g. checkpoints/stage2/step_001000)")
    # Data
    parser.add_argument("--hf_repo",       type=str, default="shreethar/FYP-Stage2-dataset",
                        help="HuggingFace dataset repo with pre-materialised Stage 1 subset")
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--num_workers",   type=int, default=2)
    parser.add_argument("--max_seq_len",   type=int, default=1024)
    parser.add_argument("--split",         type=str, default="train",
                        help="HuggingFace dataset split to use (e.g. train, test)")
    parser.add_argument("--subset_ratio",  type=float, default=1.0,
                        help="Train on a smaller percentage of the dataset (e.g. 0.15 for 15%)")
    # Training schedule
    parser.add_argument("--total_steps",   type=int, default=4500)
    parser.add_argument("--warmup_steps",  type=int, default=3000)
    parser.add_argument("--save_steps",    type=int, default=500)
    # WandB
    parser.add_argument("--wandb_project", type=str, default="reasonflow-vla")
    parser.add_argument("--wandb_run",     type=str, default="stage2-distillation")
    parser.add_argument("--no_wandb",      action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    # ── Config ─────────────────────────────────────────────────────────────
    cfg = Stage2Config(
        stage1_ckpt_dir=args.stage1_ckpt,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        use_wandb=not args.no_wandb,
    )

    # ── Tokenizer ──────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    # Automatically fetch end_think_token_id directly from the tokenizer
    # Since </think> is already in the Qwen3 register as the user specified
    tok = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    think_end_token_id = tok.convert_tokens_to_ids("</think>")
    if think_end_token_id is None or think_end_token_id == tok.unk_token_id:
        think_end_token_id = tok.encode("</think>", add_special_tokens=False)[-1]
    
    answer_token_id = think_end_token_id
    logger.info(f"Dynamically fetched </think> token ID for distillation target: {answer_token_id}")

    # ── Processor & Dataloader ─────────────────────────────────────────────
    # Import here so the module can be used without these heavy deps
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transformers import AutoProcessor
    from stage2_dataloader import build_stage2_dataloader

    logger.info(f"Loading processor from {cfg.base_model_name} …")
    processor = AutoProcessor.from_pretrained(cfg.base_model_name, trust_remote_code=True)

    dataloader = build_stage2_dataloader(
        processor=processor,
        hf_repo=args.hf_repo,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_seq_len,
        subset_ratio=args.subset_ratio,
    )
    logger.info(f"DataLoader ready: {len(dataloader.dataset):,} samples.")

    # ── Reward functions ───────────────────────────────────────────────────
    from rewards.action_reward import ActionAlignedReward, CombinedActionReward
    from rewards.qa_reward     import FormatReward
    reward_fns     = [CombinedActionReward(ActionAlignedReward(), FormatReward())]
    reward_weights = [1.0]   # 0.9/0.1 split baked into CombinedActionReward

    # ── Train ──────────────────────────────────────────────────────────────
    train_stage2(
        cfg=cfg,
        dataloader=dataloader,
        reward_fns=reward_fns,
        reward_weights=reward_weights,
        resume_from=args.resume_from,
        answer_token_id=answer_token_id,
    )