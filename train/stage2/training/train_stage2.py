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
from training.grpo_teacher   import GRPOTeacher, RolloutBuffer
from training.student_losses import StudentLossComputer, build_student_loss_computer


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
    save_steps:          int  = 100                             # was 500 — more recovery points
    log_steps:           int  = 10
    grad_clip:           float = 1.0
    grad_accum_steps:    int  = 32

    # Optimizers
    teacher_lr:          float = 0.5e-4
    student_lr:          float = 1e-4
    verbalizer_lr:       float = 2.5e-4
    weight_decay:        float = 0.01

    # LoRA
    lora_rank:           int  = 64
    lora_alpha:          int  = 128
    verbalizer_lora_rank: int = 32

    # GRPO
    G:                   int  = 5
    gen_temperature:     float = 0.9
    gen_max_new_tokens:  int  = 512
    kl_coef:             float = 0.05

    # Architecture
    M:                   int  = 6     # reasoning latents
    K:                   int  = 5     # spatial tokens / waypoints

    # Loss weights
    lambda_distill:      float = 0.5
    lambda_ans:          float = 50.0

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

    # Collapse detection
    reward_collapse_window:  int   = 20    # freeze teacher if reward=0 for this many consecutive steps
    lm_collapse_threshold:   float = 0.01  # skip verbalizer update if LM loss falls below this


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
        # Spatial tokens + SpatialMLP — 10x higher LR!
        # Because they are trained from scratch, they need a higher LR than the pre-trained LoRA weights.
        {
            "params": list(student.spatial_mlp.parameters()) + [student.spatial_tokens],
            "lr": cfg.student_lr * 10.0,
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
            # Optimizers (commented out to save space/time)
            # "teacher_opt":    teacher_opt.state_dict(),
            # "student_opt":    student_opt.state_dict(),
            # "verbalizer_opt": verbalizer_opt.state_dict(),
            # Schedulers
            # "teacher_sched":    teacher_sched.state_dict(),
            # "student_sched":    student_sched.state_dict(),
            # "verbalizer_sched": verbalizer_sched.state_dict(),
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

    # Restore optimizer and scheduler states (commented out)
    # teacher_opt.load_state_dict(state["teacher_opt"])
    # student_opt.load_state_dict(state["student_opt"])
    # verbalizer_opt.load_state_dict(state["verbalizer_opt"])
    # teacher_sched.load_state_dict(state["teacher_sched"])
    # student_sched.load_state_dict(state["student_sched"])
    # verbalizer_sched.load_state_dict(state["verbalizer_sched"])

    logger.info(f"Resumed from checkpoint at step {step}")
    return step


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def log_memory(tag: str):
    pass # Commented out to reduce console spam now that OOM issues are resolved
    # if torch.cuda.is_available():
    #     allocated = torch.cuda.memory_allocated() / (1024**3)
    #     reserved = torch.cuda.memory_reserved() / (1024**3)
    #     logger.info(f"[Memory] {tag}: Allocated={allocated:.2f} GB | Reserved={reserved:.2f} GB")

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
    tokenizer.padding_side = 'left'

    # ------------------------------------------------------------------
    # 2. Build models
    # ------------------------------------------------------------------
    log_memory("Before loading models")

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

    log_memory("After Teacher loaded")

    logger.info("Building Student …")
    student = LatentStudent(
        model_name=cfg.base_model_name,
        M=cfg.M,
        K=cfg.K,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        end_think_token_id=answer_token_id,
    ).to(device)

    log_memory("After Student loaded")

    logger.info("Building Verbalizer …")
    verbalizer = Verbalizer(
        model_name=cfg.verbalizer_name,
        student_hidden=student.hidden_dim,
        lora_rank=cfg.verbalizer_lora_rank,
        lora_alpha=cfg.verbalizer_lora_rank * 2,
    ).to(device)

    log_memory("After Verbalizer loaded")

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

    teacher_opt.zero_grad()
    student_opt.zero_grad()
    verbalizer_opt.zero_grad()

    # Infinite dataloader iterator
    data_iter = iter(dataloader)

    # ── Collapse detection state ────────────────────────────────────────
    reward_zero_streak = 0             # consecutive steps with reward_mean == 0
    teacher_frozen_by_watchdog = False # set True when watchdog freezes teacher
    lm_collapse_streak = 0            # consecutive steps with lm_loss < threshold
    cached_grad_norms = {}            # captured BEFORE zero_grad for correct logging

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
        pixel_values_videos = batch.get("pixel_values_videos")
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(device)
        video_grid_thw = batch.get("video_grid_thw")
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_waypoints   = batch["gt_waypoints"].to(device)          # [batch, K, 2]
        ground_truth   = batch["ground_truth"]                      # dict (stays on CPU)

        # ----------------------------------------------------------------
        # B. Teacher GRPO step
        #    Internally: rollouts → score → GRPO backward → step → h_T
        # ----------------------------------------------------------------
        log_memory(f"Step {step} - Before Teacher")

        is_accum_step = ((step + 1) % cfg.grad_accum_steps == 0) or (step == cfg.total_steps - 1)

        # When watchdog has frozen teacher, suppress optimizer step entirely
        # but still run rollouts (needed for h_T, τ+, τ-)
        effective_accum_step = is_accum_step and not teacher_frozen_by_watchdog

        buffer: RolloutBuffer = teacher.training_step(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            ground_truth=ground_truth,
            reward_fns=reward_fns,
            reward_weights=reward_weights,
            optimizer=teacher_opt,
            tokenizer=tokenizer,
            grad_clip=cfg.grad_clip,
            grad_accum_steps=cfg.grad_accum_steps,
            is_accum_step=effective_accum_step,
        )
        if teacher_frozen_by_watchdog:
            teacher_opt.zero_grad()  # clear any accumulated noise
        if is_accum_step:
            teacher_sched.step()
        log_memory(f"Step {step} - After Teacher")

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
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            buffer=buffer,
            gt_waypoints=gt_waypoints,
            global_step=step,
            task_types=ground_truth.get("task_type", None),
        )
        log_memory(f"Step {step} - After Student/Verbalizer Forward")

        # ----------------------------------------------------------------
        # E. Backward passes — phase dependent
        # ----------------------------------------------------------------
        is_warmup = (step < cfg.warmup_steps)

        if is_warmup:
            # --- E1. Verbalizer backward (LM loss on τ+, z detached) ----
            loss_lm = loss_out.lm_loss / cfg.grad_accum_steps
            loss_lm.backward()
            if is_accum_step:
                v_norm = nn.utils.clip_grad_norm_(verbalizer.parameters(), cfg.grad_clip)
                if torch.isnan(v_norm) or torch.isinf(v_norm):
                    logger.warning(f"Verbalizer gradients are NaN/Inf (norm={v_norm}). Skipping step.")
                elif lm_collapse_streak >= 5:
                    logger.warning(
                        f"[Step {step}] LM loss collapsed ({loss_out.metrics.get('loss/lm_loss', 0):.6f} < {cfg.lm_collapse_threshold}) "
                        f"for {lm_collapse_streak} consecutive steps — skipping verbalizer update."
                    )
                else:
                    verbalizer_opt.step()
                verbalizer_sched.step()
                verbalizer_opt.zero_grad()

            # --- E2. Student backward (distill + ans + spatial) ----------
            loss_student = loss_out.student_total / cfg.grad_accum_steps
            loss_student.backward()
            if is_accum_step:
                # Capture grad norms BEFORE clipping (every accum step for fresh logging)
                from training.student_losses import StudentLossComputer
                cached_grad_norms = StudentLossComputer.log_student_grad_norms(student)
                
                s_norm = nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                if torch.isnan(s_norm) or torch.isinf(s_norm):
                    logger.warning(f"Student gradients are NaN/Inf (norm={s_norm}). Skipping step.")
                else:
                    student_opt.step()
                student_sched.step()
                student_opt.zero_grad()

        else:
            # --- E3. Student backward (verb + distill + ans + spatial) ---
            # Verbalizer is frozen; DPO gradient flows through CA into Student
            loss_student = loss_out.student_total / cfg.grad_accum_steps
            loss_student.backward()
            if is_accum_step:
                # Capture grad norms BEFORE clipping (every accum step for fresh logging)
                from training.student_losses import StudentLossComputer
                cached_grad_norms = StudentLossComputer.log_student_grad_norms(student)

                s_norm = nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                if torch.isnan(s_norm) or torch.isinf(s_norm):
                    logger.warning(f"Student gradients are NaN/Inf (norm={s_norm}). Skipping step.")
                else:
                    student_opt.step()
                student_sched.step()
                student_opt.zero_grad()

        # Clear latents from verbalizer to release the graph memory
        verbalizer.clear_latents()

        log_memory(f"Step {step} - After Backprop")

        # ----------------------------------------------------------------
        # E-post. Collapse watchdogs
        # ----------------------------------------------------------------
        # ── Fix 3: Reward collapse watchdog ─────────────────────────────
        reward_mean_now = float(buffer.rewards.mean().item())
        if reward_mean_now <= 0.0:
            reward_zero_streak += 1
        else:
            reward_zero_streak = 0
            if teacher_frozen_by_watchdog:
                # Reward recovered — unfreeze teacher
                teacher_frozen_by_watchdog = False
                logger.info(f"[Step {step}] Reward recovered ({reward_mean_now:.4f}) — unfreezing teacher.")

        if reward_zero_streak >= cfg.reward_collapse_window and not teacher_frozen_by_watchdog:
            teacher_frozen_by_watchdog = True
            logger.warning(
                f"\n{'='*60}\n"
                f"[WATCHDOG] Reward has been 0.0 for {reward_zero_streak} consecutive steps!\n"
                f"Freezing teacher optimizer to prevent further corruption.\n"
                f"L_distill quality gate is also active (distill_gated={loss_out.distill_gated}).\n"
                f"{'='*60}"
            )

        # ── Fix 5: LM loss collapse detection ──────────────────────────
        lm_loss_val = loss_out.metrics.get('loss/lm_loss', 1.0)
        if is_warmup and lm_loss_val < cfg.lm_collapse_threshold:
            lm_collapse_streak += 1
        else:
            lm_collapse_streak = 0

        # ----------------------------------------------------------------
        # F. Logging  (console + WandB)
        # ----------------------------------------------------------------
        if step % cfg.log_steps == 0:
            teacher_stats = GRPOTeacher.log_rollout_stats(buffer)
            m = loss_out.metrics   # shorthand

            # Determine if this batch contains trajectory tasks
            is_trajectory = False
            task_types = ground_truth.get("task_type", [])
            if isinstance(task_types, list):
                is_trajectory = any(t == "trajectory" for t in task_types)
            elif isinstance(task_types, str):
                is_trajectory = (task_types == "trajectory")
            if not task_types:
                is_trajectory = True # fallback if no info is provided

            log_msg = (
                f"Step {step:>5d}/{cfg.total_steps} | "
                f"student={m['loss/student_total']:.4f} | "
                f"distill={m['loss/l_distill']:.4f}"
            )
            if loss_out.distill_gated:
                log_msg += " [GATED]"
            
            if is_trajectory and "loss/l_ans" in m:
                log_msg += f" | ans={m['loss/l_ans']:.4f}"
                
            log_msg += (
                f" | reward_mean={teacher_stats['grpo/reward_mean']:.4f} | "
                f"phase={'warmup' if is_warmup else 'frozen'}"
            )
            if teacher_frozen_by_watchdog:
                log_msg += " | TEACHER_FROZEN"
            if is_warmup:
                log_msg += f" | lm={m.get('loss/lm_loss', 0):.4f}"
            else:
                log_msg += f" | verb={m.get('loss/l_verb', 0):.4f}"
            logger.info(log_msg)

            if use_wandb:
                wandb_payload = {
                    # ── Student losses ───────────────────────────────────
                    "loss/student_total":        m["loss/student_total"],
                    "loss/l_distill":            m["loss/l_distill"],
                    "loss/lm_loss":             m.get("loss/lm_loss", 0.0),
                    "loss/l_verb":              m.get("loss/l_verb", 0.0),

                    # ── Teacher / GRPO ───────────────────────────────────
                    **{k.replace("grpo/", "teacher/"): v for k, v in teacher_stats.items()},

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

                    # ── Distillation quality gate ─────────────────────────
                    "distill/gated":            m.get("distill/gated",       0.0),
                    "distill/max_reward":       m.get("distill/max_reward",  0.0),

                    # ── Learning rates ───────────────────────────────────
                    "lr/teacher":               teacher_sched.get_last_lr()[0],
                    "lr/student":               student_sched.get_last_lr()[0],
                    "lr/verbalizer":            (verbalizer_sched.get_last_lr()[0]
                                                 if not verbalizer.is_frozen() else 0.0),
                }

                # Conditionally log trajectory-specific metrics
                if is_trajectory:
                    if "loss/l_ans" in m:
                        wandb_payload["loss/l_ans"] = m["loss/l_ans"]
                    if "waypoints/pred_mean" in m:
                        wandb_payload["waypoints/pred_mean"] = m["waypoints/pred_mean"]
                    if "waypoints/pred_std" in m:
                        wandb_payload["waypoints/pred_std"] = m["waypoints/pred_std"]

                # ── Rollout Text Logging (every step for first 50 steps, then every 10 steps) ──
                is_text_log_step = (step < 50) or (step % 10 == 0)
                if is_text_log_step:
                    import json
                    import os
                    log_root = "logs"
                    os.makedirs(os.path.join(log_root, "generation"), exist_ok=True)
                    os.makedirs(os.path.join(log_root, "verbalizer"), exist_ok=True)
                    os.makedirs(os.path.join(log_root, "waypoint"), exist_ok=True)
                    datasets = ground_truth.get("dataset", [])

                    # ── Waypoint Table Logging ───────────────────────────────
                    try:
                        wp_table = wandb.Table(columns=["Batch_Idx", "Dataset", "Pred_Waypoints", "GT_Waypoints"])
                        wp_data = []
                        B_len = gt_waypoints.shape[0]
                        if loss_out.pred_waypoints is not None:
                            pred_wp = loss_out.pred_waypoints.cpu().tolist()
                            gt_wp = gt_waypoints.cpu().tolist()
                            for b in range(B_len):
                                ds_name = datasets[b] if b < len(datasets) else "unknown"
                                pred_str = str([[round(p[0], 3), round(p[1], 3)] for p in pred_wp[b]])
                                gt_str = str([[round(g[0], 3), round(g[1], 3)] for g in gt_wp[b]])
                                wp_table.add_data(b, ds_name, pred_str, gt_str)
                                wp_data.append({
                                    "Batch_Idx": b,
                                    "Dataset": ds_name,
                                    "Pred_Waypoints": [[round(p[0], 3), round(p[1], 3)] for p in pred_wp[b]],
                                    "GT_Waypoints": [[round(g[0], 3), round(g[1], 3)] for g in gt_wp[b]]
                                })
                            
                            wandb_payload["waypoints/pred_vs_gt_table"] = wp_table
                            # Save locally
                            wp_log_path = os.path.join(log_root, "waypoint", f"step_{step:06d}.json")
                            with open(wp_log_path, "w", encoding="utf-8") as f:
                                json.dump(wp_data, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to log waypoints table: {e}")

                    # ── Generation Table Logging ─────────────────────────────
                    try:
                        table = wandb.Table(columns=["Batch_Idx", "Rollout_Idx", "Reward", "Advantage", "Dataset", "Text"])
                        gen_data = []
                        G_len = buffer.rewards.shape[0]
                        B_len = buffer.rewards.shape[1]
                        for b in range(B_len):
                            ds_name = datasets[b] if b < len(datasets) else "unknown"
                            for g in range(G_len):
                                reward_val = float(buffer.rewards[g, b].cpu())
                                adv_val = float(buffer.advantages[g, b].cpu())
                                rollout_txt = buffer.rollout_texts[g][b]
                                table.add_data(
                                    b, 
                                    g, 
                                    reward_val, 
                                    adv_val, 
                                    ds_name,
                                    rollout_txt
                                )
                                gen_data.append({
                                    "Batch_Idx": b,
                                    "Rollout_Idx": g,
                                    "Reward": reward_val,
                                    "Advantage": adv_val,
                                    "Dataset": ds_name,
                                    "Text": rollout_txt
                                })
                        wandb_payload["rollouts/generation_samples"] = table
                        # Save locally
                        gen_log_path = os.path.join(log_root, "generation", f"step_{step:06d}.json")
                        with open(gen_log_path, "w", encoding="utf-8") as f:
                            json.dump(gen_data, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to log rollout table: {e}")

                    # ── Verbalizer Output Logging ────────────────────────────
                    try:
                        verbalizer_table = wandb.Table(columns=["Batch_Idx", "Dataset", "L_lm", "Teacher_Best_Text", "Verbalizer_Text"])
                        verb_data = []
                        from transformers import GenerationConfig
                        gen_cfg = GenerationConfig(
                            max_new_tokens=128, 
                            temperature=0.7, 
                            do_sample=True, 
                            pad_token_id=tokenizer.pad_token_id, 
                            eos_token_id=tokenizer.eos_token_id
                        )
                        gen_out = verbalizer.generate_from_latents(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            latents=loss_out.latents,
                            generation_config=gen_cfg,
                        )
                        prompt_len = input_ids.shape[1]
                        generated_ids = gen_out[:, prompt_len:]
                        gen_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        for b, txt in enumerate(gen_texts):
                            ds_name = datasets[b] if b < len(datasets) else "unknown"
                            teacher_best = buffer.rollout_texts[buffer.best_idx[b].item()][b]
                            lm_loss_val = float(m.get("loss/lm_loss", 0.0))
                            verbalizer_table.add_data(b, ds_name, lm_loss_val, teacher_best, txt)
                            verb_data.append({
                                "Batch_Idx": b,
                                "Dataset": ds_name,
                                "L_lm": lm_loss_val,
                                "Teacher_Best_Text": teacher_best,
                                "Verbalizer_Text": txt
                            })
                        wandb_payload["rollouts/verbalizer_samples"] = verbalizer_table
                        # Save locally
                        verb_log_path = os.path.join(log_root, "verbalizer", f"step_{step:06d}.json")
                        with open(verb_log_path, "w", encoding="utf-8") as f:
                            json.dump(verb_data, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to log verbalizer table: {e}")

                # Gradient norm logging — uses cached norms from Fix 1
                grad_norms = cached_grad_norms  # captured BEFORE zero_grad
                logger.info(
                    f"  grad_norm/lora={grad_norms.get('grad_norm/lora_total', 0):.4f} | "
                    f"  grad_norm/spatial={grad_norms.get('grad_norm/spatial_total', 0):.4f}"
                )
                wandb_payload["grad/lora_total"] = grad_norms.get("grad_norm/lora_total", 0.0)
                wandb_payload["grad/spatial_total"] = grad_norms.get("grad_norm/spatial_total", 0.0)

                # Finally, log everything together to keep the x-axis perfectly synced
                wandb.log(wandb_payload)

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
    parser.add_argument("--log_steps",     type=int, default=10, help="Frequency of console and WandB metrics logging")
    parser.add_argument("--grad_accum_steps", type=int, default=32, help="Number of gradient accumulation steps")
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
        log_steps=args.log_steps,
        grad_accum_steps=args.grad_accum_steps,
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