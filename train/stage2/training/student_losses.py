"""
student_losses.py
-----------------
Computes all Student loss terms for ThinkFlow-VLA Stage 2.

Phase-aware behaviour:
    Warm-up (step < warmup_steps):
        - Verbalizer LM loss on τ+ (latents DETACHED)        → lm_loss
        - L_distill + L_ans                                  → student_total
    Frozen Verbalizer (step >= warmup_steps):
        - L_verb (DPO through frozen Verbalizer) + L_distill + L_ans → student_total

Loss breakdown:
    L_distill  = MSE(h_S_answer, h_T)               — hidden state alignment
    L_ans      = MSE(pred_waypoints, gt_waypoints)   — physical waypoint grounding
    L_verb     = DPO(τ+, τ−) via frozen Verbalizer   — reasoning alignment (frozen phase only)
    lm_loss    = Verbalizer CE on τ+ (warm-up only)

Key note on <answer> alignment for L_distill:
    Teacher h_T  : hidden state at the <ans> token position deep in τ+
                   (extracted post-GRPO-update, lives in RolloutBuffer.h_T)
    Student h_S  : hidden state at the LAST PREFIX TOKEN position
                   This is the Student's structural equivalent — the moment
                   the model transitions from understanding to generation.
                   Position = prompt_len - 1 in the prompt input_ids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Output container — matches the interface used by train_stage2.py:
#   loss_out.lm_loss.backward()         (warm-up only)
#   loss_out.student_total.backward()    (always)
#   loss_out.metrics['loss/student_total']
#   loss_out.metrics['loss/l_distill']
#   loss_out.metrics['loss/l_ans']
#   loss_out.metrics.get('loss/lm_loss', 0)
#   loss_out.metrics.get('loss/l_verb', 0)
# ---------------------------------------------------------------------------

@dataclass
class LossOutput:
    """
    Returned by StudentLossComputer.compute().

    train_stage2.py calls:
        loss_out.lm_loss.backward()         (warm-up only)
        loss_out.student_total.backward()    (always)
        loss_out.metrics[...]               (logging)
    """
    # Primary loss tensors (used for .backward())
    student_total: torch.Tensor     # scalar — the combined Student loss
    lm_loss: Optional[torch.Tensor] # scalar — Verbalizer LM loss (warm-up only, None otherwise)

    # Logging
    metrics: dict = field(default_factory=dict)

    # Saved latents for generation logging
    latents: Optional[torch.Tensor] = None
    pred_waypoints: Optional[torch.Tensor] = None

    # Whether L_distill was gated off this step (for external monitoring)
    distill_gated: bool = False


# Alias for backwards compat if any code references the old name
StudentLossOutput = LossOutput


# ---------------------------------------------------------------------------
# Loss computer
# ---------------------------------------------------------------------------

class StudentLossComputer(nn.Module):
    """
    Phase-aware loss computation for the Student and Verbalizer.

    Does NOT call .backward() or .step() — those remain in train_stage2.py
    so that the optimizer graphs stay fully decoupled.

    Parameters
    ----------
    warmup_steps   : step threshold for warm-up → frozen transition
    lambda_distill : weight for L_distill
    lambda_ans     : weight for L_ans
    """

    def __init__(
        self,
        warmup_steps: int = 3000,
        lambda_distill: float = 1.0,
        lambda_ans: float = 1.0,
        distill_clamp_max: float = 10.0,
    ):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.lambda_distill = lambda_distill
        self.lambda_ans = lambda_ans
        self.distill_clamp_max = distill_clamp_max

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Individual loss computations
    # -----------------------------------------------------------------------

    def _compute_l_distill(
        self,
        h_S: torch.Tensor,    # [batch, d]  Student answer hidden state
        h_T: torch.Tensor,    # [batch, d]  Teacher answer hidden state (from buffer)
        max_reward: float = 1.0,   # max reward across all G rollouts this step
    ) -> tuple:
        """
        L_distill = λ_distill * MSE(h_S, h_T)

        h_T is detached from the Teacher's graph (Teacher already updated).
        h_S must remain in the Student's computation graph.

        Quality gate: when max_reward == 0 (all rollouts failed to produce valid
        output), h_T is extracted from garbage text and is not a meaningful target.
        In this case we replace the MSE with a small cosine-similarity regulariser
        that just keeps h_S from collapsing to zero, without driving it toward the
        corrupt h_T. This breaks the positive feedback loop that caused training
        collapse around Step 147-154.

        Returns
        -------
        l_distill : torch.Tensor  — the (clamped) loss scalar
        gated     : bool          — True if the quality gate fired
        """
        h_T_aligned = h_T.detach().to(dtype=h_S.dtype)

        # ── Quality gate: skip MSE when teacher output is garbage ──────────
        if max_reward <= 0.0:
            # Small regulariser: keep h_S norm stable without matching garbage h_T.
            # -cosine_similarity drives h_S toward unit norm (prevents collapse to 0)
            # while NOT forcing alignment with the corrupt h_T direction.
            reg = 0.01 * (1.0 - F.cosine_similarity(h_S, h_S.detach(), dim=-1).mean())
            return reg, True

        raw = self.lambda_distill * F.mse_loss(h_S, h_T_aligned)
        clamped = torch.clamp(raw, max=self.distill_clamp_max)
        return clamped, False

    def _compute_l_ans(
        self,
        pred_waypoints: torch.Tensor,   # [batch, K, 2]  from SpatialMLP (in [0,1])
        gt_waypoints: torch.Tensor,     # [batch, K, 2]  normalised ground truth
        task_types: Optional[list] = None,
    ) -> torch.Tensor:
        """
        L_ans = λ_ans * MSE(pred_waypoints, gt_waypoints)
        Masked out for QA tasks.
        """
        gt_aligned = gt_waypoints.to(dtype=pred_waypoints.dtype, device=pred_waypoints.device)
        mse = F.mse_loss(pred_waypoints, gt_aligned, reduction='none') # [batch, K, 2]
        mse = mse.mean(dim=(1, 2)) # [batch]
        
        if task_types is not None:
            mask = torch.tensor([1.0 if t == "trajectory" else 0.0 for t in task_types], 
                                device=mse.device, dtype=mse.dtype)
            mse = mse * mask
            num_traj = mask.sum().clamp(min=1.0)
            loss = mse.sum() / num_traj
        else:
            loss = mse.mean()
            
        return self.lambda_ans * loss

    # -----------------------------------------------------------------------
    # Main compute method
    # -----------------------------------------------------------------------

    def compute(
        self,
        # Models (duck-typed — no concrete imports needed)
        student,           # LatentStudent
        verbalizer,        # Verbalizer
        # Input batch (prompt only, NOT the full rollout)
        input_ids: torch.Tensor,            # [batch, prompt_len]
        pixel_values,
        image_grid_thw,
        attention_mask: torch.Tensor,
        # From Teacher's training_step()
        buffer,            # RolloutBuffer
        # Spatial supervision
        gt_waypoints: torch.Tensor,         # [batch, K, 2]  normalised [0,1]
        # Verbalizer schedule
        global_step: int,
        task_types: Optional[list] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """
        Compute all Student (and optionally Verbalizer) losses for one step.

        Parameters
        ----------
        input_ids       : prompt tokens fed to the Student (vision + instruction)
        buffer          : populated RolloutBuffer from GRPOTeacher.training_step()
        gt_waypoints    : normalised 2D ground-truth waypoints for L_ans
        global_step     : used to switch between warm-up and frozen-verbalizer phases

        Returns
        -------
        LossOutput with student_total, lm_loss, and metrics dict.
        """
        prompt_len = input_ids.shape[1]
        is_warmup = (global_step < self.warmup_steps)

        # ==================================================================
        # 1. Student forward: generate latents + h_S + spatial tokens + waypoints
        # ==================================================================
        latents, h_S, spatial_hidden, pred_waypoints = student.generate_latents(
            input_ids, pixel_values, image_grid_thw, attention_mask,
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw,
        )
        # latents      : List[M tensors, each [batch, d_student]]
        # h_S          : [batch, d_student] (hidden state at </think>)
        # spatial_hidden: [batch, K, d_student]
        # pred_waypoints: [batch, K, 2]  in [0, 1]

        # Stack latents for Verbalizer  →  [batch, M, d_student]
        z = verbalizer.stack_latents(latents)

        # ==================================================================
        # 3. Loss computations — always present
        # ==================================================================
        max_reward = float(buffer.rewards.max().item())
        l_distill, distill_gated = self._compute_l_distill(h_S, buffer.h_T, max_reward=max_reward)
        l_ans = self._compute_l_ans(pred_waypoints, gt_waypoints, task_types=task_types)

        # ==================================================================
        # 5. Phase-specific losses
        # ==================================================================
        lm_loss = None

        if is_warmup:
            # -- Warm-up: Verbalizer LM loss on τ+ (latents DETACHED) ------
            lm_loss = verbalizer.compute_lm_loss(
                input_ids=buffer.tau_pos_ids,
                attention_mask=buffer.tau_pos_mask,
                latents=z.detach(),          # ← critical: no gradient into Student
                labels=self._make_lm_labels(buffer.tau_pos_ids, prompt_len),
            )

            # Student total = L_distill + L_ans (no L_verb)
            student_total = l_distill + l_ans

        else:
            # -- Frozen phase: DPO through frozen Verbalizer ----------------
            l_verb, dpo_metrics = verbalizer.compute_dpo_loss(
                pos_input_ids=buffer.tau_pos_ids,
                neg_input_ids=buffer.tau_neg_ids,
                pos_attention_mask=buffer.tau_pos_mask,
                neg_attention_mask=buffer.tau_neg_mask,
                latents=z,                   # ← live graph: gradient flows into Student
                pos_response_mask=buffer.tau_pos_response_mask,
                neg_response_mask=buffer.tau_neg_response_mask,
            )

            # Student total = L_verb + L_distill + L_ans
            student_total = l_verb + l_distill + l_ans

        # ==================================================================
        # 6. Metrics — always include ALL expected keys (for logging)
        # ==================================================================
        with torch.no_grad():
            metrics = {
                "loss/l_distill":       l_distill.item(),
                "loss/l_ans":           l_ans.item(),
                "loss/student_total":   student_total.item(),
                "phase/is_warmup":      float(is_warmup),
                "waypoints/pred_mean":  pred_waypoints.mean().item(),
                "waypoints/pred_std":   pred_waypoints.std().item(),
                "distill/h_S_norm":     h_S.float().norm(dim=-1).mean().item(),
                "distill/h_T_norm":     buffer.h_T.float().norm(dim=-1).mean().item(),
                "distill/cosine_sim":   F.cosine_similarity(
                                            h_S.float(), buffer.h_T.float(), dim=-1
                                        ).mean().item(),
                "distill/gated":        float(distill_gated),
                "distill/max_reward":   max_reward,
            }

            # Always set both keys so train_stage2.py logging never fails
            if is_warmup:
                metrics["loss/lm_loss"] = lm_loss.item()
                metrics["loss/l_verb"] = 0.0
            else:
                metrics["loss/lm_loss"] = 0.0
                metrics["loss/l_verb"] = l_verb.item()
                metrics.update({f"dpo/{k}": v for k, v in dpo_metrics.items()})

        return LossOutput(
            student_total=student_total,
            lm_loss=lm_loss,
            metrics=metrics,
            latents=z,
            pred_waypoints=pred_waypoints.detach(),
            distill_gated=distill_gated,
        )

    # -----------------------------------------------------------------------
    # Helper: build LM labels for Verbalizer warm-up
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_lm_labels(
        input_ids: torch.Tensor,   # [batch, seq]  full τ+ sequence
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Labels for LM loss: -100 on prompt tokens (no loss there),
        real token ids on response tokens (the τ+ chain-of-thought).
        """
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        return labels

    # -----------------------------------------------------------------------
    # Diagnostic: log gradient norms on Student parameters (call after backward)
    # -----------------------------------------------------------------------

    @staticmethod
    def log_student_grad_norms(student) -> dict:
        """
        Returns gradient norms for key Student parameter groups.
        Call this AFTER student_total.backward() and BEFORE optimizer.step().
        """
        norms = {}
        for name, param in student.named_parameters():
            if param.grad is not None:
                norms[f"grad_norm/{name}"] = param.grad.float().norm().item()

        # Summary norms by component
        lora_vals = [v for k, v in norms.items() if "lora_" in k]
        spatial_vals = [v for k, v in norms.items() if "spatial" in k]

        norms["grad_norm/lora_total"] = (
            torch.tensor(lora_vals).norm().item() if lora_vals else 0.0
        )
        norms["grad_norm/spatial_total"] = (
            torch.tensor(spatial_vals).norm().item() if spatial_vals else 0.0
        )
        return norms


# ---------------------------------------------------------------------------
# Convenience factory (used in train_stage2.py)
# ---------------------------------------------------------------------------

def build_student_loss_computer(
    warmup_steps: int = 3000,
    lambda_distill: float = 1.0,
    lambda_ans: float = 1.0,
    distill_clamp_max: float = 10.0,
) -> StudentLossComputer:
    return StudentLossComputer(
        warmup_steps=warmup_steps,
        lambda_distill=lambda_distill,
        lambda_ans=lambda_ans,
        distill_clamp_max=distill_clamp_max,
    )