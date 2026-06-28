"""
grpo_teacher.py
---------------
Textual Teacher (F_θT) for ThinkFlow-VLA Stage 2.

Responsibilities (in order per training step):
  1. Generate G=5 rollout traces per input via temperature sampling
  2. Score each rollout with pluggable reward functions
  3. Compute group-relative advantage scores  A_g = (r_g - μ) / (σ + ε)
  4. Compute GRPO policy-gradient loss and update Teacher weights
  5. Identify τ+ (highest advantage) and τ- (lowest advantage)
  6. Run a FRESH forward pass of τ+ through the UPDATED Teacher
     to extract the <answer> token hidden state h_T  (target for L_distill)

Output format the Teacher is trained to produce:
    <think> ... chain-of-thought reasoning ... </think>
    <ans> x1,y1;x2,y2;x3,y3;x4,y4;x5,y5 </ans>

GRPO loss (REINFORCE with group-relative baseline, no importance-sampling clipping):
    L_GRPO = -1/G  Σ_g  A_g  *  (1/|τ_g|)  Σ_t  log π_θ(token_t | context, τ_g,<t)

Optional KL penalty against a frozen reference snapshot (disabled by default).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Protocol

from transformers import AutoModelForImageTextToText, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict

import warnings

# ---------------------------------------------------------------------------
# Reward function interface  (concrete implementations in rewards/)
# ---------------------------------------------------------------------------

class RewardFunction(Protocol):
    """
    Any reward function must implement __call__ with this signature.
    Returns a [batch] float tensor of scalar rewards, one per rollout.
    """
    def __call__(
        self,
        rollout_ids: torch.Tensor,      # [batch, seq] — generated token ids
        rollout_text: List[str],        # decoded strings for text-based rewards
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        ground_truth: dict,             # task-specific GT (waypoints, QA answers, …)
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...             # [batch]  float32


# ---------------------------------------------------------------------------
# Rollout buffer (one per training step)
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """
    Stores G rollouts for a single (batch of) input(s) after GRPO scoring.
    Passed from GRPOTeacher.training_step() to student_losses.py.
    """
    # Tokenised rollouts: list of G tensors, each [batch, seq_g]
    rollout_ids:       Optional[List[torch.Tensor]] = None
    rollout_texts:     Optional[List[List[str]]] = None    # decoded; outer=G, inner=batch
    attention_masks:   Optional[List[torch.Tensor]] = None

    # Per-rollout scalars: shape [G, batch]
    rewards:           Optional[torch.Tensor] = None
    advantages:        Optional[torch.Tensor] = None

    # Best / worst indices into the G dimension (per batch item)
    best_idx:          Optional[torch.Tensor] = None       # [batch]  int64
    worst_idx:         Optional[torch.Tensor] = None       # [batch]  int64

    # τ+ and τ- token ids and masks (best/worst selected per batch item)
    tau_pos_ids:       Optional[torch.Tensor] = None       # [batch, seq_pos]
    tau_neg_ids:       Optional[torch.Tensor] = None       # [batch, seq_neg]
    tau_pos_mask:      Optional[torch.Tensor] = None
    tau_neg_mask:      Optional[torch.Tensor] = None

    # Response-only masks (1 on generated tokens, 0 on prompt)
    tau_pos_response_mask: Optional[torch.Tensor] = None   # [batch, seq_pos]
    tau_neg_response_mask: Optional[torch.Tensor] = None   # [batch, seq_neg]

    # Answer token positions in τ+ (for L_distill)
    answer_token_pos:  Optional[torch.Tensor] = None       # [batch]  int64

    # Teacher's <answer> hidden state from the post-update forward pass
    h_T: Optional[torch.Tensor] = None   # [batch, d_teacher]; filled after update
    grpo_loss: Optional[float] = None    # for logging
    dataset_source: Optional[List[str]] = None
    kl_loss: Optional[float] = None
    kl_coef: Optional[float] = None


# ---------------------------------------------------------------------------
# Teacher model
# ---------------------------------------------------------------------------

class GRPOTeacher(nn.Module):
    """
    Textual Teacher (F_θT): Qwen3.5-4B with LoRA.
    Identical base architecture to the Latent Student; diverges through training.

    Parameters
    ----------
    model_name        : HuggingFace checkpoint (same as Student init)
    G                 : number of rollouts per input
    answer_token_id   : token id of <ans> (registered as a special token)
    lora_rank / alpha : LoRA hyperparameters
    gen_temperature   : sampling temperature for diverse rollouts
    gen_max_new_tokens: max tokens to generate per rollout
    kl_coef           : KL penalty coefficient (0 = disabled)
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "unsloth/Qwen3.5-4B",
        G: int = 5,
        answer_token_id: int = -1,          # set after tokenizer extension
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        gen_temperature: float = 1.0,
        gen_max_new_tokens: int = 512,
        kl_coef: float = 0.0,
        target_kl: float = 0.02,
        use_gradient_checkpointing: bool = True,
        offload_ref_model: bool = True,
        backward_batch_size: int = 1,
        max_completion_length: int = 512,
        epsilon: float = 0.2,
    ):
        super().__init__()
        self.G = G
        self.end_think_token_id = answer_token_id  # Rename variable internally
        self.gen_temperature  = gen_temperature
        self.gen_max_new_tokens = gen_max_new_tokens
        self.kl_coef = kl_coef
        self.target_kl = target_kl
        self.offload_ref_model = offload_ref_model
        self.max_completion_length = max_completion_length
        self.epsilon = epsilon
        self.backward_batch_size = backward_batch_size

        # ------------------------------------------------------------------
        # 1. Base VLM
        # ------------------------------------------------------------------
        base = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        # Freeze vision encoder (if present)
        if hasattr(base.model, "visual"):
            for param in base.model.visual.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. LoRA
        # ------------------------------------------------------------------
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "out_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"],
            bias="none",
        )
        self.vlm = get_peft_model(base, lora_cfg)
        self.hidden_dim: int = self.vlm.config.text_config.hidden_size   # 2048

        # ------------------------------------------------------------------
        # 3. Optional frozen reference snapshot for KL penalty
        #    Created lazily on first call when kl_coef > 0
        # ------------------------------------------------------------------
        self._ref_model: Optional[nn.Module] = None

        if use_gradient_checkpointing:
            self.vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.vlm.config.use_cache = False

    # -----------------------------------------------------------------------
    # Reference model for KL (lazy init, frozen copy of initial Teacher)
    # -----------------------------------------------------------------------

    def _ensure_ref_model(self):
        """Create a frozen reference snapshot (used only when kl_coef > 0)."""
        if self._ref_model is not None:
            return
        import copy
        self._ref_model = copy.deepcopy(self.vlm)
        for p in self._ref_model.parameters():
            p.requires_grad = False
        self._ref_model.eval()
        if not self.offload_ref_model:
            device = next(self.vlm.parameters()).device
            self._ref_model.to(device)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_input_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed tokens and splice in visual encoder features."""
        embeds = self.vlm.model.model.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            mask = (input_ids == self.vlm.config.image_token_id)
            if mask.any():
                with torch.no_grad():
                    img_feats = (self.vlm.model.model.visual(pixel_values, grid_thw=image_grid_thw) if image_grid_thw is not None else self.vlm.model.model.visual(pixel_values))
                embeds = embeds.clone()
                
                if not isinstance(img_feats, torch.Tensor):
                    # Qwen3_5/Qwen2VL returns BaseModelOutputWithPooling where last_hidden_state is unmerged
                    # and pooler_output is the projected/merged 2560-dim tensor.
                    img_feats = getattr(img_feats, 'pooler_output', img_feats[0])
                    
                embeds[mask] = img_feats.to(embeds.dtype)
            
        if pixel_values_videos is not None:
            self._video_token_id = getattr(self.vlm.config, "video_token_id", 248057)
            mask_vid = (input_ids == self._video_token_id)
            if mask_vid.any():
                with torch.no_grad():
                    vid_feats = (self.vlm.model.model.visual(pixel_values_videos, grid_thw=video_grid_thw) if video_grid_thw is not None else self.vlm.model.model.visual(pixel_values_videos))
                embeds = embeds.clone()
                if not isinstance(vid_feats, torch.Tensor):
                    vid_feats = getattr(vid_feats, 'pooler_output', vid_feats[0])
                embeds[mask_vid] = vid_feats.to(embeds.dtype)
                
        return embeds

    def _find_think_end_positions(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Locate the </think> token position in each sequence.
        Falls back to the last non-padding position if </think> is not found
        (should not happen after SFT, but guards against edge cases).

        Parameters
        ----------
        token_ids : [batch, seq]

        Returns
        -------
        positions : [batch]  int64
        """
        batch_size = token_ids.shape[0]
        positions  = torch.zeros(batch_size, dtype=torch.long, device=token_ids.device)

        for i in range(batch_size):
            matches = (token_ids[i] == self.end_think_token_id).nonzero(as_tuple=False)
            if matches.numel() > 0:
                positions[i] = matches[0, 0]           # first occurrence
            else:
                # Fallback: last non-pad token
                non_pad = (token_ids[i] != 0).nonzero(as_tuple=False)
                positions[i] = non_pad[-1, 0] if non_pad.numel() > 0 else 0

        return positions

    def _compute_response_mask(
        self,
        full_ids: torch.Tensor,       # [batch, seq]
        prompt_len: int,
    ) -> torch.Tensor:
        """1 on generated (response) tokens, 0 on prompt tokens and padding."""
        mask = torch.zeros_like(full_ids, dtype=torch.float)
        mask[:, prompt_len:] = (full_ids[:, prompt_len:] != 0).float()
        return mask

    # -----------------------------------------------------------------------
    # Step 1: Generate G rollouts
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate_rollouts(
        self,
        input_ids: torch.Tensor,            # [batch, prompt_len]
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        tokenizer,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[List[str]], List[torch.Tensor]]:
        """
        Generate G independent rollout traces via temperature sampling.
        The Teacher generates: <think>...</think><ans>...</ans>

        Returns
        -------
        all_ids   : List[G]  each [batch, seq_g]  — full (prompt + response) ids
        all_texts : List[G]  each List[batch str]  — decoded response text only
        all_masks : List[G]  each [batch, seq_g]   — full attention masks
        """
        prompt_len  = input_ids.shape[1]
        all_ids     = []
        all_texts   = []
        all_masks   = []

        gen_config = GenerationConfig(
            do_sample=True,
            temperature=self.gen_temperature,
            max_new_tokens=self.gen_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=self.G,
        )

        was_training = self.vlm.training
        self.vlm.eval()

        batch_size = input_ids.shape[0]

        try:
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "generation_config": gen_config,
                "use_cache": True,
                "return_dict_in_generate": False,
            }
            if pixel_values is not None:
                gen_kwargs["pixel_values"] = pixel_values
                gen_kwargs["image_grid_thw"] = image_grid_thw
            if pixel_values_videos is not None:
                gen_kwargs["pixel_values_videos"] = pixel_values_videos
                gen_kwargs["video_grid_thw"] = video_grid_thw

            outputs = self.vlm.generate(**gen_kwargs)
            # outputs shape: [batch * G, prompt_len + new_tokens]
            
            # Reshape from [B * G, seq_len] to [B, G, seq_len]
            # Transformers generates them interleaved: b0_g0, b0_g1... b1_g0, b1_g1...
            seq_len = outputs.shape[1]
            outputs = outputs.view(batch_size, self.G, seq_len)
            
            for g in range(self.G):
                g_outputs = outputs[:, g, :]  # [batch, seq_len]

                # Pad to consistent length within this rollout (already done by generate)
                response_ids = g_outputs[:, prompt_len:]   # [batch, new_tokens]

                # Decode response portion only
                texts = tokenizer.batch_decode(
                    response_ids, skip_special_tokens=False
                )

                # Build full attention mask (1 on all non-pad positions)
                full_mask = (g_outputs != tokenizer.pad_token_id).long()

                all_ids.append(g_outputs)
                all_texts.append(texts)
                all_masks.append(full_mask)
        finally:
            if was_training:
                self.vlm.train()

        return all_ids, all_texts, all_masks

    # -----------------------------------------------------------------------
    # Step 2: Score rollouts  (rewards injected from outside)
    # -----------------------------------------------------------------------

    def score_rollouts(
        self,
        all_ids: List[torch.Tensor],
        all_texts: List[List[str]],
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        ground_truth: dict,
        reward_fns: List[RewardFunction],
        reward_weights: Optional[List[float]] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score all G rollouts using one or more reward functions.

        Parameters
        ----------
        reward_fns      : list of RewardFunction callables
        reward_weights  : per-function weights (uniform if None)

        Returns
        -------
        rewards : [G, batch]  float32
        """
        G         = len(all_ids)
        batch     = all_ids[0].shape[0]
        weights   = reward_weights or [1.0 / len(reward_fns)] * len(reward_fns)

        rewards = torch.zeros(G, batch, device=all_ids[0].device)

        for g in range(G):
            r_g = torch.zeros(batch, device=all_ids[0].device)
            for fn, w in zip(reward_fns, weights):
                reward_out = fn(
                    rollout_ids=all_ids[g],
                    rollout_text=all_texts[g],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    ground_truth=ground_truth,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                )
                if isinstance(reward_out, torch.Tensor):
                    reward_out = reward_out.to(r_g.device).float()
                else:
                    # In case a reward fn returns a list of floats
                    reward_out = torch.tensor(reward_out, device=r_g.device, dtype=torch.float32)

                r_g = r_g + w * reward_out
            rewards[g] = r_g

        return rewards   # [G, batch]

    # -----------------------------------------------------------------------
    # Step 3: Compute group-relative advantages
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """
        Group-relative advantage normalisation within each batch item's G rollouts.
        Uses TRL's epsilon (1e-4) for better numerical stability.

            A_g = (r_g - mean_over_G(r)) / (std_over_G(r) + ε)

        Parameters
        ----------
        rewards : [G, batch]

        Returns
        -------
        advantages : [G, batch]
        """
        # Compute mean and std over the G dimension for each batch item
        mean = rewards.mean(dim=0, keepdim=True)   # [1, batch]
        std  = rewards.std(dim=0, keepdim=True)    # [1, batch]

        advantages = (rewards - mean) / (std + eps)   # [G, batch]
        # Clip to prevent any single outlier rollout from dominating the gradient
        return torch.clamp(advantages, min=-5.0, max=5.0)

    # -----------------------------------------------------------------------
    # Step 4: DR-GRPO per-token loss (ported from TRL GRPOTrainer)
    # -----------------------------------------------------------------------

    def compute_grpo_loss(
        self,
        all_ids: List[torch.Tensor],        # List[G], each [batch, seq_g]
        all_masks: List[torch.Tensor],      # List[G], each [batch, seq_g]
        advantages: torch.Tensor,           # [G, batch]
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        prompt_len: int,
        grad_accum_steps: int = 1,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DR-GRPO per-token policy-gradient loss (ported from TRL).

        Uses per-token importance weighting with PPO clipping and
        dr_grpo normalization: loss = sum(per_token_loss * mask) / (B * max_completion_length).

        Since num_iterations=1, old_per_token_logps = per_token_logps.detach(),
        so we skip the expensive pre-computation forward pass entirely.

        Optional KL penalty if kl_coef > 0.

        Returns
        -------
        (grpo_loss, kl_loss, raw_kl) : tuple of scalars
        """
        if self.kl_coef > 0:
            self._ensure_ref_model()

        G     = len(all_ids)
        B     = all_ids[0].shape[0]
        device = all_ids[0].device
        chunk_size = getattr(self, "backward_batch_size", 1)

        # ── Pre-compute per-batch-item image/video ownership ──────────────
        image_token_id = getattr(self.vlm.config, "image_token_id", None)
        video_token_id = getattr(self.vlm.config, "video_token_id", 248057)
        ref_ids = all_ids[0]  # [B, seq]
        has_image = [False] * B
        has_video = [False] * B
        if pixel_values is not None and image_token_id is not None:
            for b in range(B):
                has_image[b] = (ref_ids[b] == image_token_id).any().item()
        if pixel_values_videos is not None:
            for b in range(B):
                has_video[b] = (ref_ids[b] == video_token_id).any().item()

        # ── Chunked Forward & Backward Pass ───────────────────────────────
        # With num_iterations=1, TRL uses per_token_logps.detach() as old_logps,
        # meaning the importance ratio is always 1.0 at the start. This lets us
        # skip the expensive pre-computation forward pass entirely.
        total_rollout_loss = 0.0
        total_kl_loss = 0.0
        total_raw_kl = 0.0

        if self.kl_coef > 0 and self.offload_ref_model:
            self._ref_model.to(device)

        for g in range(G):
            for chunk_start in range(0, B, chunk_size):
                chunk_end = min(chunk_start + chunk_size, B)
                chunk_len = chunk_end - chunk_start
                
                # 1. Slice rollout to chunk size
                batch_id = all_ids[g][chunk_start:chunk_end]
                batch_mask = all_masks[g][chunk_start:chunk_end]
                
                # Response (completion) mask
                resp_mask = torch.zeros_like(batch_id, dtype=torch.float)
                resp_mask[:, prompt_len:] = (batch_id[:, prompt_len:] != 0).float()

                # 2. Correctly slice the flattened image/video patches
                pre_img_count = sum(has_image[:chunk_start])
                chunk_img_count = sum(has_image[chunk_start:chunk_end])
                
                if chunk_img_count > 0:
                    thw = image_grid_thw[pre_img_count : pre_img_count + chunk_img_count]
                    num_patches = thw.prod(dim=-1).sum().item()
                    offset = int(image_grid_thw[:pre_img_count].prod(dim=-1).sum().item()) if pre_img_count > 0 else 0
                    pv = pixel_values[offset : offset + num_patches]
                else:
                    pv = None
                    thw = None

                pre_vid_count = sum(has_video[:chunk_start])
                chunk_vid_count = sum(has_video[chunk_start:chunk_end])
                
                if chunk_vid_count > 0:
                    v_thw = video_grid_thw[pre_vid_count : pre_vid_count + chunk_vid_count]
                    num_patches = v_thw.prod(dim=-1).sum().item()
                    offset = int(video_grid_thw[:pre_vid_count].prod(dim=-1).sum().item()) if pre_vid_count > 0 else 0
                    pv_v = pixel_values_videos[offset : offset + num_patches]
                else:
                    pv_v = None
                    v_thw = None

                # 3. Forward pass
                inputs_embeds = self._build_input_embeds(batch_id, pv, thw, pv_v, v_thw)
                
                position_ids = batch_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(batch_mask == 0, 1)

                out = self.vlm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                logits = out.logits.float()

                # 4. Per-token log-probs (completion tokens only)
                target_ids = batch_id[:, 1:]
                logits_shifted = logits[:, :-1, :]
                target_logits = logits_shifted.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                per_token_logps = (target_logits - torch.logsumexp(logits_shifted, dim=-1)).clamp(min=-100.0, max=0.0)
                
                # Completion mask (shifted to align with per_token_logps)
                completion_mask = resp_mask[:, 1:]  # [chunk_len, seq-1]

                # 5. DR-GRPO per-token loss (TRL formulation)
                # With num_iterations=1, old_per_token_logps == per_token_logps.detach()
                # ratio = exp(log_p - log_p.detach()). Forward pass is 1.0, but backward pass retains gradients!
                adv_g = advantages[g, chunk_start:chunk_end]  # [chunk_len]
                ratio = torch.exp(per_token_logps - per_token_logps.detach())
                per_token_loss = -adv_g.unsqueeze(1) * ratio
                
                # 6. KL penalty (per-token, added to loss)
                raw_kl_val = 0.0
                if self.kl_coef > 0:
                    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        ref_out = self._ref_model(
                            inputs_embeds=inputs_embeds.detach(),
                            attention_mask=batch_mask,
                            position_ids=position_ids,
                            use_cache=False,
                            return_dict=True,
                        )
                        ref_logits = ref_out.logits[:, :-1, :]
                        ref_target_logits = ref_logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                        ref_per_token_logps = (ref_target_logits - torch.logsumexp(ref_logits, dim=-1)).clamp(min=-100.0, max=0.0)
                        
                    # Unbiased KL estimator: exp(ref - cur) - (ref - cur) - 1
                    kl_ratio = torch.clamp(ref_per_token_logps.detach() - per_token_logps, min=-20.0, max=20.0)
                    per_token_kl = torch.exp(kl_ratio) - kl_ratio - 1.0
                    per_token_loss = per_token_loss + self.kl_coef * per_token_kl
                    raw_kl_val = ((per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)).item()
                    del ref_out, ref_logits, ref_target_logits, ref_per_token_logps

                # 7. DR-GRPO normalization: divide by (B_total * max_completion_length)
                # B_total = G * B (total rollouts across all groups)
                loss_chunk = (per_token_loss * completion_mask).sum() / (G * B * self.max_completion_length)
                loss_chunk = loss_chunk / grad_accum_steps

                # 8. NaN guard
                if torch.isnan(loss_chunk) or torch.isinf(loss_chunk):
                    loss_chunk = torch.tensor(0.0, device=device, requires_grad=True)
                
                loss_chunk.backward()
                
                total_rollout_loss += loss_chunk.item() * grad_accum_steps
                total_kl_loss += raw_kl_val * chunk_len
                total_raw_kl += raw_kl_val * chunk_len
                
                # Clean up
                del out, logits, inputs_embeds, logits_shifted, position_ids
                del target_logits, per_token_logps, per_token_loss, loss_chunk
                torch.cuda.empty_cache()

        if self.kl_coef > 0 and self.offload_ref_model:
            self._ref_model.cpu()
            torch.cuda.empty_cache()

        return (
            torch.tensor(total_rollout_loss, device=device),
            torch.tensor(total_kl_loss, device=device),
            torch.tensor(total_raw_kl, device=device),
        )

    # -----------------------------------------------------------------------
    # Step 5 + 6: Identify τ+/τ- and extract h_T
    # -----------------------------------------------------------------------

    def select_best_worst(
        self,
        all_ids: List[torch.Tensor],
        all_masks: List[torch.Tensor],
        all_texts: List[List[str]],
        advantages: torch.Tensor,   # [G, batch]
        prompt_len: int,
    ) -> Tuple[
        torch.Tensor, torch.Tensor,   # τ+ ids, τ+ mask
        torch.Tensor, torch.Tensor,   # τ- ids, τ- mask
        List[List[str]], List[List[str]],  # τ+ texts, τ- texts
        torch.Tensor, torch.Tensor,   # τ+ response mask, τ- response mask
        torch.Tensor,                 # answer_token_positions in τ+
    ]:
        """
        Select per-batch-item τ+ (highest advantage) and τ- (lowest advantage).
        Pads τ+ and τ- to a uniform length within each set.
        """
        # Per batch item: argmax / argmin over G dim
        best_idx  = advantages.argmax(dim=0)  # [batch]
        worst_idx = advantages.argmin(dim=0)  # [batch]

        batch = all_ids[0].shape[0]
        device = all_ids[0].device

        # Gather selected sequences (variable length → pad to max)
        def gather_selected(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            selected_ids  = [all_ids[indices[i]][i]  for i in range(batch)]
            selected_mask = [all_masks[indices[i]][i] for i in range(batch)]
            # Pad to max length
            max_len = max(s.shape[0] for s in selected_ids)
            padded_ids  = torch.zeros(batch, max_len, dtype=torch.long, device=device)
            padded_mask = torch.zeros(batch, max_len, dtype=torch.long, device=device)
            for i, (s_ids, s_mask) in enumerate(zip(selected_ids, selected_mask)):
                L = s_ids.shape[0]
                padded_ids[i, :L]  = s_ids
                padded_mask[i, :L] = s_mask
            return padded_ids, padded_mask

        tau_pos_ids, tau_pos_mask = gather_selected(best_idx)
        tau_neg_ids, tau_neg_mask = gather_selected(worst_idx)

        # Gather corresponding texts
        tau_pos_texts = [all_texts[best_idx[i]][i]  for i in range(batch)]
        tau_neg_texts = [all_texts[worst_idx[i]][i] for i in range(batch)]

        # Response-only masks
        tau_pos_response = self._compute_response_mask(tau_pos_ids, prompt_len)
        tau_neg_response = self._compute_response_mask(tau_neg_ids, prompt_len)

        # </think> token positions in τ+
        think_end_pos = self._find_think_end_positions(tau_pos_ids)

        return (
            tau_pos_ids, tau_pos_mask,
            tau_neg_ids, tau_neg_mask,
            tau_pos_texts, tau_neg_texts,
            tau_pos_response, tau_neg_response,
            think_end_pos,
        )

    @torch.no_grad()
    def extract_think_end_hidden_state(
        self,
        tau_pos_ids: torch.Tensor,      # [batch, seq_pos]
        tau_pos_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        think_end_token_pos: torch.Tensor, # [batch]
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Separate forward pass through the UPDATED Teacher on τ+.
        Extracts the hidden state at the </think> token position.

        Called AFTER the Teacher's GRPO optimizer step so that h_T reflects
        the post-update weights — matching Algorithm 1's sequential ordering.

        Returns
        -------
        h_T : [batch, d_teacher]
        """
        inputs_embeds = self._build_input_embeds(
            tau_pos_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw
        )

        out = self.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=tau_pos_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        h_all = out.hidden_states[-1]   # [batch, seq, d]
        batch  = h_all.shape[0]

        h_T = h_all[
            torch.arange(batch, device=h_all.device),
            think_end_token_pos,
        ]   # [batch, d]

        return h_T

    # -----------------------------------------------------------------------
    # Full training step (orchestrates Steps 1–6)
    # -----------------------------------------------------------------------

    def training_step(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        ground_truth: dict,
        reward_fns: List[RewardFunction],
        reward_weights: Optional[List[float]],
        optimizer: torch.optim.Optimizer,
        tokenizer,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        is_accum_step: bool = True,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> RolloutBuffer:
        """
        Execute the complete Teacher GRPO step and return a populated RolloutBuffer.

        The RolloutBuffer carries τ+, τ-, advantages, answer positions, and h_T
        to student_losses.py for the Student update.

        Parameters
        ----------
        optimizer : Teacher's optimizer (separate from Student's)
        grad_clip : gradient norm clipping value

        Returns
        -------
        buffer : RolloutBuffer  (h_T is set and ready for L_distill)
        """

        assert self.end_think_token_id > 0, (
            "end_think_token_id not provided properly"
        )

        prompt_len = input_ids.shape[1]

        # --- Step 1: Generate G rollouts (no_grad) -------------------------
        all_ids, all_texts, all_masks = self.generate_rollouts(
            input_ids, pixel_values, image_grid_thw, attention_mask, tokenizer, pixel_values_videos, video_grid_thw
        )

        # --- Step 2: Score --------------------------------------------------
        rewards = self.score_rollouts(
            all_ids, all_texts,
            pixel_values, image_grid_thw,
            ground_truth, reward_fns, reward_weights,
            pixel_values_videos, video_grid_thw,
        )   # [G, batch]

        # --- Step 3: Advantages --------------------------------------------
        advantages = self.compute_advantages(rewards)   # [G, batch]

        # compute_grpo_loss now internally chunks the forward/backward passes 
        # to prevent OOM. Gradients are automatically accumulated.
        grpo_loss_val, kl_loss_val, raw_kl_val = self.compute_grpo_loss(
            all_ids, all_masks, advantages,
            pixel_values, image_grid_thw, prompt_len,
            grad_accum_steps=grad_accum_steps,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

        # KL is fixed — no adaptive adjustment. PPO-Clip is the primary policy constraint.
        if self.kl_coef > 0:
            mean_kl = raw_kl_val.item() / (self.G * input_ids.shape[0])
            if mean_kl > self.target_kl * 3.0:
                # Emergency log-only warning: KL is very large, indicating possible instability
                warnings.warn(
                    f"HIGH KL ALERT: mean_kl={mean_kl:.4f} >> target={self.target_kl}. "
                    f"PPO-Clip should have prevented this. Check for data anomalies.",
                    stacklevel=2,
                )
        
        if is_accum_step:
            grad_norm = nn.utils.clip_grad_norm_(self.vlm.parameters(), grad_clip)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                warnings.warn(f"Teacher gradients are NaN/Inf (norm={grad_norm}). Skipping optimizer step.", stacklevel=2)
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

        # --- Step 5: Select τ+ and τ- -------------------------------------
        (
            tau_pos_ids, tau_pos_mask,
            tau_neg_ids, tau_neg_mask,
            tau_pos_texts, tau_neg_texts,
            tau_pos_response, tau_neg_response,
            think_end_pos,
        ) = self.select_best_worst(
            all_ids, all_masks, all_texts, advantages, prompt_len
        )

        # --- Step 6: Extract h_T from UPDATED Teacher ----------------------
        h_T = self.extract_think_end_hidden_state(
            tau_pos_ids, tau_pos_mask,
            pixel_values, image_grid_thw,
            think_end_pos,
            pixel_values_videos, video_grid_thw,
        )

        # --- Pack into RolloutBuffer ---------------------------------------
        buffer = RolloutBuffer(
            rollout_ids=all_ids,
            rollout_texts=all_texts,
            attention_masks=all_masks,
            rewards=rewards,
            advantages=advantages,
            best_idx=advantages.argmax(dim=0),
            worst_idx=advantages.argmin(dim=0),
            tau_pos_ids=tau_pos_ids,
            tau_neg_ids=tau_neg_ids,
            tau_pos_mask=tau_pos_mask,
            tau_neg_mask=tau_neg_mask,
            tau_pos_response_mask=tau_pos_response,
            tau_neg_response_mask=tau_neg_response,
            answer_token_pos=think_end_pos,
            h_T=h_T,
            grpo_loss=grpo_loss_val.item(),
            dataset_source=ground_truth.get("dataset", None),
            kl_loss=kl_loss_val.item(),
            kl_coef=self.kl_coef,
        )

        return buffer

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    @staticmethod
    def log_rollout_stats(buffer: RolloutBuffer) -> dict:
        stats = {
            "grpo/reward_mean":    buffer.rewards.mean().item(),
            "grpo/reward_max":     buffer.rewards.max().item(),
        }
        
        if buffer.kl_loss is not None:
            stats["grpo/kl_loss"] = buffer.kl_loss
            
        if hasattr(buffer, "kl_coef") and buffer.kl_coef is not None:
            stats["grpo/kl_coef"] = buffer.kl_coef
            
        if buffer.dataset_source is not None:
            ds_rewards = {}
            for b_idx, ds in enumerate(buffer.dataset_source):
                if ds not in ds_rewards:
                    ds_rewards[ds] = []
                # Rewards is [G, B], mean across G for this batch item
                ds_rewards[ds].append(buffer.rewards[:, b_idx].mean().item())
            
            for ds, rews in ds_rewards.items():
                stats[f"grpo/reward_mean_{ds}"] = sum(rews) / len(rews)
                
        return stats

    def print_trainable_parameters(self):
        self.vlm.print_trainable_parameters()