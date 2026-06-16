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
    rollout_ids:       List[torch.Tensor]
    rollout_texts:     List[List[str]]    # decoded; outer=G, inner=batch
    attention_masks:   List[torch.Tensor]

    # Per-rollout scalars: shape [G, batch]
    rewards:           torch.Tensor
    advantages:        torch.Tensor

    # Best / worst indices into the G dimension (per batch item)
    best_idx:          torch.Tensor       # [batch]  int64
    worst_idx:         torch.Tensor       # [batch]  int64

    # τ+ and τ- token ids and masks (best/worst selected per batch item)
    tau_pos_ids:       torch.Tensor       # [batch, seq_pos]
    tau_neg_ids:       torch.Tensor       # [batch, seq_neg]
    tau_pos_mask:      torch.Tensor
    tau_neg_mask:      torch.Tensor

    # Response-only masks (1 on generated tokens, 0 on prompt)
    tau_pos_response_mask: torch.Tensor   # [batch, seq_pos]
    tau_neg_response_mask: torch.Tensor   # [batch, seq_neg]

    # Answer token positions in τ+ (for L_distill)
    answer_token_pos:  torch.Tensor       # [batch]  int64

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
        gen_temperature: float = 0.9,
        gen_max_new_tokens: int = 512,
        kl_coef: float = 0.0,
        target_kl: float = 0.02,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.G = G
        self.end_think_token_id = answer_token_id  # Rename variable internally
        self.gen_temperature  = gen_temperature
        self.gen_max_new_tokens = gen_max_new_tokens
        self.kl_coef = kl_coef
        self.target_kl = target_kl

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
    def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Group-relative advantage normalisation within each batch item's G rollouts.

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

        # Warning for near-zero variance
        low_var_mask = std.squeeze(0) < (eps * 10)
        if low_var_mask.any():
            n = low_var_mask.sum().item()
            warnings.warn(
                f"{n}/{low_var_mask.shape[0]} batch items have near-zero reward "
                f"variance across G rollouts. Advantages will be ~0 for these items "
                f"(no learning signal). Consider increasing gen_temperature or "
                f"checking your reward function.",
                stacklevel=2,
            )
        return (rewards - mean) / (std + eps)      # [G, batch]

    # -----------------------------------------------------------------------
    # Step 4: GRPO policy-gradient loss  +  Teacher update
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
        GRPO policy-gradient loss (REINFORCE with group-relative baseline).

        For each rollout g and each response token t:
            contrib_g = A_g * (1/|response_g|) * Σ_t log π_θ(token_t | ctx, τ_g,<t)

        L_GRPO = -mean_over_batch( mean_over_G( contrib_g ) )

        Optional KL penalty if kl_coef > 0:
            L_total = L_GRPO + kl_coef * KL(π_θ || π_ref)

        Returns
        -------
        loss : scalar
        """
        if self.kl_coef > 0:
            self._ensure_ref_model()

        G     = len(all_ids)
        B     = all_ids[0].shape[0]
        device = all_ids[0].device

        # ── Pre-compute per-batch-item image/video ownership ──────────────
        # image_grid_thw / video_grid_thw only contain entries for samples
        # that actually have images / videos.  We must map b_idx → the
        # correct row in those tensors (not a 1:1 b_idx mapping).
        image_token_id = getattr(self.vlm.config, "image_token_id", None)
        video_token_id = getattr(self.vlm.config, "video_token_id", 248057)

        # Use the FIRST rollout's prompt portion (identical across all G)
        # to detect which batch items carry image vs video tokens.
        ref_ids = all_ids[0]  # [B, seq]

        # has_image[b] / has_video[b] = True if batch item b has those tokens
        has_image = [False] * B
        has_video = [False] * B
        if pixel_values is not None and image_token_id is not None:
            for b in range(B):
                has_image[b] = (ref_ids[b] == image_token_id).any().item()
        if pixel_values_videos is not None:
            for b in range(B):
                has_video[b] = (ref_ids[b] == video_token_id).any().item()

        # ── Step B: Chunked Forward & Backward Pass ───────────────────────
        # By processing one rollout group (B sequences) at a time and calling 
        # .backward(), PyTorch instantly frees the massive activation memory and 
        # gradient tensors. Peak VRAM becomes B instead of G*B!
        
        total_rollout_loss = 0.0
        total_kl_loss = 0.0
        total_raw_kl = 0.0
        
        if self.kl_coef > 0:
            self._ref_model.to(device)

        for g in range(G):
            for b_idx in range(B):
                # 1. Slice rollout to EXACTLY 1 sequence (Absolute minimum peak VRAM)
                batch_id = all_ids[g][b_idx:b_idx+1]      # [1, seq_g]
                batch_mask = all_masks[g][b_idx:b_idx+1]  # [1, seq_g]
                
                # Response mask & lengths
                resp_mask = torch.zeros_like(batch_id, dtype=torch.float)
                resp_mask[:, prompt_len:] = (batch_id[:, prompt_len:] != 0).float()
                resp_lens = resp_mask.sum(dim=-1).clamp(min=1)  # [1]

                # 2. Correctly slice the flattened image/video patches
                #    Only pass pixel_values if THIS batch item has image tokens.
                if pixel_values is not None and has_image[b_idx]:
                    # Count how many image-bearing items precede b_idx
                    img_row = sum(has_image[:b_idx])
                    thw = image_grid_thw[img_row:img_row+1]  # [1, 3]
                    num_patches = thw.prod(dim=-1).item()
                    offset = int(image_grid_thw[:img_row].prod(dim=-1).sum().item()) if img_row > 0 else 0
                    pv = pixel_values[offset : offset + num_patches]
                else:
                    pv = None
                    thw = None

                if pixel_values_videos is not None and has_video[b_idx]:
                    vid_row = sum(has_video[:b_idx])
                    v_thw = video_grid_thw[vid_row:vid_row+1]
                    num_patches = v_thw.prod(dim=-1).item()
                    offset = int(video_grid_thw[:vid_row].prod(dim=-1).sum().item()) if vid_row > 0 else 0
                    pv_v = pixel_values_videos[offset : offset + num_patches]
                else:
                    pv_v = None
                    v_thw = None

                # 3. Forward pass for just ONE sequence
                inputs_embeds = self._build_input_embeds(batch_id, pv, thw, pv_v, v_thw)
                
                # Explicitly compute position_ids to bypass Qwen3.5 RoPE bug when inputs_embeds is used
                position_ids = batch_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(batch_mask == 0, 1)

                out = self.vlm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                logits = out.logits.float()  # [1, seq_g, vocab]

                # 4. Log-probs (with numerical stability clamps on log-probs only)
                target_ids = batch_id[:, 1:]
                logits_shifted = logits[:, :-1, :]
                target_logits = logits_shifted.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                token_log_p = target_logits - torch.logsumexp(logits_shifted, dim=-1) # [1, seq-1]
                # Clamp log-probs to valid range (log-probs should be ≤ 0)
                token_log_p = token_log_p.clamp(min=-100.0, max=0.0)

                # Mean log-prob per response
                resp_mask_shifted = resp_mask[:, 1:]
                mean_log_p = (token_log_p * resp_mask_shifted).sum(dim=-1) / resp_lens  # [1]

                # 5. Rollout Loss
                adv_g = advantages[g, b_idx:b_idx+1]  # [1]
                loss_gb = -(adv_g * mean_log_p).mean() / (G * B * grad_accum_steps)
                
                # 6. KL Loss
                kl_gb = torch.tensor(0.0, device=device)
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
                        ref_token_p = (ref_target_logits - torch.logsumexp(ref_logits, dim=-1)).clamp(min=-100.0, max=0.0)
                        
                    kl = (token_log_p - ref_token_p.detach()) * resp_mask_shifted
                    # Clamp individual token KL to be non-negative to avoid negative KL exploits
                    kl = torch.clamp(kl, min=0.0)
                    kl_per_token = kl.sum(dim=-1) / resp_lens
                    raw_kl_val = kl_per_token.mean().item()
                    kl_gb = (self.kl_coef * kl_per_token.mean()) / (G * B * grad_accum_steps)

                # 7. NaN guard: skip this chunk if loss is NaN/Inf
                chunk_loss = loss_gb + kl_gb
                if torch.isnan(chunk_loss) or torch.isinf(chunk_loss):
                    # Zero out gradients from this corrupted chunk
                    chunk_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    chunk_loss.backward()  # no-op backward to keep graph consistent
                else:
                    chunk_loss.backward()
                
                total_rollout_loss += (loss_gb.item() * G * B * grad_accum_steps) if not torch.isnan(loss_gb) else 0.0
                total_kl_loss += (kl_gb.item() * G * B * grad_accum_steps) if not torch.isnan(kl_gb) else 0.0
                total_raw_kl += raw_kl_val
                
                # Explicitly delete massive tensors BEFORE next micro-batch!
                del out, logits, inputs_embeds, logits_shifted, position_ids
                del target_logits, token_log_p, chunk_loss, loss_gb
                if self.kl_coef > 0:
                    del ref_out, ref_logits, ref_target_logits, ref_token_p, kl_gb
                
                torch.cuda.empty_cache()

        if self.kl_coef > 0:
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

        # Adaptive KL adjustment
        if self.kl_coef > 0:
            # G * B is the total number of sequences in the batch
            mean_kl = raw_kl_val.item() / (self.G * input_ids.shape[0])
            old_kl_coef = self.kl_coef
            if mean_kl > self.target_kl * 1.5:
                self.kl_coef = min(self.kl_coef * 1.5, 0.5)
            elif mean_kl < self.target_kl / 1.5:
                self.kl_coef = max(self.kl_coef / 1.5, 0.001)
            
            if self.kl_coef != old_kl_coef:
                warnings.warn(
                    f"Adaptive KL: mean_kl={mean_kl:.6f} vs target={self.target_kl} "
                    f"-> kl_coef adjusted {old_kl_coef:.5f} -> {self.kl_coef:.5f}",
                    stacklevel=2
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