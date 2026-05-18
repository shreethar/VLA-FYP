"""
latent_student.py
-----------------
Latent Student (Fθ) for ThinkFlow-VLA Stage 2.

Key mechanics:
  1. Encode visual+instruction prefix → KV cache + hidden states
  2. Autoregressive latent loop (M=6 steps):
       - Feed previous hidden state directly as next input embedding (NO vocab lookup)
       - Extract final-layer hidden state → this IS z_m ∈ R^d (NO projection head)
       - Repeat, accumulating z_1 ... z_M
  3. Append K=5 learnable spatial tokens → process in one parallel forward pass
  4. Spatial token output hidden states → SpatialMLP → 2D waypoints

Separate utility methods for:
  - <answer> token hidden state extraction  (L_distill)
  - Mid-layer (L/2) visual token features   (L_spatial / Spatial Forcing)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class SpatialMLP(nn.Module):
    """
    Projects K spatial token hidden states → 2D waypoints.
    Input:  [batch, K, d]
    Output: [batch, K, 2]   (normalized coordinates in [0, 1])
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Sigmoid(),   # keep waypoints in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [batch, K, 2]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LatentStudent(nn.Module):
    """
    Wraps Qwen2.5-VL-4B with LoRA to act as the Latent Student (Fθ).

    Parameters
    ----------
    model_name   : HuggingFace repo ID for the base VLM checkpoint
    M            : number of continuous reasoning latent vectors to generate
    K            : number of learnable spatial tokens
    lora_rank    : LoRA rank r
    lora_alpha   : LoRA scaling α
    lora_dropout : dropout on LoRA layers
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-4B-Instruct",
        M: int = 6,
        K: int = 5,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.M = M
        self.K = K

        # ------------------------------------------------------------------
        # 1. Load base VLM in bf16
        # ------------------------------------------------------------------
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Freeze vision encoder — only language layers train
        for param in base.visual.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. Wrap with LoRA (language transformer layers only)
        # ------------------------------------------------------------------
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        self.vlm = get_peft_model(base, lora_cfg)

        # Convenience references
        self.hidden_dim: int = self.vlm.config.hidden_size       # 2048 for 4B
        self.num_layers: int = self.vlm.config.num_hidden_layers  # 28 for 4B
        self.mid_layer_idx: int = self.num_layers // 2            # 14

        # ------------------------------------------------------------------
        # 3. K learnable spatial tokens  [K, d]
        #    Randomly initialized (std=0.02 matches typical embedding init)
        # ------------------------------------------------------------------
        self.spatial_tokens = nn.Parameter(
            torch.randn(K, self.hidden_dim) * 0.02
        )

        # ------------------------------------------------------------------
        # 4. SpatialMLP: spatial hidden states → 2D waypoints
        # ------------------------------------------------------------------
        self.spatial_mlp = SpatialMLP(self.hidden_dim)

        # Cache the image pad token id for visual mask construction
        self._image_token_id: int = self.vlm.config.image_token_id

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @property
    def _base_transformer(self):
        """Returns the underlying base LM (no LM head, no visual encoder)."""
        # Qwen2.5-VL: vlm.model is the LlamaModel-style transformer stack
        return self.vlm.model

    @property
    def _embed_tokens(self):
        return self._base_transformer.embed_tokens

    def _build_input_embeds(
        self,
        input_ids: torch.Tensor,           # [batch, seq]
        pixel_values: Optional[torch.Tensor],   # [total_patches, C, H, W]
        image_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Convert input_ids → embeddings, replacing <image_pad> positions
        with visual encoder output (same logic as Qwen2.5-VL's own forward).

        Returns: inputs_embeds [batch, seq, d]
        """
        inputs_embeds = self._embed_tokens(input_ids)  # [batch, seq, d]

        if pixel_values is not None:
            # Visual encoder is frozen; run in no_grad to save memory
            with torch.no_grad():
                image_embeds = self.vlm.visual(
                    pixel_values, grid_thw=image_grid_thw
                )  # [total_visual_tokens, d]

            image_mask = (input_ids == self._image_token_id)  # [batch, seq]
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[image_mask] = image_embeds.to(inputs_embeds.dtype)

        return inputs_embeds  # [batch, seq, d]

    # -----------------------------------------------------------------------
    # Core: prefix encoding
    # -----------------------------------------------------------------------

    def encode_prefix(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass over the visual+instruction prefix.
        Stores KV cache for efficient continuation in the latent loop.

        Returns
        -------
        prefix_last_hidden : [batch, d]
            Hidden state of the last prefix token.
            This is the seed embedding fed into latent step 1.
        past_key_values : tuple
            KV cache covering all prefix positions.
        """
        inputs_embeds = self._build_input_embeds(
            input_ids, pixel_values, image_grid_thw
        )

        outputs = self._base_transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

        # Last token hidden state is the seed for the latent loop
        prefix_last_hidden = outputs.last_hidden_state[:, -1, :]  # [batch, d]
        past_key_values = outputs.past_key_values

        return prefix_last_hidden, past_key_values

    # -----------------------------------------------------------------------
    # Core: latent + spatial generation
    # -----------------------------------------------------------------------

    def generate_latents(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Autoregressive latent generation followed by parallel spatial decoding.

        Loop logic
        ----------
        Seed  : last prefix hidden state h_{prefix} → input to step 1
        Step m: transformer(current_embed) → h_m = z_m  ← this IS the latent
                z_m fed directly as input_embed to step m+1 (NO vocab lookup)
        Total : M=6 continuous latent vectors z_1 … z_M

        Then  : K=5 spatial tokens processed in one parallel step → MLP → waypoints

        Returns
        -------
        latents       : List[Tensor[batch, d]], length M
        spatial_hidden: [batch, K, d]
        waypoints     : [batch, K, 2]  (values in [0,1])
        """
        batch_size = input_ids.shape[0]
        prefix_len  = input_ids.shape[1]
        device      = input_ids.device

        # ---- Encode prefix ------------------------------------------------
        seed_hidden, past_kv = self.encode_prefix(
            input_ids, pixel_values, image_grid_thw, attention_mask
        )

        # Attention mask starts at prefix length; we'll extend it each step
        # Shape: [batch, prefix_len]
        current_attn = attention_mask

        # The seed is the last prefix hidden state.
        # It becomes the input_embed for the FIRST latent step.
        current_embed = seed_hidden.unsqueeze(1)  # [batch, 1, d]

        # ---- Autoregressive latent loop -----------------------------------
        latents: List[torch.Tensor] = []

        for _ in range(self.M):
            # Extend attention mask by one position for this latent token
            new_col = torch.ones(
                batch_size, 1, device=device, dtype=current_attn.dtype
            )
            current_attn = torch.cat([current_attn, new_col], dim=1)
            # current_attn shape: [batch, prefix_len + step_index]

            # Single-step transformer forward (KV cache covers all prior positions)
            step_out = self._base_transformer(
                inputs_embeds=current_embed,    # [batch, 1, d]
                attention_mask=current_attn,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )

            # z_m = final-layer hidden state at this position
            z_m = step_out.last_hidden_state[:, 0, :]  # [batch, d]
            past_kv = step_out.past_key_values

            latents.append(z_m)

            # Feed z_m directly as next input embedding — bypasses vocabulary
            current_embed = z_m.unsqueeze(1)  # [batch, 1, d]

        # ---- Spatial tokens (parallel) ------------------------------------
        # Expand [K, d] → [batch, K, d]
        spatial_embeds = (
            self.spatial_tokens
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .to(dtype=current_embed.dtype, device=device)
        )

        # Extend attention mask for K spatial positions at once
        spatial_attn = torch.ones(
            batch_size, self.K, device=device, dtype=current_attn.dtype
        )
        current_attn = torch.cat([current_attn, spatial_attn], dim=1)

        # One parallel forward pass over all K spatial tokens
        spatial_out = self._base_transformer(
            inputs_embeds=spatial_embeds,   # [batch, K, d]
            attention_mask=current_attn,
            past_key_values=past_kv,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        spatial_hidden = spatial_out.last_hidden_state  # [batch, K, d]
        waypoints = self.spatial_mlp(spatial_hidden)    # [batch, K, 2]

        return latents, spatial_hidden, waypoints

    # -----------------------------------------------------------------------
    # Utility: <answer> token hidden state  (for L_distill)
    # -----------------------------------------------------------------------

    def get_answer_hidden_state(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        answer_token_positions: torch.Tensor,   # [batch] — int64 token indices
    ) -> torch.Tensor:
        """
        Standard full-sequence forward pass. Extracts the hidden state at the
        <answer> token position for each item in the batch.

        Used by student_losses.py to compute L_distill against the Teacher's
        cached h_T (extracted after the Teacher's GRPO update).

        Returns
        -------
        h_answer : [batch, d]
        """
        inputs_embeds = self._build_input_embeds(
            input_ids, pixel_values, image_grid_thw
        )

        outputs = self._base_transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        h_all = outputs.last_hidden_state  # [batch, seq, d]

        # Index per-sample answer position
        batch_idx = torch.arange(batch_size := h_all.shape[0], device=h_all.device)
        h_answer = h_all[batch_idx, answer_token_positions]  # [batch, d]

        return h_answer

    # -----------------------------------------------------------------------
    # Utility: mid-layer visual features  (for L_spatial / Spatial Forcing)
    # -----------------------------------------------------------------------

    def get_mid_layer_visual_features(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with output_hidden_states=True.
        Extracts hidden states at layer L/2 (layer 14 for the 28-layer 4B model)
        at visual token positions only.

        This is the x_V used in:
            L_spatial = -CosSim(MLP(x_V), VGGT(I))

        Returns
        -------
        x_V : [batch, num_visual_tokens, d]
        """
        inputs_embeds = self._build_input_embeds(
            input_ids, pixel_values, image_grid_thw
        )

        outputs = self._base_transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,  # Need all layers to index L/2
            return_dict=True,
        )

        # hidden_states is a tuple of length (num_layers + 1):
        #   index 0  = embedding layer output
        #   index i  = output of transformer layer i-1
        # → layer L/2 output lives at index (mid_layer_idx + 1)
        mid_hidden = outputs.hidden_states[self.mid_layer_idx + 1]  # [batch, seq, d]

        # Mask to visual token positions
        image_mask = (input_ids == self._image_token_id)  # [batch, seq]

        # Number of visual tokens per sample (assumed equal within a batch)
        num_visual = image_mask[0].sum().item()

        x_V = mid_hidden[image_mask].view(
            input_ids.shape[0], num_visual, self.hidden_dim
        )  # [batch, num_visual_tokens, d]

        return x_V

    # -----------------------------------------------------------------------
    # Standard forward (for non-latent use cases, e.g. Stage 1 SFT eval)
    # -----------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            labels=labels,
        )

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def print_trainable_parameters(self):
        self.vlm.print_trainable_parameters()
        spatial_params = self.spatial_tokens.numel()
        mlp_params = sum(p.numel() for p in self.spatial_mlp.parameters())
        print(f"  spatial_tokens: {spatial_params:,} params")
        print(f"  spatial_mlp:    {mlp_params:,} params")