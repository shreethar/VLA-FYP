# Qwen3.5 Migration — Changes Summary

## Files Rewritten

### 1. `latent_student.py` — Latent Student (Fθ)

```diff:latent_student.py
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
===
"""
latent_student.py
-----------------
Latent Student (Fθ) for ThinkFlow-VLA Stage 2.

Base model: Qwen3.5-4B (hybrid Gated DeltaNet + Attention architecture)

Key mechanics:
  1. Build input embeddings with visual token injection
  2. Autoregressive latent loop (M=6 steps):
       - Feed full growing sequence through the model
       - Extract final-position hidden state → this IS z_m ∈ R^d (NO projection head)
       - Concatenate z_m to the sequence, repeat
       - Accumulate z_1 ... z_M
  3. Append K=5 learnable spatial tokens → process in one parallel forward pass
  4. Spatial token output hidden states → SpatialMLP → 2D waypoints

Architecture note:
  Qwen3.5 uses a hybrid 3:1 pattern of Gated DeltaNet (linear attention)
  and standard Gated Attention layers.  DeltaNet layers use recurrent state
  instead of KV cache, so we use concat-based sequence growth for the
  latent loop (matching the validated notebook pattern) rather than KV caching.

Separate utility methods for:
  - <answer> token hidden state extraction  (L_distill)
  - Mid-layer (L/2) visual token features   (L_spatial / Spatial Forcing)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from transformers import AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model


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
    Wraps Qwen3.5-4B with LoRA to act as the Latent Student (Fθ).

    Architecture
    ------------
    Qwen3.5-4B is a hybrid VLM with:
      - 32 transformer layers (3:1 Gated DeltaNet : standard Attention)
      - Hidden dim: 2560
      - Vocab: 248,320
      - Vision encoder: Qwen3_5VisionModel with patch merger → d=2560

    Model hierarchy (after PEFT wrapping):
      self.vlm                          → PeftModel
      self.vlm.model                    → Qwen3_5ForConditionalGeneration
      self.vlm.model.model              → Qwen3_5Model  (.visual, .language_model)
      self.vlm.model.model.language_model → Qwen3_5TextModel (.embed_tokens, .layers, .norm)
      self.vlm.model.lm_head            → Linear

    Because DeltaNet layers use recurrent state (not KV cache), the latent
    loop uses concat-based sequence growth following the validated notebook
    pattern rather than KV caching.

    Parameters
    ----------
    model_name   : HuggingFace repo ID or local path for the VLM checkpoint
    M            : number of continuous reasoning latent vectors to generate
    K            : number of learnable spatial tokens
    lora_rank    : LoRA rank r
    lora_alpha   : LoRA scaling α
    lora_dropout : dropout on LoRA layers
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-4B",
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
        base = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        # ------------------------------------------------------------------
        # 2. Wrap with LoRA
        # ------------------------------------------------------------------
        # Target modules span both layer types in the hybrid architecture:
        #   - Standard Attention (every 4th layer): q_proj, k_proj, v_proj, o_proj
        #   - Gated DeltaNet (other layers):        out_proj, in_proj_qkv, in_proj_z
        #   - MLP (all layers):                     gate_proj, up_proj, down_proj
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                # Standard attention projections
                "q_proj", "k_proj", "v_proj", "o_proj",
                # Gated DeltaNet main projections
                "out_proj", "in_proj_qkv", "in_proj_z",
                # Feed-forward network
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        self.vlm = get_peft_model(base, lora_cfg)

        # Convenience references
        # Qwen3.5 stores LM config under text_config
        text_cfg = self.vlm.config.text_config
        self.hidden_dim: int = text_cfg.hidden_size        # 2560 for 4B
        self.num_layers: int = text_cfg.num_hidden_layers  # 32 for 4B
        self.mid_layer_idx: int = self.num_layers // 2     # 16

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
    def _inner_model(self):
        """
        Returns the Qwen3_5Model (contains .visual and .language_model).

        Hierarchy: self.vlm (PeftModel) → .model (Qwen3_5ForConditionalGeneration)
                   → .model (Qwen3_5Model)

        Calling _inner_model(inputs_embeds=...) routes through .language_model
        which has LoRA layers physically injected by PEFT.
        """
        return self.vlm.model.model

    @property
    def _embed_tokens(self):
        """Returns the token embedding layer."""
        return self._inner_model.language_model.embed_tokens

    def _build_input_embeds(
        self,
        input_ids: torch.Tensor,                # [batch, seq]
        pixel_values: Optional[torch.Tensor],    # preprocessed image tensor
        image_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Convert input_ids → embeddings, replacing <image_pad> positions
        with visual encoder output (same logic as Qwen3.5's own forward).

        Returns: inputs_embeds [batch, seq, d]
        """
        inputs_embeds = self._embed_tokens(input_ids)  # [batch, seq, d]

        if pixel_values is not None:
            # Visual encoder is frozen; run in no_grad to save memory
            with torch.no_grad():
                image_embeds = self._inner_model.visual(
                    pixel_values, grid_thw=image_grid_thw
                )  # [total_visual_tokens, d]

            image_mask = (input_ids == self._image_token_id)  # [batch, seq]
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[image_mask] = image_embeds.to(inputs_embeds.dtype)

        return inputs_embeds  # [batch, seq, d]

    def _forward_inner(
        self,
        inputs_embeds: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        """
        Run inputs_embeds through the inner model (Qwen3_5Model).
        This routes to the language_model (Qwen3_5TextModel) internally.

        Matches the notebook pattern: model.model(inputs_embeds=..., ...)
        """
        return self._inner_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    # -----------------------------------------------------------------------
    # Core: latent + spatial generation
    # -----------------------------------------------------------------------

    def generate_latents(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Autoregressive latent generation followed by parallel spatial decoding.

        Uses concat-based sequence growth (NOT KV cache) because Qwen3.5's
        Gated DeltaNet layers use recurrent state instead of KV cache.
        This matches the validated notebook pattern.

        Loop logic
        ----------
        1. Build prefix embeddings (with visual tokens injected)
        2. For m = 1..M:
             forward(full_sequence) → z_m = last_hidden_state[:, -1, :]
             concat z_m to sequence
        3. Concat K spatial token embeddings
        4. forward(full_sequence) → spatial hidden states → MLP → waypoints

        Returns
        -------
        latents       : List[Tensor[batch, d]], length M
        spatial_hidden: [batch, K, d]
        waypoints     : [batch, K, 2]  (values in [0,1])
        """
        device = input_ids.device

        # ---- Build prefix embeddings (with visual token injection) --------
        current_embeds = self._build_input_embeds(
            input_ids, pixel_values, image_grid_thw
        )  # [batch, prefix_len, d]

        # ---- Autoregressive latent loop -----------------------------------
        # Concat-based: each step feeds the full growing sequence
        latents: List[torch.Tensor] = []

        for _ in range(self.M):
            outputs = self._forward_inner(current_embeds)
            # z_m = final-position hidden state
            z_m = outputs.last_hidden_state[:, -1:, :]  # [batch, 1, d]
            latents.append(z_m.squeeze(1))               # [batch, d]

            # Grow the sequence by concatenating z_m
            current_embeds = torch.cat([current_embeds, z_m], dim=1)

        # ---- Spatial tokens (parallel) ------------------------------------
        batch_size = input_ids.shape[0]
        # Expand [K, d] → [batch, K, d]
        spatial_embeds = (
            self.spatial_tokens
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .to(dtype=current_embeds.dtype, device=device)
        )

        # Concat spatial tokens to the full sequence
        full_embeds = torch.cat([current_embeds, spatial_embeds], dim=1)

        # One forward pass over the entire sequence including spatial tokens
        spatial_out = self._forward_inner(full_embeds)

        # Extract spatial token hidden states (last K positions)
        spatial_hidden = spatial_out.last_hidden_state[:, -self.K:, :]  # [batch, K, d]
        waypoints = self.spatial_mlp(spatial_hidden)                     # [batch, K, 2]

        return latents, spatial_hidden, waypoints

    # -----------------------------------------------------------------------
    # Utility: <answer> token hidden state  (for L_distill)
    # -----------------------------------------------------------------------

    def get_answer_hidden_state(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
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

        outputs = self._forward_inner(inputs_embeds)

        h_all = outputs.last_hidden_state  # [batch, seq, d]

        # Index per-sample answer position
        batch_idx = torch.arange(h_all.shape[0], device=h_all.device)
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
    ) -> torch.Tensor:
        """
        Forward pass with output_hidden_states=True.
        Extracts hidden states at layer L/2 (layer 16 for the 32-layer 4B model)
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

        outputs = self._forward_inner(inputs_embeds, output_hidden_states=True)

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
        print(f"  spatial_mlp:    {mlp_params:,} params")
```

**Key changes:**

| Aspect | Before (Qwen2.5-VL) | After (Qwen3.5) |
|---|---|---|
| Model class | `Qwen2_5_VLForConditionalGeneration` | `AutoModelForImageTextToText` |
| Default model | `Qwen/Qwen2.5-VL-4B-Instruct` | `Qwen/Qwen3.5-4B` |
| Hidden dim | 2048 | **2560** |
| Layers | 28 | **32** |
| Mid-layer idx | 14 | **16** |
| Latent loop | KV-cache based (`use_cache=True`) | **Concat-based** (notebook pattern) |
| LoRA targets | `q/k/v/o/gate/up/down_proj` | + **`out_proj`, `in_proj_qkv`, `in_proj_z`** (DeltaNet) |
| Model hierarchy | `vlm.model` (flat) | `vlm.model.model` → `.language_model` |
| Config access | `vlm.config.hidden_size` | `vlm.config.text_config.hidden_size` |

> [!IMPORTANT]
> The latent loop now follows the **exact notebook pattern**: full sequence concat at each step instead of KV caching. This is necessary because Gated DeltaNet layers use recurrent state, not KV cache.

### 2. `verbalizer.py` — Verbalizer (Vψ)

```diff:verbalizer.py
"""
verbalizer.py
-------------
Verbalizer (Vψ) for ThinkFlow-VLA Stage 2.

Architecture
------------
Base: Qwen/Qwen3-0.6B with LoRA rank=32 on attention layers.

At EVERY transformer layer, a new CrossAttentionBlock is inserted:
    h_l  = OriginalTransformerLayer_l(h_{l-1})          ← untouched SA + FFN
    h_l  = CrossAttentionBlock_l(Q=h_l, K=z, V=z)      ← new CA reads Student latents

z = stack of Student's M=6 latent vectors, shape [batch, M, d_student].
Since d_student (2048) ≠ d_verbalizer, each CA block has its own K/V projection.

Training schedule (controlled externally by train_stage2.py):
  Steps 0 – 3000  : warm-up  — CA blocks + LoRA trainable, LM loss on τ+
  Steps 3000 – 4500: frozen  — all Vψ params frozen, DPO gradient flows into Student

Loss functions implemented here:
  compute_lm_loss    : cross-entropy on τ+ tokens     (warm-up phase)
  compute_dpo_loss   : DPO preference loss on τ+/τ−   (both phases, but only updates
                       Vψ params during warm-up; Student params always)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


# ---------------------------------------------------------------------------
# Cross-Attention Block (one per Verbalizer transformer layer)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    A single cross-attention block that reads M Student latent vectors.

    Q  = hidden states from the current Verbalizer layer  [batch, seq, d_verb]
    K,V = Student latents z projected to d_verb            [batch, M,   d_verb]

    Output replaces the query sequence via residual + pre-norm:
        h = LayerNorm(h + MultiheadAttn(Q=h, K=k, V=v))
    """

    def __init__(
        self,
        query_dim: int,   # Verbalizer hidden size  (d_verb)
        kv_dim: int,      # Student hidden size     (d_student = 2048)
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert query_dim % num_heads == 0, (
            f"query_dim {query_dim} must be divisible by num_heads {num_heads}"
        )

        # Project Student latents into Verbalizer's space for K and V
        self.k_proj = nn.Linear(kv_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, query_dim, bias=False)

        # Standard multi-head cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pre-norm on queries (applied before attention, following modern practice)
        self.q_norm = nn.LayerNorm(query_dim, eps=1e-6)

        # Post-norm on residual output
        self.out_norm = nn.LayerNorm(query_dim, eps=1e-6)

    def forward(
        self,
        hidden: torch.Tensor,    # [batch, seq, d_verb]
        latents: torch.Tensor,   # [batch, M, d_student]
    ) -> torch.Tensor:
        # Project latents → K and V in verbalizer space
        k = self.k_proj(latents)  # [batch, M, d_verb]
        v = self.v_proj(latents)  # [batch, M, d_verb]

        # Normalize queries before attention
        q = self.q_norm(hidden)   # [batch, seq, d_verb]

        # Cross-attention: every verbalizer token attends to all M latents
        ca_out, _ = self.attn(q, k, v)  # [batch, seq, d_verb]

        # Residual connection + post-norm
        return self.out_norm(hidden + ca_out)  # [batch, seq, d_verb]


# ---------------------------------------------------------------------------
# Verbalizer
# ---------------------------------------------------------------------------

class Verbalizer(nn.Module):
    """
    Qwen3-0.6B with per-layer cross-attention blocks conditioned on Student latents.

    Parameters
    ----------
    model_name     : HuggingFace repo ID for the 0.6B base model
    student_hidden : hidden size of the Student VLM (d_student = 2048)
    lora_rank      : LoRA rank for the base attention layers
    lora_alpha     : LoRA scaling
    ca_num_heads   : attention heads inside each CrossAttentionBlock
    ca_dropout     : dropout in cross-attention (0 during distillation)
    dpo_beta       : β temperature for DPO loss
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        student_hidden: int = 2048,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        ca_num_heads: int = 8,
        ca_dropout: float = 0.0,
        dpo_beta: float = 0.1,
    ):
        super().__init__()
        self.dpo_beta = dpo_beta

        # ------------------------------------------------------------------
        # 1. Load Qwen3-0.6B
        # ------------------------------------------------------------------
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # ------------------------------------------------------------------
        # 2. Wrap with LoRA (rank=32, attention layers only)
        # ------------------------------------------------------------------
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.lm = get_peft_model(base, lora_cfg)

        # Infer verbalizer hidden dim and layer count from config
        self.hidden_dim: int = self.lm.config.hidden_size
        self.num_layers: int = self.lm.config.num_hidden_layers

        # ------------------------------------------------------------------
        # 3. Insert one CrossAttentionBlock per transformer layer
        # ------------------------------------------------------------------
        self.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(
                query_dim=self.hidden_dim,
                kv_dim=student_hidden,
                num_heads=ca_num_heads,
                dropout=ca_dropout,
            )
            for _ in range(self.num_layers)
        ])

        # Freeze tracking
        self._frozen: bool = False

    # -----------------------------------------------------------------------
    # Internal: manual layer-by-layer forward with CA injection
    # -----------------------------------------------------------------------

    def _forward_with_latents(
        self,
        input_ids: torch.Tensor,        # [batch, seq]
        attention_mask: torch.Tensor,   # [batch, seq]
        latents: torch.Tensor,          # [batch, M, d_student]  — stacked z_1…z_M
        labels: Optional[torch.Tensor] = None,  # [batch, seq] for LM loss
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Manual transformer forward that injects CA after every layer.

        Returns
        -------
        logits : [batch, seq, vocab_size]
        loss   : scalar CE loss if labels provided, else None
        """
        # Access the inner model layers
        # Qwen3 structure: lm.model.model.embed_tokens / .layers / .norm
        transformer = self.lm.model   # the base LlamaModel-style stack

        # --- Embedding layer ---
        hidden = transformer.embed_tokens(input_ids)  # [batch, seq, d_verb]

        # --- Build causal attention mask for manual iteration ---
        # HuggingFace models accept the 2D bool mask and handle causal internally,
        # but for manual iteration we pass it through each layer directly.
        # We'll build a 4D mask matching what the layers expect.
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Rotary position embeddings (Qwen3 uses RoPE)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.long()

        # Cache for RoPE cos/sin (shared across layers)
        cache_position = torch.arange(seq_len, device=device)
        
        # Build the 4D causal mask Qwen3 layers expect
        causal_mask = transformer._update_causal_mask(
            attention_mask,
            hidden,
            cache_position,
            past_key_values=None,
            output_attentions=False,
        )

        # --- Layer-by-layer forward with CA injection ---
        for layer_idx, layer in enumerate(transformer.layers):
            # Standard transformer layer (SA + FFN)
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
            )
            hidden = layer_out[0]  # [batch, seq, d_verb]

            # Cross-attention: read Student latents
            # Gradient flows through latents back to Student when not detached
            hidden = self.ca_blocks[layer_idx](hidden, latents)

        # --- Final norm ---
        hidden = transformer.norm(hidden)  # [batch, seq, d_verb]

        # --- LM head ---
        logits = self.lm.lm_head(hidden)  # [batch, seq, vocab_size]

        # --- Optional LM loss ---
        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def _compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,       # [batch, seq]
        attention_mask: torch.Tensor,  # [batch, seq]
        latents: torch.Tensor,         # [batch, M, d_student]
        response_mask: torch.Tensor,   # [batch, seq] — 1 on response tokens only
    ) -> torch.Tensor:
        """
        Compute per-sequence sum of log-probabilities over response tokens.

        Used by compute_dpo_loss to get log π_ψ(τ | z).

        Returns
        -------
        log_probs : [batch]  — sum of token log-probs over response positions
        """
        logits, _ = self._forward_with_latents(input_ids, attention_mask, latents)

        # log-softmax over vocab
        log_probs_all = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]

        # Shift to align: logits at position i predict token i+1
        # Predicted log-prob of token i+1 is log_probs_all[:, i, token_{i+1}]
        shift_log_probs = log_probs_all[:, :-1, :]         # [batch, seq-1, vocab]
        shift_labels    = input_ids[:, 1:]                  # [batch, seq-1]
        shift_mask      = response_mask[:, 1:]              # [batch, seq-1]

        # Gather the log-prob of each ground-truth token
        token_log_probs = shift_log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)  # [batch, seq-1]

        # Sum over response positions only
        seq_log_probs = (token_log_probs * shift_mask).sum(dim=-1)  # [batch]

        return seq_log_probs

    # -----------------------------------------------------------------------
    # Public loss functions
    # -----------------------------------------------------------------------

    def compute_lm_loss(
        self,
        input_ids: torch.Tensor,       # [batch, seq]  — τ+ sequence
        attention_mask: torch.Tensor,
        latents: torch.Tensor,         # [batch, M, d_student]  — Student z (DETACHED during warm-up)
        labels: torch.Tensor,          # [batch, seq]  — τ+ with -100 on prefix positions
    ) -> torch.Tensor:
        """
        Warm-up loss: standard cross-entropy on τ+ tokens.
        Trains the CA blocks and LoRA to learn to read Student latents.

        NOTE: during warm-up, pass latents.detach() so gradients do NOT
        flow back into the Student yet. The Student is updated by L_distill
        and L_ans only during warm-up.

        Returns
        -------
        lm_loss : scalar
        """
        _, loss = self._forward_with_latents(
            input_ids, attention_mask, latents, labels=labels
        )
        return loss

    def compute_dpo_loss(
        self,
        pos_input_ids: torch.Tensor,   # [batch, seq] — τ+ tokenized
        neg_input_ids: torch.Tensor,   # [batch, seq] — τ− tokenized
        pos_attention_mask: torch.Tensor,
        neg_attention_mask: torch.Tensor,
        latents: torch.Tensor,         # [batch, M, d_student] — Student z (NO detach here)
        pos_response_mask: torch.Tensor,  # [batch, seq] — 1 on τ+ response tokens
        neg_response_mask: torch.Tensor,  # [batch, seq] — 1 on τ− response tokens
        ref_pos_log_probs: Optional[torch.Tensor] = None,  # [batch] reference model log-probs
        ref_neg_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        DPO preference loss.

        After Verbalizer is frozen (step > 3000), all gradients from this loss
        flow through the latents tensor back into the Student's parameters.

        DPO objective:
            L_DPO = -E[ log σ( β * (log π(τ+|z) − log π_ref(τ+|z))
                                 − β * (log π(τ−|z) − log π_ref(τ−|z)) ) ]

        If reference log-probs not provided (common simplification), reduces to:
            L_DPO = -E[ log σ( β * (log π(τ+|z) − log π(τ−|z)) ) ]

        Parameters
        ----------
        ref_pos_log_probs / ref_neg_log_probs:
            Pass pre-computed reference model log-probs if using a reference
            policy (e.g., initial Verbalizer checkpoint). Pass None to use
            the simplified reference-free variant.

        Returns
        -------
        dpo_loss : scalar
        metrics  : dict with reward margin and accuracy for logging
        """
        # Log-probs under current Verbalizer policy
        # latents NOT detached → gradient flows into Student when Vψ is frozen
        log_pi_pos = self._compute_sequence_log_probs(
            pos_input_ids, pos_attention_mask, latents, pos_response_mask
        )  # [batch]
        log_pi_neg = self._compute_sequence_log_probs(
            neg_input_ids, neg_attention_mask, latents, neg_response_mask
        )  # [batch]

        # DPO reward margins
        if ref_pos_log_probs is not None and ref_neg_log_probs is not None:
            # Full DPO: subtract reference log-probs
            pi_log_ratios_pos = log_pi_pos - ref_pos_log_probs
            pi_log_ratios_neg = log_pi_neg - ref_neg_log_probs
        else:
            # Simplified (reference-free) DPO
            pi_log_ratios_pos = log_pi_pos
            pi_log_ratios_neg = log_pi_neg

        reward_margin = self.dpo_beta * (pi_log_ratios_pos - pi_log_ratios_neg)

        # DPO loss: -log sigmoid(reward_margin)
        dpo_loss = -F.logsigmoid(reward_margin).mean()

        # Logging metrics
        with torch.no_grad():
            metrics = {
                "dpo_loss":       dpo_loss.item(),
                "reward_margin":  reward_margin.mean().item(),
                "dpo_accuracy":   (reward_margin > 0).float().mean().item(),
                "log_pi_pos":     log_pi_pos.mean().item(),
                "log_pi_neg":     log_pi_neg.mean().item(),
            }

        return dpo_loss, metrics

    # -----------------------------------------------------------------------
    # Warm-up → freeze transition
    # -----------------------------------------------------------------------

    def freeze_for_student_training(self):
        """
        Called at step 3000.

        Freezes ALL Verbalizer parameters (base model + LoRA + CA blocks).
        After this, DPO gradients flow through the latents tensor only,
        updating the Student's weights — not the Verbalizer's.
        """
        if self._frozen:
            return

        for param in self.parameters():
            param.requires_grad = False

        self._frozen = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Verbalizer] Frozen. Trainable params remaining: {trainable:,}")

    def unfreeze_ca_and_lora(self):
        """
        Re-enables gradient flow through CA blocks and LoRA layers.
        Mainly useful for warm-up resumption after a checkpoint reload.
        """
        # CA blocks are always fully trainable
        for param in self.ca_blocks.parameters():
            param.requires_grad = True

        # LoRA layers: only the lora_A/lora_B matrices, not the frozen base weights
        for name, param in self.lm.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        self._frozen = False

    def is_frozen(self) -> bool:
        return self._frozen

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def print_trainable_parameters(self):
        self.lm.print_trainable_parameters()
        ca_params = sum(p.numel() for p in self.ca_blocks.parameters())
        print(f"  ca_blocks (all layers): {ca_params:,} params")

    @staticmethod
    def stack_latents(latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Converts the Student's output (List of M tensors [batch, d]) into
        the [batch, M, d] tensor the Verbalizer expects.

        Call this before passing latents into any Verbalizer method.
        """
        return torch.stack(latents, dim=1)  # [batch, M, d]
===
"""
verbalizer.py
-------------
Verbalizer (Vψ) for ThinkFlow-VLA Stage 2.

Base model: Qwen3.5-0.8B (hybrid Gated DeltaNet + Attention architecture)

Architecture
------------
At EVERY transformer layer, a new CrossAttentionBlock is inserted:
    h_l  = OriginalTransformerLayer_l(h_{l-1})          ← untouched SA/DeltaNet + FFN
    h_l  = CrossAttentionBlock_l(Q=h_l, K=z, V=z)      ← new CA reads Student latents

z = stack of Student's M=6 latent vectors, shape [batch, M, d_student].
Since d_student (2560) ≠ d_verbalizer (1024), each CA block has its own K/V projection.

Qwen3.5-0.8B architecture:
  - 24 transformer layers (3:1 Gated DeltaNet : standard Attention)
  - Hidden dim: 1024
  - Vocab: 248,320

Model hierarchy (after PEFT wrapping):
  self.lm                              → PeftModel
  self.lm.model                        → Qwen3_5ForCausalLM
  self.lm.model.model                  → Qwen3_5TextModel (.embed_tokens, .layers, .norm)
  self.lm.model.lm_head                → Linear

Because DeltaNet layers use recurrent state (not KV cache), the manual
layer-by-layer forward does NOT use KV caching.

Training schedule (controlled externally by train_stage2.py):
  Steps 0 – 3000  : warm-up  — CA blocks + LoRA trainable, LM loss on τ+
  Steps 3000 – 4500: frozen  — all Vψ params frozen, DPO gradient flows into Student

Loss functions implemented here:
  compute_lm_loss    : cross-entropy on τ+ tokens     (warm-up phase)
  compute_dpo_loss   : DPO preference loss on τ+/τ−   (both phases, but only updates
                       Vψ params during warm-up; Student params always)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# Cross-Attention Block (one per Verbalizer transformer layer)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    A single cross-attention block that reads M Student latent vectors.

    Q  = hidden states from the current Verbalizer layer  [batch, seq, d_verb]
    K,V = Student latents z projected to d_verb            [batch, M,   d_verb]

    Output replaces the query sequence via residual + pre-norm:
        h = LayerNorm(h + MultiheadAttn(Q=h, K=k, V=v))
    """

    def __init__(
        self,
        query_dim: int,   # Verbalizer hidden size  (d_verb = 1024)
        kv_dim: int,      # Student hidden size     (d_student = 2560)
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert query_dim % num_heads == 0, (
            f"query_dim {query_dim} must be divisible by num_heads {num_heads}"
        )

        # Project Student latents into Verbalizer's space for K and V
        self.k_proj = nn.Linear(kv_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, query_dim, bias=False)

        # Standard multi-head cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pre-norm on queries (applied before attention, following modern practice)
        self.q_norm = nn.LayerNorm(query_dim, eps=1e-6)

        # Post-norm on residual output
        self.out_norm = nn.LayerNorm(query_dim, eps=1e-6)

    def forward(
        self,
        hidden: torch.Tensor,    # [batch, seq, d_verb]
        latents: torch.Tensor,   # [batch, M, d_student]
    ) -> torch.Tensor:
        # Project latents → K and V in verbalizer space
        k = self.k_proj(latents)  # [batch, M, d_verb]
        v = self.v_proj(latents)  # [batch, M, d_verb]

        # Normalize queries before attention
        q = self.q_norm(hidden)   # [batch, seq, d_verb]

        # Cross-attention: every verbalizer token attends to all M latents
        ca_out, _ = self.attn(q, k, v)  # [batch, seq, d_verb]

        # Residual connection + post-norm
        return self.out_norm(hidden + ca_out)  # [batch, seq, d_verb]


# ---------------------------------------------------------------------------
# Verbalizer
# ---------------------------------------------------------------------------

class Verbalizer(nn.Module):
    """
    Qwen3.5-0.8B with per-layer cross-attention blocks conditioned on Student latents.

    Parameters
    ----------
    model_name     : HuggingFace repo ID for the 0.8B base model
    student_hidden : hidden size of the Student VLM (d_student = 2560)
    lora_rank      : LoRA rank for the base attention layers
    lora_alpha     : LoRA scaling
    ca_num_heads   : attention heads inside each CrossAttentionBlock
    ca_dropout     : dropout in cross-attention (0 during distillation)
    dpo_beta       : β temperature for DPO loss
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        student_hidden: int = 2560,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        ca_num_heads: int = 8,
        ca_dropout: float = 0.0,
        dpo_beta: float = 0.1,
    ):
        super().__init__()
        self.dpo_beta = dpo_beta

        # ------------------------------------------------------------------
        # 1. Load Qwen3.5-0.8B
        # ------------------------------------------------------------------
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        # ------------------------------------------------------------------
        # 2. Wrap with LoRA
        # ------------------------------------------------------------------
        # Target modules span both layer types in the hybrid architecture:
        #   - Standard Attention: q_proj, k_proj, v_proj, o_proj
        #   - Gated DeltaNet:     out_proj, in_proj_qkv, in_proj_z
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                # Standard attention projections
                "q_proj", "k_proj", "v_proj", "o_proj",
                # Gated DeltaNet main projections
                "out_proj", "in_proj_qkv", "in_proj_z",
            ],
            bias="none",
        )
        self.lm = get_peft_model(base, lora_cfg)

        # Infer verbalizer hidden dim and layer count from config
        self.hidden_dim: int = self.lm.config.hidden_size        # 1024 for 0.8B
        self.num_layers: int = self.lm.config.num_hidden_layers  # 24 for 0.8B

        # ------------------------------------------------------------------
        # 3. Insert one CrossAttentionBlock per transformer layer
        # ------------------------------------------------------------------
        self.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(
                query_dim=self.hidden_dim,
                kv_dim=student_hidden,
                num_heads=ca_num_heads,
                dropout=ca_dropout,
            )
            for _ in range(self.num_layers)
        ])

        # Freeze tracking
        self._frozen: bool = False

    # -----------------------------------------------------------------------
    # Internal: access helpers for the Qwen3.5 model hierarchy
    # -----------------------------------------------------------------------

    @property
    def _transformer(self):
        """
        Returns the Qwen3_5TextModel (the actual transformer stack).

        Hierarchy after PEFT wrapping:
          self.lm (PeftModel) → .model (Qwen3_5ForCausalLM)
          → .model (Qwen3_5TextModel: .embed_tokens, .layers, .norm)
        """
        return self.lm.model.model

    @property
    def _lm_head(self):
        """Returns the LM head (Linear layer for logit projection)."""
        return self.lm.model.lm_head

    # -----------------------------------------------------------------------
    # Internal: manual layer-by-layer forward with CA injection
    # -----------------------------------------------------------------------

    def _forward_with_latents(
        self,
        input_ids: torch.Tensor,        # [batch, seq]
        attention_mask: torch.Tensor,    # [batch, seq]
        latents: torch.Tensor,           # [batch, M, d_student]  — stacked z_1…z_M
        labels: Optional[torch.Tensor] = None,  # [batch, seq] for LM loss
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Manual transformer forward that injects CA after every layer.

        Because Qwen3.5 uses a hybrid architecture (Gated DeltaNet + standard
        Attention), we iterate through the layers and handle both types.
        Neither type uses KV caching in this forward path.

        Returns
        -------
        logits : [batch, seq, vocab_size]
        loss   : scalar CE loss if labels provided, else None
        """
        transformer = self._transformer

        # --- Embedding layer ---
        hidden = transformer.embed_tokens(input_ids)  # [batch, seq, d_verb]

        # --- Build causal mask and position IDs ---
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(
            seq_len, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        cache_position = torch.arange(seq_len, device=device)

        # Build the 4D causal mask the layers expect
        causal_mask = transformer._update_causal_mask(
            attention_mask,
            hidden,
            cache_position,
            past_key_values=None,
            output_attentions=False,
        )

        # --- Layer-by-layer forward with CA injection ---
        for layer_idx, layer in enumerate(transformer.layers):
            # Standard transformer layer (DeltaNet or Attention + FFN)
            # Both layer types accept the same interface
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
            )
            hidden = layer_out[0]  # [batch, seq, d_verb]

            # Cross-attention: read Student latents
            # Gradient flows through latents back to Student when not detached
            hidden = self.ca_blocks[layer_idx](hidden, latents)

        # --- Final norm ---
        hidden = transformer.norm(hidden)  # [batch, seq, d_verb]

        # --- LM head ---
        logits = self._lm_head(hidden)  # [batch, seq, vocab_size]

        # --- Optional LM loss ---
        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def _compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,       # [batch, seq]
        attention_mask: torch.Tensor,   # [batch, seq]
        latents: torch.Tensor,          # [batch, M, d_student]
        response_mask: torch.Tensor,    # [batch, seq] — 1 on response tokens only
    ) -> torch.Tensor:
        """
        Compute per-sequence sum of log-probabilities over response tokens.

        Used by compute_dpo_loss to get log π_ψ(τ | z).

        Returns
        -------
        log_probs : [batch]  — sum of token log-probs over response positions
        """
        logits, _ = self._forward_with_latents(input_ids, attention_mask, latents)

        # log-softmax over vocab
        log_probs_all = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]

        # Shift to align: logits at position i predict token i+1
        # Predicted log-prob of token i+1 is log_probs_all[:, i, token_{i+1}]
        shift_log_probs = log_probs_all[:, :-1, :]         # [batch, seq-1, vocab]
        shift_labels    = input_ids[:, 1:]                  # [batch, seq-1]
        shift_mask      = response_mask[:, 1:]              # [batch, seq-1]

        # Gather the log-prob of each ground-truth token
        token_log_probs = shift_log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)  # [batch, seq-1]

        # Sum over response positions only
        seq_log_probs = (token_log_probs * shift_mask).sum(dim=-1)  # [batch]

        return seq_log_probs

    # -----------------------------------------------------------------------
    # Public loss functions
    # -----------------------------------------------------------------------

    def compute_lm_loss(
        self,
        input_ids: torch.Tensor,       # [batch, seq]  — τ+ sequence
        attention_mask: torch.Tensor,
        latents: torch.Tensor,         # [batch, M, d_student]  — Student z (DETACHED during warm-up)
        labels: torch.Tensor,          # [batch, seq]  — τ+ with -100 on prefix positions
    ) -> torch.Tensor:
        """
        Warm-up loss: standard cross-entropy on τ+ tokens.
        Trains the CA blocks and LoRA to learn to read Student latents.

        NOTE: during warm-up, pass latents.detach() so gradients do NOT
        flow back into the Student yet. The Student is updated by L_distill
        and L_ans only during warm-up.

        Returns
        -------
        lm_loss : scalar
        """
        _, loss = self._forward_with_latents(
            input_ids, attention_mask, latents, labels=labels
        )
        return loss

    def compute_dpo_loss(
        self,
        pos_input_ids: torch.Tensor,   # [batch, seq] — τ+ tokenized
        neg_input_ids: torch.Tensor,    # [batch, seq] — τ− tokenized
        pos_attention_mask: torch.Tensor,
        neg_attention_mask: torch.Tensor,
        latents: torch.Tensor,          # [batch, M, d_student] — Student z (NO detach here)
        pos_response_mask: torch.Tensor,   # [batch, seq] — 1 on τ+ response tokens
        neg_response_mask: torch.Tensor,   # [batch, seq] — 1 on τ− response tokens
        ref_pos_log_probs: Optional[torch.Tensor] = None,  # [batch] reference model log-probs
        ref_neg_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        DPO preference loss.

        After Verbalizer is frozen (step > 3000), all gradients from this loss
        flow through the latents tensor back into the Student's parameters.

        DPO objective:
            L_DPO = -E[ log σ( β * (log π(τ+|z) − log π_ref(τ+|z))
                                 − β * (log π(τ−|z) − log π_ref(τ−|z)) ) ]

        If reference log-probs not provided (common simplification), reduces to:
            L_DPO = -E[ log σ( β * (log π(τ+|z) − log π(τ−|z)) ) ]

        Parameters
        ----------
        ref_pos_log_probs / ref_neg_log_probs:
            Pass pre-computed reference model log-probs if using a reference
            policy (e.g., initial Verbalizer checkpoint). Pass None to use
            the simplified reference-free variant.

        Returns
        -------
        dpo_loss : scalar
        metrics  : dict with reward margin and accuracy for logging
        """
        # Log-probs under current Verbalizer policy
        # latents NOT detached → gradient flows into Student when Vψ is frozen
        log_pi_pos = self._compute_sequence_log_probs(
            pos_input_ids, pos_attention_mask, latents, pos_response_mask
        )  # [batch]
        log_pi_neg = self._compute_sequence_log_probs(
            neg_input_ids, neg_attention_mask, latents, neg_response_mask
        )  # [batch]

        # DPO reward margins
        if ref_pos_log_probs is not None and ref_neg_log_probs is not None:
            # Full DPO: subtract reference log-probs
            pi_log_ratios_pos = log_pi_pos - ref_pos_log_probs
            pi_log_ratios_neg = log_pi_neg - ref_neg_log_probs
        else:
            # Simplified (reference-free) DPO
            pi_log_ratios_pos = log_pi_pos
            pi_log_ratios_neg = log_pi_neg

        reward_margin = self.dpo_beta * (pi_log_ratios_pos - pi_log_ratios_neg)

        # DPO loss: -log sigmoid(reward_margin)
        dpo_loss = -F.logsigmoid(reward_margin).mean()

        # Logging metrics
        with torch.no_grad():
            metrics = {
                "dpo_loss":       dpo_loss.item(),
                "reward_margin":  reward_margin.mean().item(),
                "dpo_accuracy":   (reward_margin > 0).float().mean().item(),
                "log_pi_pos":     log_pi_pos.mean().item(),
                "log_pi_neg":     log_pi_neg.mean().item(),
            }

        return dpo_loss, metrics

    # -----------------------------------------------------------------------
    # Warm-up → freeze transition
    # -----------------------------------------------------------------------

    def freeze_for_student_training(self):
        """
        Called at step 3000.

        Freezes ALL Verbalizer parameters (base model + LoRA + CA blocks).
        After this, DPO gradients flow through the latents tensor only,
        updating the Student's weights — not the Verbalizer's.
        """
        if self._frozen:
            return

        for param in self.parameters():
            param.requires_grad = False

        self._frozen = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Verbalizer] Frozen. Trainable params remaining: {trainable:,}")

    def unfreeze_ca_and_lora(self):
        """
        Re-enables gradient flow through CA blocks and LoRA layers.
        Mainly useful for warm-up resumption after a checkpoint reload.
        """
        # CA blocks are always fully trainable
        for param in self.ca_blocks.parameters():
            param.requires_grad = True

        # LoRA layers: only the lora_A/lora_B matrices, not the frozen base weights
        for name, param in self.lm.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        self._frozen = False

    def is_frozen(self) -> bool:
        return self._frozen

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def print_trainable_parameters(self):
        self.lm.print_trainable_parameters()
        ca_params = sum(p.numel() for p in self.ca_blocks.parameters())
        print(f"  ca_blocks (all layers): {ca_params:,} params")

    @staticmethod
    def stack_latents(latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Converts the Student's output (List of M tensors [batch, d]) into
        the [batch, M, d] tensor the Verbalizer expects.

        Call this before passing latents into any Verbalizer method.
        """
        return torch.stack(latents, dim=1)  # [batch, M, d]
        return torch.stack(latents, dim=1)  # [batch, M, d]
```

**Key changes:**

| Aspect | Before (Qwen3-0.6B) | After (Qwen3.5-0.8B) |
|---|---|---|
| Default model | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3.5-0.8B` |
| Hidden dim | inferred (was ~768) | **1024** |
| Layers | inferred | **24** |
| Student hidden | 2048 | **2560** |
| LoRA targets | `q/k/v/o_proj` only | + **`out_proj`, `in_proj_qkv`, `in_proj_z`** (DeltaNet) |
| Model hierarchy | `self.lm.model` (ambiguous) | `self.lm.model.model` → Qwen3_5TextModel |
| LM head access | `self.lm.lm_head` | `self.lm.model.lm_head` |
| `_update_causal_mask` | Assumed Qwen3 API | Same (Qwen3.5 uses same interface) |

### 3. `spatial_forcing.py` — Spatial Forcing Loss

```diff:spatial_forcing.py
"""
spatial_forcing.py
------------------
Spatial Forcing auxiliary loss for ThinkFlow-VLA Stage 2.

Loss:
    L_spatial = -CosSim( ProjectionMLP(pool(x_V)), pool(Extractor(I)) )

Where:
    x_V        — mid-layer (L/2) visual token hidden states from the Student
                 already extracted via LatentStudent.get_mid_layer_visual_features()
                 shape: [batch, num_visual_tokens, d_student=2048]
    Extractor  — frozen VGGT or DINOv2 feature extractor; zero inference overhead
                 at deployment because it is training-only
    pool(·)    — mean pool over spatial/patch dimension → [batch, d]
    MLP        — trainable projection on Student side: [d_student → d_extractor]

Design:
    - Extractor is ALWAYS frozen (no_grad); only MLP trains
    - Spatial resolution mismatch between Student visual tokens and extractor
      patches is handled by mean pooling both sides before loss computation
    - Two extractor backends supported:
        "dinov2"  → facebook/dinov2-large  (d=1024) or dinov2-base (d=768)
        "vggt"    → configurable checkpoint (user-specified)
    - Backend is pluggable via the FrozenExtractor base class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional

from transformers import AutoModel


# ---------------------------------------------------------------------------
# Frozen extractor base + concrete implementations
# ---------------------------------------------------------------------------

class FrozenExtractor(ABC, nn.Module):
    """
    Base class for frozen spatial feature extractors.
    All parameters are frozen at construction time.
    output_dim must be set by subclasses.
    """
    output_dim: int

    def __init__(self):
        super().__init__()

    def _freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    @abstractmethod
    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from raw pixel values.

        Parameters
        ----------
        pixel_values : [batch, C, H, W]  — normalised image tensor

        Returns
        -------
        features : [batch, d_extractor]
            Mean-pooled over spatial/patch dimension.
            This happens inside each extractor so the calling code is uniform.
        """
        ...

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.extract(pixel_values)


class DINOv2Extractor(FrozenExtractor):
    """
    Frozen DINOv2 feature extractor.

    Supported checkpoints:
        "facebook/dinov2-large"  → output_dim = 1024
        "facebook/dinov2-base"   → output_dim = 768
        "facebook/dinov2-small"  → output_dim = 384
    """

    _DIM_MAP = {
        "facebook/dinov2-large":  1024,
        "facebook/dinov2-base":   768,
        "facebook/dinov2-small":  384,
        "facebook/dinov2-giant":  1536,
    }

    def __init__(self, checkpoint: str = "facebook/dinov2-large"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        )
        self.output_dim = self._DIM_MAP.get(checkpoint, 1024)
        self._freeze_all()

    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        DINOv2 outputs last_hidden_state: [batch, num_patches+1, d]
        Index 0 is the [CLS] token; patches start at index 1.
        We mean-pool patch tokens only.
        """
        out = self.model(pixel_values=pixel_values, return_dict=True)
        patch_features = out.last_hidden_state[:, 1:, :]  # [batch, num_patches, d]
        return patch_features.mean(dim=1)                 # [batch, d]


class VGGTExtractor(FrozenExtractor):
    """
    Frozen VGGT feature extractor.

    The exact checkpoint path will be confirmed by the user; this class accepts
    any AutoModel-compatible repo. Pass output_dim explicitly since VGGT's
    hidden size varies by variant.

    Typical usage:
        VGGTExtractor(checkpoint="<user-confirmed-repo>", output_dim=1024)
    """

    def __init__(self, checkpoint: str, output_dim: int = 1024):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.output_dim = output_dim
        self._freeze_all()

    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        VGGT's exact output format depends on its architecture.
        This implementation assumes last_hidden_state is available and
        mean-pools it. Update the indexing once the checkpoint is confirmed.
        """
        out = self.model(pixel_values=pixel_values, return_dict=True)

        if hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state  # [batch, seq, d]
            # If first token is CLS-like, skip it; otherwise pool everything
            if feats.shape[1] > 1:
                feats = feats[:, 1:, :]
            return feats.mean(dim=1)       # [batch, d]

        raise AttributeError(
            "VGGTExtractor: model output has no 'last_hidden_state'. "
            "Inspect the output keys and update extract() accordingly "
            "once the checkpoint is confirmed."
        )


# ---------------------------------------------------------------------------
# Trainable projection MLP (Student side only)
# ---------------------------------------------------------------------------

class ProjectionMLP(nn.Module):
    """
    Projects mean-pooled Student mid-layer visual features into the extractor's
    feature space so cosine similarity is well-defined.

    Input:  [batch, d_student]   (mean-pooled x_V)
    Output: [batch, d_extractor] (L2-normalised for CosSim stability)

    Three-layer MLP with GELU activations and a final L2-norm.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        mid_dim = (in_dim + out_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=False),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)                         # [batch, d_extractor]
        return F.normalize(projected, dim=-1)           # unit-norm for CosSim


# ---------------------------------------------------------------------------
# Spatial Forcing Loss
# ---------------------------------------------------------------------------

class SpatialForcingLoss(nn.Module):
    """
    Full Spatial Forcing module: frozen extractor + trainable ProjectionMLP.

    Only ProjectionMLP trains; extractor is always inference-only.

    Parameters
    ----------
    extractor_type  : "dinov2" or "vggt"
    extractor_ckpt  : HuggingFace repo ID for the chosen extractor
    student_dim     : Student hidden size (d_student = 2048)
    extractor_dim   : Output dim of the extractor (pass explicitly for VGGT)
    lambda_sf       : Loss scale λ (default 0.1 per the FYP spec)
    """

    def __init__(
        self,
        extractor_type: str = "dinov2",
        extractor_ckpt: str = "facebook/dinov2-large",
        student_dim: int = 2048,
        extractor_dim: Optional[int] = None,
        lambda_sf: float = 0.1,
    ):
        super().__init__()
        self.lambda_sf = lambda_sf

        # ------------------------------------------------------------------
        # 1. Frozen extractor
        # ------------------------------------------------------------------
        if extractor_type == "dinov2":
            self.extractor = DINOv2Extractor(checkpoint=extractor_ckpt)
        elif extractor_type == "vggt":
            if extractor_dim is None:
                raise ValueError(
                    "extractor_dim must be provided explicitly for VGGT "
                    "until the checkpoint is confirmed."
                )
            self.extractor = VGGTExtractor(
                checkpoint=extractor_ckpt, output_dim=extractor_dim
            )
        else:
            raise ValueError(
                f"Unknown extractor_type '{extractor_type}'. "
                "Choose 'dinov2' or 'vggt'."
            )

        d_ext = self.extractor.output_dim

        # ------------------------------------------------------------------
        # 2. Trainable projection MLP (Student side only)
        # ------------------------------------------------------------------
        self.proj_mlp = ProjectionMLP(in_dim=student_dim, out_dim=d_ext)

    # -----------------------------------------------------------------------
    # Feature extraction helpers (called separately for clarity in the loop)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def extract_reference_features(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the frozen extractor and L2-normalise.
        Called ONCE per batch before the Student forward pass.
        Cache the result and reuse — no need to re-extract within one step.

        Parameters
        ----------
        pixel_values : [batch, C, H, W]  — preprocessed for the extractor's
                       expected normalisation (ImageNet stats for DINOv2)

        Returns
        -------
        ref_feats : [batch, d_extractor]  unit-norm
        """
        feats = self.extractor(pixel_values)           # [batch, d_ext]
        return F.normalize(feats.float(), dim=-1)      # unit-norm, fp32 for precision

    # -----------------------------------------------------------------------
    # Main loss computation
    # -----------------------------------------------------------------------

    def compute_loss(
        self,
        x_V: torch.Tensor,             # [batch, num_visual_tokens, d_student]
        ref_feats: torch.Tensor,        # [batch, d_extractor]  — from extract_reference_features()
    ) -> torch.Tensor:
        """
        Compute the Spatial Forcing loss:
            L_spatial = -mean( CosSim( ProjectionMLP(pool(x_V)), ref_feats ) )

        Parameters
        ----------
        x_V       : mid-layer visual features from LatentStudent.get_mid_layer_visual_features()
        ref_feats : pre-extracted extractor features (unit-norm)

        Returns
        -------
        loss : scalar (already scaled by lambda_sf)
        """
        # Pool Student visual tokens: [batch, num_visual_tokens, d] → [batch, d]
        pooled_student = x_V.mean(dim=1)                      # [batch, d_student]

        # Project into extractor space and unit-norm
        projected = self.proj_mlp(pooled_student.float())     # [batch, d_ext], unit-norm

        # Cosine similarity per sample (both sides are unit-norm, so dot = cosim)
        cos_sim = (projected * ref_feats).sum(dim=-1)         # [batch]

        # Negative cosine similarity (maximising alignment = minimising negative cosim)
        loss = -cos_sim.mean()

        return self.lambda_sf * loss

    def forward(
        self,
        x_V: torch.Tensor,
        pixel_values_for_extractor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience wrapper: extract reference features + compute loss in one call.
        Use compute_loss() directly if you've pre-cached ref_feats for efficiency.

        Returns
        -------
        loss : scalar (scaled by lambda_sf)
        """
        ref_feats = self.extract_reference_features(pixel_values_for_extractor)
        return self.compute_loss(x_V, ref_feats)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def print_trainable_parameters(self):
        extractor_params = sum(p.numel() for p in self.extractor.parameters())
        mlp_params       = sum(p.numel() for p in self.proj_mlp.parameters())
        print(f"  extractor (frozen):  {extractor_params:,} params")
        print(f"  projection_mlp:      {mlp_params:,} params  [TRAINABLE]")
        print(f"  lambda_sf:           {self.lambda_sf}")
===
"""
spatial_forcing.py
------------------
Spatial Forcing auxiliary loss for ThinkFlow-VLA Stage 2.

Loss:
    L_spatial = -CosSim( ProjectionMLP(pool(x_V)), pool(Extractor(I)) )

Where:
    x_V        — mid-layer (L/2) visual token hidden states from the Student
                 already extracted via LatentStudent.get_mid_layer_visual_features()
                 shape: [batch, num_visual_tokens, d_student=2560]
    Extractor  — frozen VGGT or DINOv2 feature extractor; zero inference overhead
                 at deployment because it is training-only
    pool(·)    — mean pool over spatial/patch dimension → [batch, d]
    MLP        — trainable projection on Student side: [d_student → d_extractor]

Design:
    - Extractor is ALWAYS frozen (no_grad); only MLP trains
    - Spatial resolution mismatch between Student visual tokens and extractor
      patches is handled by mean pooling both sides before loss computation
    - Two extractor backends supported:
        "dinov2"  → facebook/dinov2-large  (d=1024) or dinov2-base (d=768)
        "vggt"    → configurable checkpoint (user-specified)
    - Backend is pluggable via the FrozenExtractor base class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional

from transformers import AutoModel


# ---------------------------------------------------------------------------
# Frozen extractor base + concrete implementations
# ---------------------------------------------------------------------------

class FrozenExtractor(ABC, nn.Module):
    """
    Base class for frozen spatial feature extractors.
    All parameters are frozen at construction time.
    output_dim must be set by subclasses.
    """
    output_dim: int

    def __init__(self):
        super().__init__()

    def _freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    @abstractmethod
    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from raw pixel values.

        Parameters
        ----------
        pixel_values : [batch, C, H, W]  — normalised image tensor

        Returns
        -------
        features : [batch, d_extractor]
            Mean-pooled over spatial/patch dimension.
            This happens inside each extractor so the calling code is uniform.
        """
        ...

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.extract(pixel_values)


class DINOv2Extractor(FrozenExtractor):
    """
    Frozen DINOv2 feature extractor.

    Supported checkpoints:
        "facebook/dinov2-large"  → output_dim = 1024
        "facebook/dinov2-base"   → output_dim = 768
        "facebook/dinov2-small"  → output_dim = 384
    """

    _DIM_MAP = {
        "facebook/dinov2-large":  1024,
        "facebook/dinov2-base":   768,
        "facebook/dinov2-small":  384,
        "facebook/dinov2-giant":  1536,
    }

    def __init__(self, checkpoint: str = "facebook/dinov2-large"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        )
        self.output_dim = self._DIM_MAP.get(checkpoint, 1024)
        self._freeze_all()

    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        DINOv2 outputs last_hidden_state: [batch, num_patches+1, d]
        Index 0 is the [CLS] token; patches start at index 1.
        We mean-pool patch tokens only.
        """
        out = self.model(pixel_values=pixel_values, return_dict=True)
        patch_features = out.last_hidden_state[:, 1:, :]  # [batch, num_patches, d]
        return patch_features.mean(dim=1)                 # [batch, d]


class VGGTExtractor(FrozenExtractor):
    """
    Frozen VGGT feature extractor.

    The exact checkpoint path will be confirmed by the user; this class accepts
    any AutoModel-compatible repo. Pass output_dim explicitly since VGGT's
    hidden size varies by variant.

    Typical usage:
        VGGTExtractor(checkpoint="<user-confirmed-repo>", output_dim=1024)
    """

    def __init__(self, checkpoint: str, output_dim: int = 1024):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.output_dim = output_dim
        self._freeze_all()

    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        VGGT's exact output format depends on its architecture.
        This implementation assumes last_hidden_state is available and
        mean-pools it. Update the indexing once the checkpoint is confirmed.
        """
        out = self.model(pixel_values=pixel_values, return_dict=True)

        if hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state  # [batch, seq, d]
            # If first token is CLS-like, skip it; otherwise pool everything
            if feats.shape[1] > 1:
                feats = feats[:, 1:, :]
            return feats.mean(dim=1)       # [batch, d]

        raise AttributeError(
            "VGGTExtractor: model output has no 'last_hidden_state'. "
            "Inspect the output keys and update extract() accordingly "
            "once the checkpoint is confirmed."
        )


# ---------------------------------------------------------------------------
# Trainable projection MLP (Student side only)
# ---------------------------------------------------------------------------

class ProjectionMLP(nn.Module):
    """
    Projects mean-pooled Student mid-layer visual features into the extractor's
    feature space so cosine similarity is well-defined.

    Input:  [batch, d_student]   (mean-pooled x_V)
    Output: [batch, d_extractor] (L2-normalised for CosSim stability)

    Three-layer MLP with GELU activations and a final L2-norm.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        mid_dim = (in_dim + out_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=False),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)                         # [batch, d_extractor]
        return F.normalize(projected, dim=-1)           # unit-norm for CosSim


# ---------------------------------------------------------------------------
# Spatial Forcing Loss
# ---------------------------------------------------------------------------

class SpatialForcingLoss(nn.Module):
    """
    Full Spatial Forcing module: frozen extractor + trainable ProjectionMLP.

    Only ProjectionMLP trains; extractor is always inference-only.

    Parameters
    ----------
    extractor_type  : "dinov2" or "vggt"
    extractor_ckpt  : HuggingFace repo ID for the chosen extractor
    student_dim     : Student hidden size (d_student = 2560 for Qwen3.5-4B)
    extractor_dim   : Output dim of the extractor (pass explicitly for VGGT)
    lambda_sf       : Loss scale λ (default 0.1 per the FYP spec)
    """

    def __init__(
        self,
        extractor_type: str = "dinov2",
        extractor_ckpt: str = "facebook/dinov2-large",
        student_dim: int = 2560,
        extractor_dim: Optional[int] = None,
        lambda_sf: float = 0.1,
    ):
        super().__init__()
        self.lambda_sf = lambda_sf

        # ------------------------------------------------------------------
        # 1. Frozen extractor
        # ------------------------------------------------------------------
        if extractor_type == "dinov2":
            self.extractor = DINOv2Extractor(checkpoint=extractor_ckpt)
        elif extractor_type == "vggt":
            if extractor_dim is None:
                raise ValueError(
                    "extractor_dim must be provided explicitly for VGGT "
                    "until the checkpoint is confirmed."
                )
            self.extractor = VGGTExtractor(
                checkpoint=extractor_ckpt, output_dim=extractor_dim
            )
        else:
            raise ValueError(
                f"Unknown extractor_type '{extractor_type}'. "
                "Choose 'dinov2' or 'vggt'."
            )

        d_ext = self.extractor.output_dim

        # ------------------------------------------------------------------
        # 2. Trainable projection MLP (Student side only)
        # ------------------------------------------------------------------
        self.proj_mlp = ProjectionMLP(in_dim=student_dim, out_dim=d_ext)

    # -----------------------------------------------------------------------
    # Feature extraction helpers (called separately for clarity in the loop)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def extract_reference_features(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the frozen extractor and L2-normalise.
        Called ONCE per batch before the Student forward pass.
        Cache the result and reuse — no need to re-extract within one step.

        Parameters
        ----------
        pixel_values : [batch, C, H, W]  — preprocessed for the extractor's
                       expected normalisation (ImageNet stats for DINOv2)

        Returns
        -------
        ref_feats : [batch, d_extractor]  unit-norm
        """
        feats = self.extractor(pixel_values)           # [batch, d_ext]
        return F.normalize(feats.float(), dim=-1)      # unit-norm, fp32 for precision

    # -----------------------------------------------------------------------
    # Main loss computation
    # -----------------------------------------------------------------------

    def compute_loss(
        self,
        x_V: torch.Tensor,             # [batch, num_visual_tokens, d_student]
        ref_feats: torch.Tensor,        # [batch, d_extractor]  — from extract_reference_features()
    ) -> torch.Tensor:
        """
        Compute the Spatial Forcing loss:
            L_spatial = -mean( CosSim( ProjectionMLP(pool(x_V)), ref_feats ) )

        Parameters
        ----------
        x_V       : mid-layer visual features from LatentStudent.get_mid_layer_visual_features()
        ref_feats : pre-extracted extractor features (unit-norm)

        Returns
        -------
        loss : scalar (already scaled by lambda_sf)
        """
        # Pool Student visual tokens: [batch, num_visual_tokens, d] → [batch, d]
        pooled_student = x_V.mean(dim=1)                      # [batch, d_student]

        # Project into extractor space and unit-norm
        projected = self.proj_mlp(pooled_student.float())     # [batch, d_ext], unit-norm

        # Cosine similarity per sample (both sides are unit-norm, so dot = cosim)
        cos_sim = (projected * ref_feats).sum(dim=-1)         # [batch]

        # Negative cosine similarity (maximising alignment = minimising negative cosim)
        loss = -cos_sim.mean()

        return self.lambda_sf * loss

    def forward(
        self,
        x_V: torch.Tensor,
        pixel_values_for_extractor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience wrapper: extract reference features + compute loss in one call.
        Use compute_loss() directly if you've pre-cached ref_feats for efficiency.

        Returns
        -------
        loss : scalar (scaled by lambda_sf)
        """
        ref_feats = self.extract_reference_features(pixel_values_for_extractor)
        return self.compute_loss(x_V, ref_feats)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def print_trainable_parameters(self):
        extractor_params = sum(p.numel() for p in self.extractor.parameters())
        mlp_params       = sum(p.numel() for p in self.proj_mlp.parameters())
        print(f"  extractor (frozen):  {extractor_params:,} params")
        print(f"  projection_mlp:      {mlp_params:,} params  [TRAINABLE]")
        print(f"  lambda_sf:           {self.lambda_sf}")
```

Only `student_dim` default changed: `2048 → 2560`.

---

## Files That Still Reference Old Models

These files were **not modified** but contain stale `Qwen2.5-VL-4B` / `2048` references:

| File | Lines | What needs updating |
|---|---|---|
| [train_stage2.py](file:///home/ubuntu/Shree_FYP/train/stage2/training/train_stage2.py) | L61 | `base_model_name` default |
| [grpo_teacher.py](file:///home/ubuntu/Shree_FYP/train/stage2/training/grpo_teacher.py) | L100, L116 | Model name, docstring |
| [tokenizer_setup.py](file:///home/ubuntu/Shree_FYP/train/stage2/tokenizer_setup.py) | L36, L61, L142 | Model name defaults |
| [test_gpu_smoke.py](file:///home/ubuntu/Shree_FYP/train/stage2/test/test_gpu_smoke.py) | L55, L123, L131, L256, L317 | Model name + `d_student` |
| [test_models_cpu.py](file:///home/ubuntu/Shree_FYP/train/stage2/test/test_models_cpu.py) | L6, L91 | Docstring references |
