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