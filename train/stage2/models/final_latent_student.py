"""
latent_student.py
-----------------
Latent Student (Fθ) for ThinkFlow-VLA Stage 2.

Backbone : Qwen/Qwen3.5-4B  (AutoModelForImageTextToText)
LoRA     : rank=64, alpha=128, all projection types including DeltaNet-specific
           q_proj, k_proj, v_proj, o_proj          — shared attention projections
           out_proj                                  — attention output
           in_proj_qkv, in_proj_z, in_proj_b, in_proj_a — GatedDeltaNet-specific
           gate_proj, up_proj, down_proj             — FFN / MoE

Model hierarchy (Qwen3.5):
    vlm = AutoModelForImageTextToText(...)
    vlm.model                           — Qwen3_5Model
    vlm.model.visual                    — vision encoder (multimodal early-fusion)
    vlm.model.language_model            — transformer stack
    vlm.model.language_model.embed_tokens
    vlm.model.language_model.layers     — 32 layers (24×DeltaNet + 8×GatedAttention)
    vlm.model.config.text_config.hidden_size  — 2560

Vision encoder:
    NOT frozen for the Student — the entire model including the vision encoder
    trains via LoRA. (Teacher's vision encoder is frozen separately in grpo_teacher.py.)

Latent generation strategy — CONCAT-BASED (training-safe):
    Why not KV-cache:
        75% of Qwen3.5 layers are GatedDeltaNet — a recurrent layer that
        maintains a fixed-size state matrix S ∈ R^{d_k × d_v} per head.
        Differentiating through HuggingFace's hybrid recurrent cache during
        a multi-step training loop is unreliable (in-place state updates may
        silently break gradients). Concat-based sequence growth uses the
        DeltaNet layers' native training mode (LinearAttentionChunk), which
        processes the full sequence in one chunk-parallel pass per step.

    The loop:
        Step 0 : full forward over [prefix_embeds]
                 → seed = last_hidden_state[:, -1, :]

        Step m (m=1..M):
            current_embeds = [prefix_embeds | seed | z₁ | ... | z_{m-1}]
            full forward (use_cache=False)
            z_m = last_hidden_state[:, -1, :]   ← last position in growing seq
            z_m fed as next token in embedding space (no vocab lookup)

        After M steps, append K=5 spatial tokens to the grown sequence,
        run one more full forward, extract last K hidden states → SpatialMLP → waypoints.

    Cost: each step re-processes a sequence growing by 1 token.
          For M=6 and typical prompt length ~512, this is negligible vs
          the prefix encoding cost.

Utility methods:
    get_answer_hidden_state        — for L_distill (h_S at <ans> position)
    get_mid_layer_visual_features  — for L_spatial (x_V at layer L/2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from transformers import AutoModelForImageTextToText
from peft import LoraConfig, TaskType, get_peft_model


# ---------------------------------------------------------------------------
# Architecture constants for Qwen3.5-4B
# ---------------------------------------------------------------------------

QWEN35_4B_HIDDEN_DIM  = 2560   # text_config.hidden_size
QWEN35_4B_NUM_LAYERS  = 32     # text_config.num_hidden_layers
QWEN35_4B_MID_LAYER   = 16     # L/2 used for Spatial Forcing features

# All LoRA target modules: shared projections + DeltaNet-specific projections + FFN
QWEN35_LORA_TARGETS = [
    # Standard attention projections (GatedAttention layers — 25%)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # Attention output projection
    "out_proj",
    # GatedDeltaNet-specific projections (DeltaNet layers — 75%)
    "in_proj_qkv",   # combined Q/K/V input projection
    "in_proj_z",     # gate projection
    "in_proj_b",     # beta scalar (forgetting gate)
    "in_proj_a",     # alpha projection
    # FFN / MoE projections (shared across all layer types)
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# SpatialMLP : [batch, K, d] → [batch, K, 2]
# ---------------------------------------------------------------------------

class SpatialMLP(nn.Module):
    """
    Projects K spatial token hidden states into 2D waypoints.

    Architecture: 3-layer MLP with GELU activations.
    Final Sigmoid constrains output to [0, 1] — matching Stage 1's
    normalised coordinate convention.

    Input:  [batch, K, hidden_dim]
    Output: [batch, K, 2]   (x, y in [0, 1])
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, K, d]  →  [batch, K, 2]
        return self.net(x)


# ---------------------------------------------------------------------------
# Latent Student
# ---------------------------------------------------------------------------

class LatentStudent(nn.Module):
    """
    Qwen3.5-4B wrapped with LoRA acting as the Latent Student (Fθ).

    The Student diverges from the Teacher (which is initialised from the same
    Stage 1 checkpoint) through its training objective: instead of generating
    discrete text tokens it produces M=6 continuous latent vectors z₁…z_M
    and K=5 spatial waypoints via a learnable SpatialMLP.

    Parameters
    ----------
    model_name   : HuggingFace repo ID   (default: Qwen/Qwen3.5-4B)
    M            : number of continuous reasoning latents  (default: 6)
    K            : number of learnable spatial tokens       (default: 5)
    lora_rank    : LoRA rank r                              (default: 64)
    lora_alpha   : LoRA scaling α                           (default: 128)
    lora_dropout : dropout inside LoRA layers               (default: 0.05)
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
        # 1. Load Qwen3.5-4B with flash_attention_2
        #
        #    AutoModelForImageTextToText is the correct class for Qwen3.5
        #    which uses early-fusion multimodal tokens (not Qwen2.5-VL's
        #    separate visual encoder class).
        #
        #    trust_remote_code=True required for the qwen3_5 architecture
        #    until it is merged into transformers main.
        #
        #    NOTE: The vision encoder is NOT frozen here. Unlike the Teacher
        #    (which freezes its vision encoder in grpo_teacher.py), the Student
        #    trains its vision encoder via LoRA to allow visual feature
        #    adaptation alongside language LoRA.
        # ------------------------------------------------------------------
        base = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            trust_remote_code=True,
        )

        # ------------------------------------------------------------------
        # 2. Wrap with LoRA — all projection types targeted
        #
        #    Includes both standard attention projections and
        #    GatedDeltaNet-specific projections (in_proj_*).
        #    This covers 100% of the hybrid architecture's trainable weights.
        # ------------------------------------------------------------------
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=QWEN35_LORA_TARGETS,
            bias="none",
        )
        self.vlm = get_peft_model(base, lora_cfg)

        # ------------------------------------------------------------------
        # 3. Read architecture constants from config
        #
        #    Qwen3.5 stores language model config under text_config.
        #    Fall back to known 4B defaults if the attribute is missing
        #    (e.g. when running with mocked config in tests).
        # ------------------------------------------------------------------
        text_cfg = getattr(self.vlm.model.config, "text_config", self.vlm.model.config)

        self.hidden_dim: int    = getattr(text_cfg, "hidden_size",       QWEN35_4B_HIDDEN_DIM)
        self.num_layers: int    = getattr(text_cfg, "num_hidden_layers",  QWEN35_4B_NUM_LAYERS)
        self.mid_layer_idx: int = self.num_layers // 2    # 16 for 32-layer 4B model

        # Image/visual token id — used to locate visual token positions in x_V
        self._image_token_id: Optional[int] = getattr(
            self.vlm.model.config, "image_token_id", None
        )

        # ------------------------------------------------------------------
        # 4. K=5 learnable spatial tokens  [K, hidden_dim]
        #
        #    Randomly initialised (std=0.02, matching typical embedding init).
        #    Processed in parallel after the latent loop; their output hidden
        #    states are passed through SpatialMLP to produce 2D waypoints.
        # ------------------------------------------------------------------
        self.spatial_tokens = nn.Parameter(
            torch.randn(K, self.hidden_dim) * 0.02
        )

        # ------------------------------------------------------------------
        # 5. SpatialMLP : spatial hidden states → normalised 2D waypoints
        # ------------------------------------------------------------------
        self.spatial_mlp = SpatialMLP(self.hidden_dim)

    # -----------------------------------------------------------------------
    # Internal property shortcuts
    # -----------------------------------------------------------------------

    @property
    def _language_model(self) -> nn.Module:
        """
        The transformer stack inside the Qwen3.5 model.
        vlm.model.language_model contains embed_tokens and layers.
        """
        return self.vlm.model.language_model

    @property
    def _embed_tokens(self) -> nn.Embedding:
        """Token embedding table for building input embeddings."""
        return self._language_model.embed_tokens

    @property
    def _visual_encoder(self) -> nn.Module:
        """Vision encoder — trains via LoRA (NOT frozen for Student)."""
        return self.vlm.model.visual

    # -----------------------------------------------------------------------
    # Input embedding construction
    # -----------------------------------------------------------------------

    def _build_input_embeds(
        self,
        input_ids: torch.Tensor,                    # [batch, seq]
        pixel_values: Optional[torch.Tensor],        # [total_patches, C, H, W]  or None
        image_grid_thw: Optional[torch.Tensor],      # grid dimensions for packing
    ) -> torch.Tensor:
        """
        Convert input_ids to embeddings, then splice in visual encoder
        features at image token positions.

        Vision encoder runs with gradients (no no_grad block) because the
        Student trains its visual encoder. This differs from the Teacher
        which wraps visual encoding in no_grad.

        Returns
        -------
        inputs_embeds : [batch, seq, hidden_dim]
        """
        # Text token embeddings
        embeds = self._embed_tokens(input_ids)   # [batch, seq, d]

        # Visual token splicing
        if pixel_values is not None:
            if image_grid_thw is not None:
                img_feats = self._visual_encoder(pixel_values, grid_thw=image_grid_thw)
            else:
                img_feats = self._visual_encoder(pixel_values)
            # img_feats: [total_visual_tokens, d]

            if self._image_token_id is not None:
                mask = (input_ids == self._image_token_id)   # [batch, seq]
                if mask.any():
                    embeds = embeds.clone()
                    embeds[mask] = img_feats.to(embeds.dtype)

        return embeds   # [batch, seq, d]

    # -----------------------------------------------------------------------
    # Prefix encoding (step 0 of the latent loop)
    # -----------------------------------------------------------------------

    def encode_prefix(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process the full prompt (vision + instruction tokens) through the
        language model.

        Returns
        -------
        prefix_embeds : [batch, prompt_len, hidden_dim]
            The raw embedding tensor for the prompt.
            Carried through the latent loop as the growing sequence base —
            each step concatenates a new token onto this.

        seed_hidden : [batch, hidden_dim]
            Hidden state of the last prompt token (last_hidden_state[:, -1, :]).
            Becomes the first token fed into the latent loop (step m=1 input).

        Design note:
            Unlike the KV-cache approach, we return the raw embeddings (not
            past_key_values) because the concat strategy re-processes the
            full growing sequence at each latent step. This is the correct
            training path for GatedDeltaNet layers, which use chunk-parallel
            computation over full sequences during training.
        """
        prefix_embeds = self._build_input_embeds(
            input_ids, pixel_values, image_grid_thw
        )   # [batch, prompt_len, d]

        out = self._language_model(
            inputs_embeds=prefix_embeds,
            attention_mask=attention_mask,
            use_cache=False,          # no KV-cache for training
            output_hidden_states=False,
            return_dict=True,
        )

        # Seed for the latent loop = last prompt token's hidden state
        seed_hidden = out.last_hidden_state[:, -1, :]   # [batch, d]

        return prefix_embeds, seed_hidden

    # -----------------------------------------------------------------------
    # Core: concat-based latent generation + spatial decoding
    # -----------------------------------------------------------------------

    def generate_latents(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Generate M=6 continuous latent vectors via concat-based autoregressive
        loop, then decode K=5 spatial tokens into 2D waypoints.

        Concat strategy (training-safe for GatedDeltaNet):
        ---------------------------------------------------
        At each step m, the growing sequence is:
            [prefix_embeds | seed | z₁ | z₂ | ... | z_{m-1}]
        
        A single full forward pass (use_cache=False) is run over this sequence.
        The last position's hidden state becomes z_m.
        z_m is then appended to the sequence for step m+1.

        This triggers LinearAttentionChunk in GatedDeltaNet layers (the native
        training compute mode), ensuring correct gradient flow through all M
        steps back to the Student's LoRA weights.

        Parameters
        ----------
        input_ids, pixel_values, image_grid_thw, attention_mask:
            Standard batch inputs for the prompt.

        Returns
        -------
        latents        : List of M tensors, each [batch, hidden_dim]
                         z₁, z₂, ..., z_M — the continuous reasoning latents.
                         These are the raw final-layer hidden states — no
                         projection head, no vocabulary lookup.
        spatial_hidden : [batch, K, hidden_dim]
                         Output hidden states of the K spatial tokens.
                         Passed to SpatialMLP for waypoint prediction.
        waypoints      : [batch, K, 2]
                         Predicted 2D waypoints in normalised [0, 1] space.
        """
        batch_size = input_ids.shape[0]
        device     = input_ids.device

        # ------------------------------------------------------------------
        # Step 0: Encode prefix
        # prefix_embeds  [B, prompt_len, d] — reused as growing sequence base
        # seed_hidden    [B, d]             — input to first latent step
        # ------------------------------------------------------------------
        prefix_embeds, seed_hidden = self.encode_prefix(
            input_ids, pixel_values, image_grid_thw, attention_mask
        )

        # Growing sequence and mask — start with just the prefix
        current_embeds = prefix_embeds          # [B, prompt_len, d]
        current_mask   = attention_mask         # [B, prompt_len]

        # The seed becomes the first token appended to the sequence
        current_token = seed_hidden             # [B, d]

        latents: List[torch.Tensor] = []

        # ------------------------------------------------------------------
        # Steps 1..M: autoregressive latent loop
        # ------------------------------------------------------------------
        for m in range(self.M):
            # Append current_token as the newest position in the sequence
            current_embeds = torch.cat(
                [current_embeds, current_token.unsqueeze(1)], dim=1
            )   # [B, prompt_len + m + 1, d]

            # Extend attention mask by one position (always attend to new token)
            current_mask = torch.cat(
                [current_mask,
                 torch.ones(batch_size, 1, device=device, dtype=current_mask.dtype)],
                dim=1,
            )   # [B, prompt_len + m + 1]

            # Full forward pass over the growing sequence
            # use_cache=False forces LinearAttentionChunk in DeltaNet layers
            out = self._language_model(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )

            # z_m = hidden state at the last (newest) position
            # This IS the continuous latent — no projection, no vocab lookup
            z_m = out.last_hidden_state[:, -1, :]   # [B, d]
            latents.append(z_m)

            # Feed z_m as the next token in embedding space
            current_token = z_m   # [B, d]

        # ------------------------------------------------------------------
        # Spatial tokens: K=5 learnable parameters, processed in parallel
        #
        # Append all K spatial tokens at once to the fully-grown sequence
        # [prefix | seed | z₁ | ... | z_M | s₁ | ... | s_K]
        # Extract the last K hidden states → SpatialMLP → waypoints
        # ------------------------------------------------------------------
        spatial_embeds = (
            self.spatial_tokens                     # [K, d]
            .unsqueeze(0)                           # [1, K, d]
            .expand(batch_size, -1, -1)             # [B, K, d]
            .to(dtype=current_embeds.dtype)
        )

        # Extend mask for K new positions
        spatial_mask = torch.ones(
            batch_size, self.K, device=device, dtype=current_mask.dtype
        )

        spatial_input = torch.cat([current_embeds, spatial_embeds], dim=1)   # [B, seq+K, d]
        spatial_attn  = torch.cat([current_mask,   spatial_mask],   dim=1)   # [B, seq+K]

        sp_out = self._language_model(
            inputs_embeds=spatial_input,
            attention_mask=spatial_attn,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # The last K positions in the output correspond to the spatial tokens
        spatial_hidden = sp_out.last_hidden_state[:, -self.K:, :]   # [B, K, d]
        waypoints      = self.spatial_mlp(spatial_hidden)            # [B, K, 2]

        return latents, spatial_hidden, waypoints

    # -----------------------------------------------------------------------
    # <answer> token hidden state  (used for L_distill)
    # -----------------------------------------------------------------------

    def get_answer_hidden_state(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        answer_token_positions: torch.Tensor,   # [batch]  int64 — per-sample index
    ) -> torch.Tensor:
        """
        Standard full-sequence forward pass. Returns the hidden state at the
        <ans> token position for each item in the batch.

        This is h_S — the Student's structural equivalent of the Teacher's
        h_T. Used by student_losses.py to compute L_distill = MSE(h_S, h_T).

        The <ans> position is the last prompt token (prompt_len - 1), which
        marks the Student's transition from context to generation — mirroring
        the Teacher's <ans> token which marks the same transition to coordinate output.

        Returns
        -------
        h_S : [batch, hidden_dim]
        """
        embeds = self._build_input_embeds(input_ids, pixel_values, image_grid_thw)

        out = self._language_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        h_all     = out.last_hidden_state                              # [B, seq, d]
        batch_idx = torch.arange(h_all.shape[0], device=h_all.device)
        return h_all[batch_idx, answer_token_positions]                # [B, d]

    # -----------------------------------------------------------------------
    # Mid-layer visual features  (used for L_spatial / Spatial Forcing)
    # -----------------------------------------------------------------------

    def get_mid_layer_visual_features(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass with output_hidden_states=True.
        Returns hidden states at layer L/2 = 16 at visual token positions.

        These are x_V — used in:
            L_spatial = -CosSim( ProjectionMLP(x_V), VGGT(I) )

        Layer indexing:
            hidden_states tuple index 0  = embedding layer output
            hidden_states tuple index i  = output of transformer layer i-1
            → layer L/2 output lives at index (mid_layer_idx + 1) = 17

        Returns
        -------
        x_V : [batch, num_visual_tokens, hidden_dim]
              or [batch, 1, hidden_dim] if no visual tokens (text-only batch)
        """
        embeds = self._build_input_embeds(input_ids, pixel_values, image_grid_thw)

        out = self._language_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,   # need all layer outputs to index L/2
            return_dict=True,
        )

        # Index into the hidden_states tuple to get layer L/2 output
        # hidden_states[0] = embed output, hidden_states[k] = layer k-1 output
        mid_hidden = out.hidden_states[self.mid_layer_idx + 1]   # [B, seq, d]

        # Guard: if no image_token_id configured, return all token features
        if self._image_token_id is None:
            return mid_hidden

        # Locate visual token positions
        image_mask = (input_ids == self._image_token_id)   # [B, seq]
        n_visual   = image_mask[0].sum().item()

        # Guard: text-only batch (no visual tokens)
        if n_visual == 0:
            return torch.zeros(
                input_ids.shape[0], 1, self.hidden_dim,
                device=mid_hidden.device,
                dtype=mid_hidden.dtype,
            )

        # Gather visual token features: [total_visual_tokens, d] → [B, n_visual, d]
        x_V = mid_hidden[image_mask].view(
            input_ids.shape[0], n_visual, self.hidden_dim
        )
        return x_V

    # -----------------------------------------------------------------------
    # Standard forward (for SFT eval / non-latent use cases)
    # -----------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Standard causal LM forward pass — delegates to the full vlm.
        Used for Stage 1 SFT evaluation and any non-latent inference.
        """
        return self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            labels=labels,
        )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def print_trainable_parameters(self):
        """Print LoRA trainable parameter counts and architecture info."""
        self.vlm.print_trainable_parameters()
        sp_tok = self.spatial_tokens.numel()
        sp_mlp = sum(p.numel() for p in self.spatial_mlp.parameters())
        print(f"  spatial_tokens : {sp_tok:,} params  [TRAINABLE]")
        print(f"  spatial_mlp    : {sp_mlp:,} params  [TRAINABLE]")
        print(f"  hidden_dim     : {self.hidden_dim}")
        print(f"  num_layers     : {self.num_layers}  (mid_layer={self.mid_layer_idx})")
        print(f"  M (latents)    : {self.M}")
        print(f"  K (spatial)    : {self.K}")