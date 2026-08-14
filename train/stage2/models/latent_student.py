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

Full token sequence generated during training:
    [prompt (ending with <think>)] → z_1 → z_2 → ... → z_M → </think> → s_1 → ... → s_K

    Where:
        <think>   is the LAST TOKEN of input_ids (managed by the processor /
                  prompt template upstream — the Student receives the prompt
                  already terminated with <think>).
        z_1..z_M  are M=6 continuous hidden-state vectors (not token IDs).
        </think>  is always appended manually via embed_tokens — we do not
                  rely on the model autoregressively predicting it.
        s_1..s_K  are K=5 learnable spatial token parameters.

    There is no <answer> tag anywhere in the Student's sequence.

Hidden state extraction:
    h_S  = last_hidden_state at </think> position  → L_distill = ||h_T - h_S||²
    The </think> position is index -(K+1) from the end of the final sequence:
        ... | </think> | s_1 | s_2 | ... | s_K
               -(K+1)    -K              -1

    For K=5:  </think> is at index -6.

    This gives h_S full visibility of all M latent tokens via the concat
    sequence before the final forward pass — the correct informational
    counterpart to the Teacher's h_T (which sees all CoT text tokens).

Latent generation strategy — CONCAT-BASED (training-safe):
    Why not KV-cache:
        75% of Qwen3.5 layers are GatedDeltaNet — a recurrent layer that
        maintains a fixed-size state matrix S ∈ R^{d_k × d_v} per head.
        Differentiating through HuggingFace's hybrid recurrent cache during
        a multi-step training loop is unreliable (in-place state updates may
        silently break gradients). Concat-based sequence growth uses the
        DeltaNet layers' native training mode (LinearAttentionChunk), which
        processes the full sequence in one chunk-parallel pass per step.

    The loop (steps 1..M):
        current_embeds grows: [prefix | seed | z₁ | ... | z_{m-1}]
        full forward (use_cache=False) at each step
        z_m = last_hidden_state[:, -1, :]  ← last position in growing seq

    Final forward (single pass after loop):
        Input:  [prefix | z₁ | ... | z_M | </think>_embed | s₁ | ... | s_K]
        h_S     = last_hidden_state[:, -(K+1), :]   ← at </think>
        spatial = last_hidden_state[:, -K:, :]       ← at s_1..s_K

Utility method:
    get_mid_layer_visual_features  — for L_spatial / Spatial Forcing (Stage 4)

Note on get_answer_hidden_state:
    REMOVED. h_S is now returned directly by generate_latents() as the second
    element of its 4-tuple return value. Callers should update accordingly:
        OLD: latents, sp_h, wp = student.generate_latents(...)
             h_S = student.get_answer_hidden_state(...)
        NEW: latents, h_S, sp_h, wp = student.generate_latents(...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from transformers import AutoModelForImageTextToText
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import hf_hub_download
import os


# ---------------------------------------------------------------------------
# Architecture constants for Qwen3.5-4B
# ---------------------------------------------------------------------------

QWEN35_4B_HIDDEN_DIM  = 2560   # text_config.hidden_size
QWEN35_4B_NUM_LAYERS  = 32     # text_config.num_hidden_layers
QWEN35_4B_MID_LAYER   = 16     # L/2 used for Spatial Forcing features

# All LoRA target modules: shared projections + DeltaNet-specific + FFN
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

        # Initializing weight to exactly 0.0 blocks gradient flow backwards (grad_x = W^T * grad_y = 0).
        # We use a small normal distribution instead to keep outputs near 0.0 (preventing
        # sigmoid saturation) while allowing gradients to flow to lower layers.
        nn.init.normal_(self.net[4].weight, std=0.01)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, K, d]  →  [batch, K, 2]
        return self.net(x)


# ---------------------------------------------------------------------------
# Latent Student
# ---------------------------------------------------------------------------

class LatentStudent(nn.Module):
    """
    Qwen3.5-4B wrapped with LoRA acting as the Latent Student (Fθ).

    The Student diverges from the Teacher (initialised from the same Stage 1
    checkpoint) through its generation mechanism: instead of producing discrete
    text reasoning tokens it generates M=6 continuous latent vectors z_1..z_M,
    then closes with a </think> token, followed by K=5 learnable spatial tokens
    that decode to 2D waypoints. There is no <answer> tag in the Student's
    sequence.

    Parameters
    ----------
    model_name         : HuggingFace repo ID   (default: shreethar/stage1_unsloth)
    M                  : number of continuous reasoning latents  (default: 6)
    K                  : number of learnable spatial tokens       (default: 5)
    lora_rank          : LoRA rank r                              (default: 64)
    lora_alpha         : LoRA scaling α                           (default: 128)
    lora_dropout       : dropout inside LoRA layers               (default: 0.05)
    new_vocab_size     : pass len(tokenizer) after any token registration; -1 = skip
    end_think_token_id : token ID for </think> — required by generate_latents.
                         Can also be supplied per-call to generate_latents().
    """

    def __init__(
        self,
        model_name: str = "shreethar/stage1_unsloth",
        M: int = 6,
        K: int = 5,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        new_vocab_size: int = -1,
        end_think_token_id: Optional[int] = None,
        use_lora: bool = True,
    ):
        super().__init__()
        self.M = M
        self.K = K
        self.end_think_token_id = end_think_token_id

        # ------------------------------------------------------------------
        # 1. Load Qwen3.5-4B with flash_attention_2
        #
        #    AutoModelForImageTextToText is the correct class for Qwen3.5
        #    which uses early-fusion multimodal tokens (not Qwen2.5-VL's
        #    separate visual encoder class).
        #
        #    The vision encoder is NOT frozen here. Unlike the Teacher (which
        #    freezes its vision encoder in grpo_teacher.py), the Student
        #    trains its vision encoder via LoRA.
        # ------------------------------------------------------------------
        base = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        # ------------------------------------------------------------------
        # 2. Wrap with LoRA or keep base model
        # ------------------------------------------------------------------
        if use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=QWEN35_LORA_TARGETS,
                bias="none",
            )
            self.vlm = get_peft_model(base, lora_cfg)

            # Explicitly enable gradient checkpointing for the Student
            self.vlm.enable_input_require_grads()
            self.vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            self.vlm = base
            
        self.vlm.config.use_cache = False

        if new_vocab_size > 0:
            self.vlm.resize_token_embeddings(new_vocab_size)

        # ------------------------------------------------------------------
        # 3. Read architecture constants from config
        # ------------------------------------------------------------------
        text_cfg = getattr(
            base.model.config, "text_config", base.model.config
        )
        self.hidden_dim: int    = getattr(text_cfg, "hidden_size",      QWEN35_4B_HIDDEN_DIM)
        self.num_layers: int    = getattr(text_cfg, "num_hidden_layers", QWEN35_4B_NUM_LAYERS)
        self.mid_layer_idx: int = self.num_layers // 2    # 16 for 32-layer 4B model

        self._image_token_id: Optional[int] = getattr(
            base.model.config, "image_token_id", None
        )

        # ------------------------------------------------------------------
        # 4. K=5 learnable spatial tokens  [K, hidden_dim]
        # ------------------------------------------------------------------
        self.spatial_tokens = nn.Parameter(
            torch.randn(K, self.hidden_dim) * 0.02
        )
        # Semantic Initialization: Give the spatial tokens a massive "head start" 
        # by initializing them with the embeddings of actual, common tokens 
        # (rather than complete noise), so the self-attention layers already 
        # know how to interact with them.
        with torch.no_grad():
            try:
                embed_weight = base.get_input_embeddings().weight
                self.spatial_tokens.data.copy_(embed_weight[1000 : 1000 + K].clone())
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 5. SpatialMLP : spatial hidden states → normalised 2D waypoints
        # ------------------------------------------------------------------
        self.spatial_mlp = SpatialMLP(self.hidden_dim)

        self.spatial_tokens.data = self.spatial_tokens.data.to(torch.bfloat16)
        self.spatial_mlp.to(torch.bfloat16)

    # -----------------------------------------------------------------------
    # Internal property shortcuts
    # -----------------------------------------------------------------------

    @property
    def _language_model(self) -> nn.Module:
        """
        The transformer stack inside the Qwen3.5 model.
        """
        base_model = self.vlm.get_base_model() if hasattr(self.vlm, "get_base_model") else self.vlm
        return base_model.model.language_model

    @property
    def _embed_tokens(self) -> nn.Embedding:
        """Token embedding table — used to embed </think> during generate_latents."""
        return self._language_model.embed_tokens

    @property
    def _visual_encoder(self) -> nn.Module:
        """Vision encoder — trains via LoRA (NOT frozen for Student)."""
        base_model = self.vlm.get_base_model() if hasattr(self.vlm, "get_base_model") else self.vlm
        return base_model.model.visual

    # -----------------------------------------------------------------------
    # Input embedding construction
    # -----------------------------------------------------------------------

    def _build_input_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
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
        embeds = self._embed_tokens(input_ids)   # [batch, seq, d]

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
                    if not isinstance(img_feats, torch.Tensor):
                        img_feats = getattr(img_feats, 'pooler_output', img_feats[0])
                    embeds[mask] = img_feats.to(embeds.dtype)

        if pixel_values_videos is not None:
            if video_grid_thw is not None:
                vid_feats = self._visual_encoder(pixel_values_videos, grid_thw=video_grid_thw)
            else:
                vid_feats = self._visual_encoder(pixel_values_videos)
            
            self._video_token_id = getattr(self.vlm.config, "video_token_id", 248057)
            mask_vid = (input_ids == self._video_token_id)
            if mask_vid.any():
                embeds = embeds.clone()
                if not isinstance(vid_feats, torch.Tensor):
                    vid_feats = getattr(vid_feats, 'pooler_output', vid_feats[0])
                embeds[mask_vid] = vid_feats.to(embeds.dtype)

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
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process the full prompt through the language model.

        The prompt template (managed by the processor upstream) ends with the
        <think> token as its final token. Therefore:
            seed_hidden = h at <think> position
        This is the natural anchor for the latent loop — the model has fully
        processed the image and instruction and is positioned at the start of
        its reasoning block.

        Returns
        -------
        prefix_embeds : [batch, prompt_len, hidden_dim]
            Raw embedding tensor for the prompt (including visual tokens).
            Carried through the latent loop as the growing sequence base.

        seed_hidden : [batch, hidden_dim]
            Hidden state at the last prompt token = <think> position.
            This becomes current_token for step m=1 of the latent loop.

        Design note:
            Returns raw embeddings (not past_key_values) because concat-based
            growth re-processes the full growing sequence at each step. This is
            the correct training path for GatedDeltaNet layers, which use
            chunk-parallel computation over full sequences during training.
        """
        prefix_embeds = self._build_input_embeds(
            input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw
        )   # [batch, prompt_len, d]

        out = self._language_model(
            inputs_embeds=prefix_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # seed = h at <think> (the last token of the prompt)
        seed_hidden = out.last_hidden_state[:, -1, :]   # [batch, d]

        return prefix_embeds, seed_hidden

    # -----------------------------------------------------------------------
    # Core: latent generation + h_S extraction + spatial decoding
    # -----------------------------------------------------------------------

    def _generate_reasoning_context(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run the continuous autoregressive loop shared by Stage 2 and Stage 4.

        Returns the M latent states plus the growing embeddings/mask needed by
        :meth:`generate_latents` for spatial decoding.  Stage 4's frozen
        reference model uses :meth:`generate_reasoning_latents` to stop here,
        avoiding an unnecessary spatial-decoder forward pass.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        prefix_embeds, seed_hidden = self.encode_prefix(
            input_ids,
            pixel_values,
            image_grid_thw,
            attention_mask,
            pixel_values_videos,
            video_grid_thw,
        )

        current_embeds = prefix_embeds
        current_mask = attention_mask
        current_token = seed_hidden
        latents: List[torch.Tensor] = []

        for _ in range(self.M):
            current_embeds = torch.cat(
                [current_embeds, current_token.unsqueeze(1)], dim=1
            )
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(
                        batch_size,
                        1,
                        device=device,
                        dtype=current_mask.dtype,
                    ),
                ],
                dim=1,
            )

            out = self._language_model(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            current_token = out.last_hidden_state[:, -1, :]
            latents.append(current_token)

        return latents, current_embeds, current_mask

    def generate_reasoning_latents(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Generate only z_1..z_M, without the spatial decoding pass."""
        latents, _, _ = self._generate_reasoning_context(
            input_ids,
            pixel_values,
            image_grid_thw,
            attention_mask,
            pixel_values_videos,
            video_grid_thw,
        )
        return latents

    def generate_latents(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        end_think_token_id: Optional[int] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate M=6 continuous latent vectors via concat-based autoregressive
        loop, extract h_S at </think>, then decode K=5 spatial tokens into
        2D waypoints.

        Full sequence built inside this method:
            [prefix (ends <think>)] → z_1 → ... → z_M → </think> → s_1 → ... → s_K

        Latent loop (training-safe concat strategy):
            At each step m, the growing sequence is:
                [prefix_embeds | seed | z₁ | ... | z_{m-1}]
            A full forward pass (use_cache=False) is run over this sequence.
            The last position's hidden state becomes z_m and is fed as the
            next token in embedding space (no vocab lookup).
            This triggers LinearAttentionChunk in GatedDeltaNet layers
            (native training mode), ensuring correct gradient flow through
            all M steps back to the Student's LoRA weights.

        Final forward pass (single pass after loop):
            </think> is always appended manually via embed_tokens — we do not
            rely on the model predicting it. All K spatial tokens are appended
            in the same pass.

            Tail: [</think> | s_1 | ... | s_K]  →  K+1 tokens from the end.
            Position index for h_S:
                last_hidden_state[:, -(K+1), :]
            Verification (K=5):  positions from end →
                s_5=-1, s_4=-2, s_3=-3, s_2=-4, s_1=-5, </think>=-6
                -(K+1) = -(5+1) = -6  ✓

        Parameters
        ----------
        input_ids, pixel_values, image_grid_thw, attention_mask :
            Standard batch inputs. input_ids must end with the <think> token
            (responsibility of the prompt template / processor upstream).
        end_think_token_id : int, optional
            Token ID for </think>. Overrides self.end_think_token_id if given.

        Returns
        -------
        latents        : List of M tensors, each [batch, hidden_dim]
                         z_1, ..., z_M — the continuous reasoning latents.
                         Raw final-layer hidden states; no projection or vocab lookup.
        h_S            : [batch, hidden_dim]
                         Hidden state at the </think> token position.
                         Used by student_losses.py: L_distill = MSE(h_S, h_T.detach())
        spatial_hidden : [batch, K, hidden_dim]
                         Output hidden states of the K spatial tokens.
        waypoints      : [batch, K, 2]
                         Predicted 2D waypoints in normalised [0, 1] space.
        """
        _end_think_id = (
            end_think_token_id
            if end_think_token_id is not None
            else self.end_think_token_id
        )
        assert _end_think_id is not None, (
            "end_think_token_id must be set in __init__ or passed to generate_latents()"
        )

        batch_size = input_ids.shape[0]
        device = input_ids.device

        latents, current_embeds, current_mask = self._generate_reasoning_context(
            input_ids,
            pixel_values,
            image_grid_thw,
            attention_mask,
            pixel_values_videos,
            video_grid_thw,
        )

        # ------------------------------------------------------------------
        # Final forward: append </think> | s_1 ... s_K  (one combined pass)
        #
        # </think> is embedded as a discrete token — always appended manually.
        # K spatial tokens follow immediately after.
        # Single pass extracts both h_S and spatial_hidden — no re-encoding.
        # ------------------------------------------------------------------

        # Embed </think>
        end_think_embed = self._embed_tokens(
            torch.full((batch_size, 1), _end_think_id, device=device, dtype=torch.long)
        ).to(dtype=current_embeds.dtype)   # [B, 1, d]

        # K learnable spatial token embeddings
        spatial_embeds = (
            self.spatial_tokens            # [K, d]
            .unsqueeze(0)                  # [1, K, d]
            .expand(batch_size, -1, -1)    # [B, K, d]
            .to(dtype=current_embeds.dtype)
        )

        # Tail: [</think> | s_1 | ... | s_K]  →  [B, K+1, d]
        tail_embeds = torch.cat([end_think_embed, spatial_embeds], dim=1)
        tail_mask   = torch.ones(
            batch_size, self.K + 1, device=device, dtype=current_mask.dtype
        )

        full_embeds = torch.cat([current_embeds, tail_embeds], dim=1)
        full_mask   = torch.cat([current_mask,   tail_mask],   dim=1)

        final_out = self._language_model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        final_hs = final_out.last_hidden_state   # [B, total_len, d]

        # h_S at </think>:  offset -(K+1) from end
        #   Positions from end: s_K=-1, ..., s_1=-K, </think>=-(K+1)
        h_S = final_hs[:, -(self.K + 1), :]    # [B, d]

        # spatial_hidden: last K positions  (the s_1..s_K outputs)
        spatial_hidden = final_hs[:, -self.K:, :]   # [B, K, d]
        waypoints      = self.spatial_mlp(spatial_hidden)   # [B, K, 2]

        return latents, h_S, spatial_hidden, waypoints

    # -----------------------------------------------------------------------
    # Mid-layer visual features  (used for L_spatial / Spatial Forcing — Stage 4)
    # -----------------------------------------------------------------------

    def get_mid_layer_visual_features(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        return_mask: bool = False,
        return_kv: bool = False,
    ):
        """
        Full forward pass with output_hidden_states=True.
        Returns hidden states at a selected transformer layer at visual-token
        positions.  ``layer_idx`` is zero-based; ``None`` selects L/2 = 16.

        These are x_V — used in Stage 4:
            L_spatial = 1 - CosSim( ProjectionMLP(x_V), VGGT(I) )

        With ``return_kv=True``, this also captures the selected layer's
        projected key/value activations at visual-token positions. Qwen3.5's
        linear-attention layers use a combined ``in_proj_qkv``; in that case
        the query slice is discarded and the contiguous K/V slices are
        returned. Full-attention layers concatenate ``k_proj`` and ``v_proj``.
        This is an activation diagnostic only, not a KV cache.

        Layer indexing:
            hidden_states tuple index 0 = embedding layer output
            hidden_states tuple index i+1 = output of transformer layer i

        Returns
        -------
        x_V : [batch, max_visual_tokens, hidden_dim]
              Zero-padded when samples contain different visual-token counts.
        visual_mask : [batch, max_visual_tokens], optional
              Returned when ``return_mask=True``; True marks real tokens.
        kv_V : [batch, max_visual_tokens, key_dim + value_dim], optional
              Returned when ``return_kv=True``.
        """
        selected_layer = self.mid_layer_idx if layer_idx is None else layer_idx
        if not 0 <= selected_layer < self.num_layers:
            raise ValueError(
                f"layer_idx must be in [0, {self.num_layers - 1}], got {selected_layer}"
            )

        embeds = self._build_input_embeds(
            input_ids,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
        )

        captured_kv = {}
        hook_handles = []
        if return_kv:
            layer = self._language_model.layers[selected_layer]
            if hasattr(layer, "linear_attn"):
                token_mixer = layer.linear_attn
                key_dim = int(getattr(token_mixer, "key_dim"))

                def capture_linear_kv(_module, _inputs, output):
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("in_proj_qkv hook expected a tensor output")
                    # Q and K both have key_dim; V occupies the remainder.
                    captured_kv["kv"] = output[..., key_dim:]

                hook_handles.append(
                    token_mixer.in_proj_qkv.register_forward_hook(capture_linear_kv)
                )
            elif hasattr(layer, "self_attn"):
                token_mixer = layer.self_attn

                def capture_key(_module, _inputs, output):
                    captured_kv["key"] = output

                def capture_value(_module, _inputs, output):
                    captured_kv["value"] = output

                hook_handles.extend(
                    [
                        token_mixer.k_proj.register_forward_hook(capture_key),
                        token_mixer.v_proj.register_forward_hook(capture_value),
                    ]
                )
            else:
                raise TypeError(
                    f"Layer {selected_layer} exposes neither linear_attn nor self_attn"
                )

        try:
            out = self._language_model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            for handle in hook_handles:
                handle.remove()

        kv_hidden = None
        if return_kv:
            if "kv" in captured_kv:
                kv_hidden = captured_kv["kv"]
            elif "key" in captured_kv and "value" in captured_kv:
                kv_hidden = torch.cat(
                    [captured_kv["key"], captured_kv["value"]], dim=-1
                )
            else:
                raise RuntimeError(
                    f"Layer {selected_layer} K/V projection hooks captured no output"
                )

        # hidden_states[0] = embeddings; hidden_states[i+1] = layer i output
        mid_hidden = out.hidden_states[selected_layer + 1]   # [B, seq, d]

        if self._image_token_id is None and not hasattr(self, "_video_token_id"):
            visual_mask = attention_mask.to(torch.bool)
            if return_kv and return_mask:
                return mid_hidden, visual_mask, kv_hidden
            if return_kv:
                return mid_hidden, kv_hidden
            return (mid_hidden, visual_mask) if return_mask else mid_hidden

        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self._image_token_id is not None:
            mask = mask | (input_ids == self._image_token_id)
        if hasattr(self, "_video_token_id"):
            mask = mask | (input_ids == self._video_token_id)

        counts = mask.sum(dim=1)
        max_visual = max(1, int(counts.max().item()))
        x_V = mid_hidden.new_zeros(
            input_ids.shape[0], max_visual, self.hidden_dim
        )
        kv_V = None
        if return_kv:
            kv_V = kv_hidden.new_zeros(
                input_ids.shape[0], max_visual, kv_hidden.shape[-1]
            )
        visual_mask = torch.zeros(
            input_ids.shape[0],
            max_visual,
            device=mid_hidden.device,
            dtype=torch.bool,
        )

        for batch_idx, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            x_V[batch_idx, :count] = mid_hidden[batch_idx, mask[batch_idx]]
            if kv_V is not None:
                kv_V[batch_idx, :count] = kv_hidden[batch_idx, mask[batch_idx]]
            visual_mask[batch_idx, :count] = True

        if return_kv and return_mask:
            return x_V, visual_mask, kv_V
        if return_kv:
            return x_V, kv_V
        return (x_V, visual_mask) if return_mask else x_V

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
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ):
        """
        Standard causal LM forward — delegates to the full vlm.
        Used for Stage 1 SFT evaluation and any non-latent inference.
        """
        return self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def print_trainable_parameters(self):
        """Print LoRA trainable parameter counts and architecture info."""
        if hasattr(self.vlm, "print_trainable_parameters"):
            self.vlm.print_trainable_parameters()
        sp_tok = self.spatial_tokens.numel()
        sp_mlp = sum(p.numel() for p in self.spatial_mlp.parameters())
        print(f"  spatial_tokens  : {sp_tok:,} params  [TRAINABLE]")
        print(f"  spatial_mlp     : {sp_mlp:,} params  [TRAINABLE]")
        print(f"  hidden_dim      : {self.hidden_dim}")
        print(f"  num_layers      : {self.num_layers}  (mid_layer={self.mid_layer_idx})")
        print(f"  M (latents)     : {self.M}")
        print(f"  K (spatial)     : {self.K}")
        print(f"  end_think_id    : {self.end_think_token_id}")

    # -----------------------------------------------------------------------
    # HuggingFace Hub Integration
    # -----------------------------------------------------------------------
    
    @classmethod
    def from_pretrained(
        cls, 
        repo_id: str, 
        end_think_token_id: int, 
        M: int = 6, 
        K: int = 5,
        **kwargs
    ):
        """
        Load a fully merged LatentStudent model directly from HuggingFace Hub.
        Downloads the merged VLM base weights and the custom spatial parameters.
        """
        # 1. Initialize without LoRA (base model will be loaded directly from repo_id)
        model = cls(
            model_name=repo_id,
            M=M,
            K=K,
            use_lora=False,
            end_think_token_id=end_think_token_id,
            **kwargs
        )
        
        # 2. Download and load custom spatial parameters
        try:
            custom_weights_path = hf_hub_download(repo_id=repo_id, filename="spatial_parameters.pt")
            state = torch.load(custom_weights_path, map_location="cpu")
            model.spatial_tokens.data.copy_(state["spatial_tokens"])
            model.spatial_mlp.load_state_dict(state["spatial_mlp"])
            print(f"Successfully loaded spatial parameters from {repo_id}")
        except Exception as e:
            print(f"Warning: Could not load spatial_parameters.pt from {repo_id}. Ensure it was uploaded. Error: {e}")
            
        return model
