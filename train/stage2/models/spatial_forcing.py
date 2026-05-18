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

    def __init__(self, checkpoint: str = "facebook/VGGT-1B", output_dim: int = 1024):
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
        extractor_type: str = "vggt",
        extractor_ckpt: str = "facebook/VGGT-1B",
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