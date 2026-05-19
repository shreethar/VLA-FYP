"""
spatial_forcing.py  [VGGT update]
-----------------------------------
Spatial Forcing auxiliary loss — ThinkFlow-VLA Stage 2.

Changes from previous version:
  1. VGGT: git-based install (not AutoModel).
     Import: from vggt.models.vggt import VGGT
     Load:   VGGT.from_pretrained("facebook/VGGT-1B")
     Install: see install_vggt.sh

  2. Token-level alignment (not mean-pooled).
     The Spatial Forcing paper aligns VLA visual tokens with VGGT
     patch-level spatial representations per-token, not globally pooled.
     This requires spatial resolution matching via interpolation.

  3. ProjectionMLP input: d_student=2560 (Qwen3.5-4B hidden dim)

Loss (token-level):
    x_V      : [batch, N_vis, d_student]  — mid-layer visual tokens
    vggt_feat: [batch, N_patches, d_vggt] — VGGT patch features (extracted + PE)
    
    align both to [batch, N, d] via interpolation, then:
    L_spatial = -mean_over_batch( mean_over_tokens( CosSim(MLP(x_V_i), vggt_i) ) )
    L_spatial *= lambda_sf

VGGT output format:
    predictions = model(images)   # images: [batch, num_views, C, H, W]
    The VGGT model outputs a dict of 3D attributes. For Spatial Forcing,
    we use the aggregated_tokens_list or similar intermediate features.
    See VGGTExtractor.extract() — update once confirmed on your setup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional


# ---------------------------------------------------------------------------
# ProjectionMLP (Student side — trainable)
# ---------------------------------------------------------------------------

class ProjectionMLP(nn.Module):
    """
    [batch, N, d_student] → [batch, N, d_ext]  (L2-normalised)
    Works on both token-level (N>1) and pooled (N=1) inputs.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        mid = (in_dim + out_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid, bias=False),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Linear(mid, out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, N, d] or [batch, d]
        projected = self.net(x)
        return F.normalize(projected.float(), dim=-1)


# ---------------------------------------------------------------------------
# Frozen extractor base
# ---------------------------------------------------------------------------

class FrozenExtractor(ABC, nn.Module):
    output_dim: int

    def __init__(self):
        super().__init__()

    def _freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    @abstractmethod
    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Returns patch-level features [batch, N_patches, d_ext].
        N_patches depends on image resolution.
        """
        ...

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.extract(pixel_values)


# ---------------------------------------------------------------------------
# DINOv2 (fallback extractor)
# ---------------------------------------------------------------------------

class DINOv2Extractor(FrozenExtractor):
    _DIM_MAP = {
        "facebook/dinov2-large": 1024,
        "facebook/dinov2-base":  768,
        "facebook/dinov2-small": 384,
        "facebook/dinov2-giant": 1536,
    }

    def __init__(self, checkpoint: str = "facebook/dinov2-large"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
        self.output_dim = self._DIM_MAP.get(checkpoint, 1024)
        self._freeze_all()

    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns patch tokens [batch, num_patches, d]  (skips CLS token)."""
        out = self.model(pixel_values=pixel_values, return_dict=True)
        return out.last_hidden_state[:, 1:, :]   # [batch, N_patches, d]


# ---------------------------------------------------------------------------
# VGGT extractor  (primary extractor — requires git install)
# ---------------------------------------------------------------------------

class VGGTExtractor(FrozenExtractor):
    """
    Frozen VGGT-1B spatial feature extractor.

    Install VGGT BEFORE running Stage 2 training:
        bash stage2/install_vggt.sh

    VGGT takes images shaped [batch, num_views, C, H, W].
    For single-view robot observations, unsqueeze dim 1:
        images = pixel_values.unsqueeze(1)   # [B, 1, C, H, W]

    VGGT outputs a dict of 3D attributes. We extract the aggregated
    context features which carry dense spatial information suitable
    for per-token alignment with VLA visual tokens.

    Output: [batch, N_patches, d_vggt]
    """

    # VGGT-1B outputs 1024-dim features
    VGGT_1B_DIM = 1024

    def __init__(
        self,
        checkpoint: str = "facebook/VGGT-1B",
        output_dim: int = VGGT_1B_DIM,
    ):
        super().__init__()
        try:
            from vggt.models.vggt import VGGT
        except ImportError:
            raise ImportError(
                "VGGT is not installed. Run: bash stage2/install_vggt.sh\n"
                "Then verify: python -c 'from vggt.models.vggt import VGGT'"
            )

        self.model = VGGT.from_pretrained(checkpoint)
        self.output_dim = output_dim
        self._freeze_all()

    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [batch, C, H, W] (single-view robot observation)
        
        Adds num_views=1 dimension, runs VGGT, extracts patch-level features.
        
        VGGT aggregated_tokens_list[-1] gives the final-layer context tokens
        which encode dense 3D spatial information at the patch level.
        
        Returns: [batch, N_patches, d_vggt]
        """
        # VGGT expects [batch, num_views, C, H, W]
        if pixel_values.dim() == 4:
            images = pixel_values.unsqueeze(1)   # [B, 1, C, H, W]
        else:
            images = pixel_values                # already [B, V, C, H, W]

        images = images.to(dtype=next(self.model.parameters()).dtype)

        with torch.no_grad():
            predictions = self.model(images)

        # Extract dense spatial features.
        # VGGT stores aggregated context tokens in aggregated_tokens_list.
        # The last entry is the richest (full depth of processing).
        if hasattr(predictions, "aggregated_tokens_list") and predictions.aggregated_tokens_list:
            feats = predictions.aggregated_tokens_list[-1]   # [B, V, N, d]
            # For single-view: squeeze view dim → [B, N, d]
            if feats.dim() == 4:
                feats = feats[:, 0, :, :]
            return feats   # [B, N_patches, d_vggt]

        # Fallback: if the above attribute name changes in a future VGGT version,
        # try common alternatives
        for attr in ["context_tokens", "image_features", "visual_features"]:
            if hasattr(predictions, attr):
                feats = getattr(predictions, attr)
                if feats.dim() == 4:
                    feats = feats[:, 0, :, :]
                return feats

        raise AttributeError(
            "VGGTExtractor: could not find patch-level features in VGGT output.\n"
            f"Available keys: {list(predictions.__dict__.keys()) if hasattr(predictions, '__dict__') else type(predictions)}\n"
            "Update VGGTExtractor.extract() to use the correct attribute."
        )


# ---------------------------------------------------------------------------
# Spatial Forcing Loss
# ---------------------------------------------------------------------------

class SpatialForcingLoss(nn.Module):
    """
    Token-level spatial alignment: align each VLA visual token with the
    spatially corresponding VGGT patch feature.

    L_spatial = -λ * mean_batch( mean_tokens( CosSim(MLP(x_V_i), vggt_i) ) )

    Token count mismatch between x_V (VLA visual tokens) and VGGT patches
    is handled by 1D interpolation along the spatial dimension.

    Parameters
    ----------
    extractor_type : "vggt" (default) or "dinov2" (fallback)
    extractor_ckpt : checkpoint for chosen extractor
    student_dim    : hidden size of Student VLM (2560 for Qwen3.5-4B)
    lambda_sf      : loss weight (0.1 per FYP spec)
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

        if extractor_type == "vggt":
            self.extractor = VGGTExtractor(
                checkpoint=extractor_ckpt,
                output_dim=extractor_dim or VGGTExtractor.VGGT_1B_DIM,
            )
        elif extractor_type == "dinov2":
            self.extractor = DINOv2Extractor(checkpoint=extractor_ckpt)
        else:
            raise ValueError(f"Unknown extractor_type '{extractor_type}'")

        self.proj_mlp = ProjectionMLP(
            in_dim=student_dim,
            out_dim=self.extractor.output_dim,
        )

    @torch.no_grad()
    def extract_reference_features(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Run frozen extractor. Returns token-level features [batch, N, d_ext] (L2-normed).
        Call ONCE before the Student forward pass and cache the result.
        """
        feats = self.extractor(pixel_values)          # [B, N, d]
        return F.normalize(feats.float(), dim=-1)     # unit-norm, fp32

    def compute_loss(
        self,
        x_V: torch.Tensor,       # [batch, N_vis, d_student]  — Student mid-layer features
        ref_feats: torch.Tensor,  # [batch, N_ref, d_ext]      — VGGT/DINOv2 patch features
    ) -> torch.Tensor:
        """
        Token-level cosine alignment loss.

        Handles N_vis ≠ N_ref via linear interpolation so the loss is
        resolution-invariant.

        Returns scalar (already scaled by lambda_sf).
        """
        B, N_vis, _ = x_V.shape
        N_ref = ref_feats.shape[1]

        # Project Student visual tokens → extractor space (unit-norm output)
        projected = self.proj_mlp(x_V.float())   # [B, N_vis, d_ext]

        # Align spatial dimensions if needed
        if N_vis != N_ref:
            # Interpolate projected to match ref spatial resolution
            # [B, N_vis, d] → [B, d, N_vis] → interp → [B, d, N_ref] → [B, N_ref, d]
            projected = F.interpolate(
                projected.permute(0, 2, 1),   # [B, d, N_vis]
                size=N_ref,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)               # [B, N_ref, d]
            # Re-normalise after interpolation
            projected = F.normalize(projected, dim=-1)

        # Per-token cosine similarity (both sides are unit-norm → dot product)
        cos_sim = (projected * ref_feats).sum(dim=-1)   # [B, N_ref]

        loss = -cos_sim.mean()
        return self.lambda_sf * loss

    def forward(self, x_V: torch.Tensor, pixel_values_for_extractor: torch.Tensor) -> torch.Tensor:
        """Convenience: extract + compute in one call. Prefer compute_loss() in the training loop."""
        ref_feats = self.extract_reference_features(pixel_values_for_extractor)
        return self.compute_loss(x_V, ref_feats)

    def print_trainable_parameters(self):
        ext   = sum(p.numel() for p in self.extractor.parameters())
        proj  = sum(p.numel() for p in self.proj_mlp.parameters())
        print(f"  extractor (frozen): {ext:,}")
        print(f"  proj_mlp:           {proj:,}  [TRAINABLE]")
        print(f"  lambda_sf:          {self.lambda_sf}")