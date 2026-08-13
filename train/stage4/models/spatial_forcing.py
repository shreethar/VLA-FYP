"""Spatial Forcing components for Stage 4.

This module follows the token-level alignment path used by the official
Spatial-Forcing implementation:

1. read an intermediate visual hidden state from the trainable VLA;
2. read frozen patch tokens from an intermediate VGGT aggregator layer;
3. spatially resample VGGT tokens to the VLA token grid;
4. project VLA tokens with a two-layer MLP; and
5. maximize tokenwise cosine similarity.

Only the alignment projector is trainable here. VGGT is always frozen and is
not needed at inference time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentProjector(nn.Module):
    """Two-layer projector matching the official Spatial-Forcing design."""

    def __init__(
        self,
        student_dim: int,
        target_dim: int,
        use_input_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(student_dim) if use_input_norm else nn.Identity()
        self.fc1 = nn.Linear(student_dim, target_dim, bias=True)
        self.fc2 = nn.Linear(target_dim, target_dim, bias=True)
        self.activation = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, visual_hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.input_norm(visual_hidden)
        return self.fc2(self.activation(self.fc1(hidden)))


@dataclass
class VGGTFeatureBatch:
    """Variable-view VGGT patch features for one batch."""

    features: List[torch.Tensor]  # each [views, patch_count, feature_dim]
    patch_grid: Tuple[int, int]


class VGGTExtractor(nn.Module):
    """Frozen intermediate-feature extractor backed by VGGT's aggregator.

    ``layer_index`` follows the zero-based indexing used by VGGT's
    ``aggregated_tokens_list`` and by the upstream Spatial-Forcing
    ``layers_align`` setting. Official VGGT normally caches only layers
    ``(4, 11, 17, 23)``; the requested layer is explicitly added to the cache.
    """

    def __init__(
        self,
        checkpoint: str = "facebook/VGGT-1B",
        layer_index: int = 8,
    ) -> None:
        super().__init__()
        try:
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise ImportError(
                "VGGT is required for Stage 4. Install train/stage4/requirements.txt."
            ) from exc

        full_model = VGGT.from_pretrained(checkpoint)
        self.aggregator = full_model.aggregator
        del full_model

        depth = int(getattr(self.aggregator, "depth", 24))
        if not 0 <= layer_index < depth:
            raise ValueError(
                f"VGGT layer_index must be in [0, {depth - 1}], got {layer_index}"
            )
        self.layer_index = layer_index
        self.patch_size = int(getattr(self.aggregator, "patch_size", 14))

        cached = getattr(self.aggregator, "cached_layer_indices", None)
        if cached is not None:
            if hasattr(cached, "add"):
                cached.add(layer_index)
            else:
                self.aggregator.cached_layer_indices = set(cached) | {layer_index}

        base_dim = int(self.aggregator.camera_token.shape[-1])
        # VGGT concatenates frame-attention and global-attention states.
        self.output_dim = 2 * base_dim
        self._freeze()

    def _freeze(self) -> None:
        self.aggregator.requires_grad_(False)
        self.aggregator.eval()

    def train(self, mode: bool = True):
        # A parent module's train() call must never enable VGGT checkpointing or
        # dropout. The extractor remains inference-only throughout Stage 4.
        super().train(False)
        self.aggregator.eval()
        return self

    @torch.no_grad()
    def _extract_uniform_views(self, images: torch.Tensor) -> torch.Tensor:
        """Extract [B, V, P, D] for a batch with the same view count."""
        if images.ndim != 5:
            raise ValueError(
                f"VGGT images must have shape [B,V,3,H,W], got {tuple(images.shape)}"
            )
        if images.shape[2] != 3:
            raise ValueError("VGGT images must be RGB")
        if images.min().item() < 0.0 or images.max().item() > 1.0:
            raise ValueError("VGGT images must be in the [0,1] range")

        dtype = next(self.aggregator.parameters()).dtype
        outputs, patch_start_idx = self.aggregator(images.to(dtype=dtype))
        features = outputs[self.layer_index]
        if features is None:
            raise RuntimeError(
                f"VGGT layer {self.layer_index} was not cached. "
                "Check the installed VGGT aggregator implementation."
            )
        return features[:, :, patch_start_idx:, :]

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> VGGTFeatureBatch:
        """Extract frozen features while supporting padded view batches."""
        if images.ndim == 4:
            images = images.unsqueeze(1)
        batch_size, max_views, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"VGGT image size {(height, width)} must be divisible by {self.patch_size}"
            )

        if view_mask is None:
            view_mask = torch.ones(
                batch_size, max_views, device=images.device, dtype=torch.bool
            )
        else:
            view_mask = view_mask.to(device=images.device, dtype=torch.bool)
        view_counts = view_mask.sum(dim=1)
        if (view_counts == 0).any():
            raise ValueError("Every Stage 4 sample must contain at least one image")

        per_sample: List[torch.Tensor] = []
        if torch.equal(view_counts, view_counts[:1].expand_as(view_counts)):
            views = int(view_counts[0].item())
            uniform_images = torch.stack(
                [images[i, view_mask[i]][:views] for i in range(batch_size)], dim=0
            )
            features = self._extract_uniform_views(uniform_images)
            per_sample.extend(features[i] for i in range(batch_size))
        else:
            # VGGT global attention mixes views, so padded frames cannot simply
            # be masked after extraction. Process variable-view samples alone.
            for batch_idx in range(batch_size):
                sample = images[batch_idx, view_mask[batch_idx]].unsqueeze(0)
                per_sample.append(self._extract_uniform_views(sample)[0])

        return VGGTFeatureBatch(
            features=per_sample,
            patch_grid=(height // self.patch_size, width // self.patch_size),
        )


def _closest_grid(token_count: int, aspect_ratio: float) -> Tuple[int, int]:
    """Factor ``token_count`` into the grid closest to an aspect ratio."""
    best = (1, token_count)
    best_error = float("inf")
    for height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % height:
            continue
        width = token_count // height
        for candidate in ((height, width), (width, height)):
            error = abs(candidate[1] / candidate[0] - aspect_ratio)
            if error < best_error:
                best = candidate
                best_error = error
    return best


def _add_vggt_position_embedding(
    feature_map: torch.Tensor,
    image_hw: Tuple[int, int],
    ratio: float = 0.1,
) -> torch.Tensor:
    """Apply the optional VGGT UV embedding used by upstream SF."""
    try:
        from vggt.heads.utils import create_uv_grid, position_grid_to_embed
    except ImportError as exc:
        raise ImportError("VGGT positional-embedding helpers are unavailable") from exc

    image_h, image_w = image_hw
    patch_h, patch_w = feature_map.shape[-2:]
    grid = create_uv_grid(
        patch_w,
        patch_h,
        aspect_ratio=image_w / image_h,
        dtype=feature_map.dtype,
        device=feature_map.device,
    )
    position = position_grid_to_embed(grid, feature_map.shape[1])
    position = position.permute(2, 0, 1).unsqueeze(0)
    return feature_map + ratio * position


class SpatialForcingAlignment(nn.Module):
    """Trainable projector and raw ``-cosine`` Spatial Forcing loss."""

    def __init__(
        self,
        student_dim: int,
        vggt_dim: int,
        use_input_norm: bool = False,
        use_vggt_position_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.projector = AlignmentProjector(
            student_dim=student_dim,
            target_dim=vggt_dim,
            use_input_norm=use_input_norm,
        )
        self.use_vggt_position_embedding = use_vggt_position_embedding

    def _resample_target(
        self,
        target: torch.Tensor,
        patch_grid: Tuple[int, int],
        target_token_count: int,
    ) -> torch.Tensor:
        """Bilinearly resample [V,P,D] VGGT patches to VLA token count."""
        views, patch_count, feature_dim = target.shape
        patch_h, patch_w = patch_grid
        if patch_count != patch_h * patch_w:
            raise ValueError(
                f"VGGT patch count {patch_count} does not match grid {patch_grid}"
            )

        per_view_count = max(1, round(target_token_count / views))
        target_h, target_w = _closest_grid(
            per_view_count, aspect_ratio=patch_w / patch_h
        )
        feature_map = target.permute(0, 2, 1).reshape(
            views, feature_dim, patch_h, patch_w
        )
        if self.use_vggt_position_embedding:
            feature_map = _add_vggt_position_embedding(
                feature_map,
                image_hw=(patch_h * 14, patch_w * 14),
            )
        feature_map = F.interpolate(
            feature_map.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True,
        )
        flattened = feature_map.flatten(2).transpose(1, 2).reshape(-1, feature_dim)

        if flattened.shape[0] != target_token_count:
            # Handles temporal merging or non-factorable token layouts while
            # preserving the 2-D interpolation whenever possible.
            flattened = F.interpolate(
                flattened.transpose(0, 1).unsqueeze(0),
                size=target_token_count,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
        return flattened

    def forward(
        self,
        student_visual: torch.Tensor,
        student_visual_mask: torch.Tensor,
        vggt_features: VGGTFeatureBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss, mean_cosine_similarity)`` over valid visual tokens."""
        if student_visual.ndim != 3 or student_visual_mask.ndim != 2:
            raise ValueError("Student visual features/mask must be [B,N,D] and [B,N]")
        if student_visual.shape[:2] != student_visual_mask.shape:
            raise ValueError("Student visual feature and mask shapes do not match")
        if len(vggt_features.features) != student_visual.shape[0]:
            raise ValueError("Student and VGGT batch sizes do not match")

        sample_similarities: List[torch.Tensor] = []
        for batch_idx, target in enumerate(vggt_features.features):
            valid_student = student_visual[batch_idx, student_visual_mask[batch_idx]]
            if valid_student.shape[0] == 0:
                raise ValueError("Spatial Forcing received a sample with no visual tokens")
            projected = self.projector(valid_student.float())
            target_resampled = self._resample_target(
                target.detach(),
                vggt_features.patch_grid,
                valid_student.shape[0],
            )
            cosine = F.cosine_similarity(
                projected,
                target_resampled.to(projected.device, dtype=projected.dtype),
                dim=-1,
            )
            sample_similarities.append(cosine.mean())

        mean_cosine = torch.stack(sample_similarities).mean()
        return -mean_cosine, mean_cosine.detach()


# Backward-compatible aliases for older notebooks.
ProjectionMLP = AlignmentProjector
SpatialForcingLoss = SpatialForcingAlignment
