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
    """Normalization followed by the two-layer Spatial-Forcing MLP."""

    def __init__(
        self,
        student_dim: int,
        target_dim: int,
        normalization: str = "batchnorm",
    ) -> None:
        super().__init__()
        if normalization == "batchnorm":
            self.input_norm = nn.BatchNorm1d(student_dim)
        elif normalization == "layernorm":
            self.input_norm = nn.LayerNorm(student_dim)
        elif normalization == "none":
            self.input_norm = nn.Identity()
        else:
            raise ValueError(
                "normalization must be one of: batchnorm, layernorm, none"
            )
        self.normalization = normalization
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
        if features.ndim != 4:
            raise RuntimeError(
                "VGGT cached aggregator features must preserve [B,V,P,D], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[:2] != images.shape[:2]:
            raise RuntimeError(
                "VGGT did not preserve the input view axis: "
                f"input B,V={tuple(images.shape[:2])}, "
                f"feature B,V={tuple(features.shape[:2])}"
            )
        patches = features[:, :, patch_start_idx:, :]
        expected_patches = (images.shape[-2] // self.patch_size) * (
            images.shape[-1] // self.patch_size
        )
        if patches.shape[2] != expected_patches:
            raise RuntimeError(
                f"VGGT patch slice has {patches.shape[2]} tokens per view; "
                f"expected {expected_patches} from the input grid"
            )
        return patches

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
    """Planner-view projector and ``1 - cosine`` Spatial Forcing loss.

    Qwen and VGGT token counts are never treated as correspondence metadata.
    Qwen's processor-provided ``image_grid_thw`` is converted to the
    post-merge target grid, and only the selected VGGT planner view is
    interpolated onto that grid.
    """

    def __init__(
        self,
        student_dim: int,
        vggt_dim: int,
        normalization: str = "batchnorm",
        use_vggt_position_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.projector = AlignmentProjector(
            student_dim=student_dim,
            target_dim=vggt_dim,
            normalization=normalization,
        )
        self.use_vggt_position_embedding = use_vggt_position_embedding

    def _resample_target(
        self,
        target: torch.Tensor,
        patch_grid: Tuple[int, int],
        target_grid: Tuple[int, int, int],
        vggt_patch_size: int,
    ) -> torch.Tensor:
        """Bilinearly resample one ``[P,D]`` VGGT view to explicit Qwen THW."""
        if target.ndim != 2:
            raise ValueError("Selected VGGT planner-view features must be [P,D]")
        patch_count, feature_dim = target.shape
        patch_h, patch_w = patch_grid
        if patch_count != patch_h * patch_w:
            raise ValueError(
                f"VGGT patch count {patch_count} does not match grid {patch_grid}"
            )
        target_t, target_h, target_w = target_grid
        if target_t != 1:
            raise ValueError(
                "Planner-view Spatial Forcing expects one Qwen image (T=1), "
                f"but image_grid_thw implies T={target_t}"
            )
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Invalid Qwen target grid {target_grid}")

        feature_map = target.transpose(0, 1).reshape(
            1, feature_dim, patch_h, patch_w
        )
        if self.use_vggt_position_embedding:
            feature_map = _add_vggt_position_embedding(
                feature_map,
                image_hw=(patch_h * vggt_patch_size, patch_w * vggt_patch_size),
            )
        feature_map = F.interpolate(
            feature_map.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True,
        )
        return feature_map.flatten(2).transpose(1, 2).squeeze(0)

    def forward(
        self,
        student_visual: torch.Tensor,
        student_visual_mask: torch.Tensor,
        vggt_features: VGGTFeatureBatch,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int,
        planner_view_indices: Optional[torch.Tensor] = None,
        vggt_patch_size: int = 14,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss, mean_cosine_similarity)`` over valid visual tokens."""
        if student_visual.ndim != 3 or student_visual_mask.ndim != 2:
            raise ValueError("Student visual features/mask must be [B,N,D] and [B,N]")
        if student_visual.shape[:2] != student_visual_mask.shape:
            raise ValueError("Student visual feature and mask shapes do not match")
        if len(vggt_features.features) != student_visual.shape[0]:
            raise ValueError("Student and VGGT batch sizes do not match")
        if image_grid_thw is None or image_grid_thw.ndim != 2:
            raise ValueError("image_grid_thw must be [B,3] for primary-only Qwen input")
        if image_grid_thw.shape != (student_visual.shape[0], 3):
            raise ValueError(
                "Expected exactly one Qwen image_grid_thw row per sample; "
                f"got {tuple(image_grid_thw.shape)} for batch {student_visual.shape[0]}"
            )
        if spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if planner_view_indices is None:
            planner_view_indices = torch.zeros(
                student_visual.shape[0], dtype=torch.long, device=student_visual.device
            )
        if planner_view_indices.shape != (student_visual.shape[0],):
            raise ValueError("planner_view_indices must have shape [B]")

        valid_students: List[torch.Tensor] = []
        target_resampled_per_sample: List[torch.Tensor] = []
        for batch_idx, target in enumerate(vggt_features.features):
            valid_student = student_visual[batch_idx, student_visual_mask[batch_idx]]
            if valid_student.shape[0] == 0:
                raise ValueError("Spatial Forcing received a sample with no visual tokens")
            planner_view = int(planner_view_indices[batch_idx].item())
            if not 0 <= planner_view < target.shape[0]:
                raise ValueError(
                    f"Planner view {planner_view} is outside VGGT's {target.shape[0]} views"
                )

            time, grid_h, grid_w = (
                int(value) for value in image_grid_thw[batch_idx].tolist()
            )
            if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
                raise ValueError(
                    f"Qwen grid {(time, grid_h, grid_w)} is not divisible by "
                    f"spatial_merge_size={spatial_merge_size}"
                )
            merged_grid = (
                time,
                grid_h // spatial_merge_size,
                grid_w // spatial_merge_size,
            )
            expected_tokens = math.prod(merged_grid)
            if valid_student.shape[0] != expected_tokens:
                raise ValueError(
                    "Qwen layer visual-token count disagrees with image_grid_thw: "
                    f"layer={valid_student.shape[0]}, grid={merged_grid} "
                    f"({expected_tokens} tokens)"
                )
            target_resampled = self._resample_target(
                target[planner_view].detach(),
                vggt_features.patch_grid,
                merged_grid,
                vggt_patch_size,
            )
            valid_students.append(valid_student)
            target_resampled_per_sample.append(target_resampled)

        # Project once so BatchNorm observes all visual tokens in the optimizer
        # batch and updates its running statistics once per training step.
        token_counts = [tokens.shape[0] for tokens in valid_students]
        projected_batch = self.projector(
            torch.cat(valid_students, dim=0).float()
        )
        projected_per_sample = projected_batch.split(token_counts, dim=0)

        sample_similarities: List[torch.Tensor] = []
        for projected, target_resampled in zip(
            projected_per_sample, target_resampled_per_sample
        ):
            cosine = F.cosine_similarity(
                projected,
                target_resampled.to(projected.device, dtype=projected.dtype),
                dim=-1,
            )
            sample_similarities.append(cosine.mean())

        mean_cosine = torch.stack(sample_similarities).mean()
        return 1.0 - mean_cosine, mean_cosine.detach()


# Backward-compatible aliases for older notebooks.
ProjectionMLP = AlignmentProjector
SpatialForcingLoss = SpatialForcingAlignment
