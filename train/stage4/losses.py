"""Losses for three-model Spatial Forcing fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F


def _stack_latents(latents: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(latents, torch.Tensor):
        if latents.ndim != 3:
            raise ValueError("Latent tensor must have shape [B,M,D]")
        return latents
    if not latents:
        raise ValueError("At least one latent state is required")
    return torch.stack(list(latents), dim=1)


def latent_reasoning_preservation_loss(
    sf_latents: Sequence[torch.Tensor] | torch.Tensor,
    reference_latents: Sequence[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Mean ``1-cos`` over batch items and all M reasoning states."""
    sf = _stack_latents(sf_latents).float()
    reference = _stack_latents(reference_latents).detach().to(
        device=sf.device, dtype=sf.dtype
    )
    if sf.shape != reference.shape:
        raise ValueError(
            f"Student/reference latent shapes differ: {tuple(sf.shape)} vs "
            f"{tuple(reference.shape)}"
        )
    return (1.0 - F.cosine_similarity(sf, reference, dim=-1)).mean()


def waypoint_loss(
    predicted_waypoints: torch.Tensor,
    ground_truth_waypoints: torch.Tensor,
) -> torch.Tensor:
    """Mean squared Euclidean distance over K waypoint pairs.

    This implements ``(1/K) * sum_i ||p_hat_i - p_i||_2^2`` rather than
    coordinate-wise MSE, which differs by a factor of two for 2-D waypoints.
    """
    ground_truth = ground_truth_waypoints.detach().to(
        device=predicted_waypoints.device,
        dtype=predicted_waypoints.dtype,
    )
    if predicted_waypoints.shape != ground_truth.shape:
        raise ValueError(
            f"Waypoint shapes differ: {tuple(predicted_waypoints.shape)} vs "
            f"{tuple(ground_truth.shape)}"
        )
    if predicted_waypoints.ndim != 3 or predicted_waypoints.shape[-1] != 2:
        raise ValueError("Waypoints must have shape [B,K,2]")
    return (predicted_waypoints - ground_truth).square().sum(dim=-1).mean()


@dataclass
class Stage4LossOutput:
    total: torch.Tensor
    latent: torch.Tensor
    waypoint: torch.Tensor
    spatial_forcing: torch.Tensor
    spatial_cosine: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss/total": float(self.total.detach().item()),
            "loss/latent": float(self.latent.detach().item()),
            "loss/waypoint": float(self.waypoint.detach().item()),
            "loss/spatial_forcing": float(self.spatial_forcing.detach().item()),
            "spatial/cosine": float(self.spatial_cosine.detach().item()),
        }


def combine_stage4_losses(
    latent: torch.Tensor,
    waypoint: torch.Tensor,
    spatial_forcing: torch.Tensor,
    spatial_cosine: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
) -> Stage4LossOutput:
    total = alpha * latent + beta * waypoint + gamma * spatial_forcing
    return Stage4LossOutput(
        total=total,
        latent=latent,
        waypoint=waypoint,
        spatial_forcing=spatial_forcing,
        spatial_cosine=spatial_cosine,
    )
