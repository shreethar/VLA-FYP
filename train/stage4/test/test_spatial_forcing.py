import torch

from train.stage4.losses import (
    latent_reasoning_preservation_loss,
    waypoint_loss,
)
from train.stage4.inspect_spatial_correspondence import (
    current_count_resample_vggt,
    derive_qwen_grid_specs,
    make_coordinate_features,
    metadata_resample_vggt,
)
from train.stage4.models.spatial_forcing import (
    SpatialForcingAlignment,
    VGGTFeatureBatch,
    _closest_grid,
)


def test_latent_preservation_is_zero_for_identical_states():
    latents = [torch.randn(3, 7) for _ in range(6)]
    loss = latent_reasoning_preservation_loss(latents, [z.clone() for z in latents])
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_latent_preservation_is_two_for_opposite_states():
    latents = [torch.randn(2, 5) for _ in range(6)]
    loss = latent_reasoning_preservation_loss(latents, [-z for z in latents])
    assert torch.allclose(loss, torch.tensor(2.0), atol=1e-6)


def test_waypoint_loss_is_squared_euclidean_not_coordinate_mse():
    predicted = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    target = torch.zeros_like(predicted)
    # ((1^2 + 2^2) + 0) / K=2
    assert torch.allclose(waypoint_loss(predicted, target), torch.tensor(2.5))


def test_spatial_alignment_is_tokenwise_and_differentiable():
    alignment = SpatialForcingAlignment(student_dim=4, vggt_dim=3)
    student = torch.randn(2, 4, 4, requires_grad=True)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    targets = VGGTFeatureBatch(
        features=[torch.randn(1, 4, 3), torch.randn(1, 4, 3)],
        patch_grid=(2, 2),
    )
    loss, cosine = alignment(student, mask, targets)
    assert loss.ndim == 0
    assert cosine.ndim == 0
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_closest_grid_preserves_token_count_and_aspect_ratio():
    height, width = _closest_grid(256, aspect_ratio=1.0)
    assert (height, width) == (16, 16)
    assert height * width == 256


def test_qwen_grid_metadata_applies_spatial_merge():
    specs = derive_qwen_grid_specs(torch.tensor([[1, 32, 48]]), spatial_merge_size=2)
    assert specs[0].source_thw == (1, 32, 48)
    assert specs[0].merged_thw == (1, 16, 24)
    assert specs[0].expected_tokens == 384
    assert specs[0].divisible_by_merge


def test_metadata_resampling_preserves_explicit_spatial_grid():
    coordinates = make_coordinate_features(views=1, grid=(4, 4), device=torch.device("cpu"))
    resized = metadata_resample_vggt(
        coordinates, source_grid=(4, 4), target_grid=(1, 2, 2)
    )
    assert resized.shape == (4, 3)
    expected = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
    )
    assert torch.allclose(resized, expected, atol=1e-6)


def test_metadata_resampling_handles_temporal_grid_mismatch():
    coordinates = make_coordinate_features(views=4, grid=(2, 2), device=torch.device("cpu"))
    resized = metadata_resample_vggt(
        coordinates, source_grid=(2, 2), target_grid=(2, 2, 2)
    )
    assert resized.shape == (8, 3)
    assert torch.allclose(resized[:4, 0], torch.zeros(4), atol=1e-6)
    assert torch.allclose(resized[4:, 0], torch.ones(4), atol=1e-6)


def test_count_resampler_reports_inferred_grid():
    coordinates = make_coordinate_features(views=1, grid=(4, 4), device=torch.device("cpu"))
    resized, inferred = current_count_resample_vggt(
        coordinates, source_grid=(4, 4), target_token_count=4
    )
    assert inferred == (2, 2)
    assert resized.shape == (4, 3)
