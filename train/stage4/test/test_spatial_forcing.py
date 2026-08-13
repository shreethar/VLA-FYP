import torch

from train.stage4.losses import (
    latent_reasoning_preservation_loss,
    waypoint_loss,
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
