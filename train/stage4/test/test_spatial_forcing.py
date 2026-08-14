import io
import sys

import torch
import torch.nn as nn
from PIL import Image

from train.stage4.checkpointing import restore_stage4_training_state
from train.stage4.losses import (
    latent_reasoning_preservation_loss,
    relative_kv_drift,
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
from train.stage4.stage4_dataloader import (
    MolmoActStage4Dataset,
    MATERIALIZED_FORMAT_VERSION,
    _validate_materialized_manifest,
    build_trajectory_prompt,
    extract_task_name,
    is_sampled_fingerprint,
    parse_molmoact_annotation,
    partition_for_fingerprint,
)
from train.stage4.train_stage4 import (
    Stage4Config,
    _build_optimizer_groups,
    _parameter_grad_norm,
    _update_early_stopping,
    _validate_config,
    parse_args,
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
    mask = torch.ones(2, 4, dtype=torch.bool)
    targets = VGGTFeatureBatch(
        features=[torch.randn(2, 4, 3), torch.randn(2, 4, 3)],
        patch_grid=(2, 2),
    )
    loss, cosine = alignment(
        student,
        mask,
        targets,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
        spatial_merge_size=2,
        planner_view_indices=torch.tensor([0, 0]),
    )
    assert loss.ndim == 0
    assert cosine.ndim == 0
    assert torch.allclose(loss, 1.0 - cosine, atol=1e-6)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert isinstance(alignment.projector.input_norm, nn.BatchNorm1d)
    assert int(alignment.projector.input_norm.num_batches_tracked.item()) == 1


def test_spatial_alignment_uses_only_the_selected_planner_view():
    torch.manual_seed(3)
    alignment = SpatialForcingAlignment(student_dim=4, vggt_dim=3)
    student = torch.randn(1, 4, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)
    primary = torch.randn(1, 4, 3)
    target_a = VGGTFeatureBatch(
        features=[torch.cat([primary, torch.zeros_like(primary)], dim=0)],
        patch_grid=(2, 2),
    )
    target_b = VGGTFeatureBatch(
        features=[torch.cat([primary, torch.full_like(primary, 1_000.0)], dim=0)],
        patch_grid=(2, 2),
    )
    kwargs = {
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "spatial_merge_size": 2,
        "planner_view_indices": torch.tensor([0]),
    }
    loss_a, _ = alignment(student, mask, target_a, **kwargs)
    loss_b, _ = alignment(student, mask, target_b, **kwargs)
    assert torch.allclose(loss_a, loss_b, atol=1e-7)


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


def test_molmoact_annotation_filter_and_one_to_256_normalization():
    assert parse_molmoact_annotation(None) is None
    assert parse_molmoact_annotation([[10, 20]] * 4) is None
    waypoints = parse_molmoact_annotation(
        [[1, 1], [256, 256], [128.5, 64], [10, 20], [30, 40]]
    )
    assert waypoints is not None
    assert waypoints.shape == (5, 2)
    assert torch.equal(waypoints[0], torch.tensor([0.0, 0.0]))
    assert torch.equal(waypoints[1], torch.tensor([1.0, 1.0]))
    assert torch.all((0.0 <= waypoints) & (waypoints <= 1.0))


def test_molmoact_task_extraction_discards_the_remainder():
    conversation = {
        "from": ["human", "gpt"],
        "value": [
            "The task is close the box. Notice that the trajectory is annotated.",
            "irrelevant answer",
        ],
    }
    task = extract_task_name(conversation)
    assert task == "close the box"
    prompt = build_trajectory_prompt(task)
    assert "Task: close the box." in prompt
    assert "Notice that" not in prompt


def test_molmoact_students_get_primary_while_vggt_gets_both_views():
    class FakeProcessor:
        def __init__(self):
            self.images = None
            self.messages = None

        def apply_chat_template(self, messages, **_kwargs):
            self.messages = messages
            return "rendered prompt"

        def __call__(self, *, images, **_kwargs):
            self.images = images
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.zeros(1, 4),
                "image_grid_thw": torch.tensor([[1, 4, 4]]),
            }

    processor = FakeProcessor()
    primary = Image.new("RGB", (32, 32), color=(255, 0, 0))
    wrist = Image.new("RGB", (32, 32), color=(0, 0, 255))
    row = {
        "primary": primary,
        "wrist": wrist,
        "conversations": {
            "from": ["human", "gpt"],
            "value": ["The task is close the box. Notice the trace.", "answer"],
        },
        "annotation": [[10, 20], [20, 30], [30, 40], [40, 50], [50, 60]],
    }
    dataset = MolmoActStage4Dataset(
        [row],
        processor=processor,
        sample_ratio=1.0,
        split_ratios=(1.0, 0.0, 0.0),
    )
    sample = next(iter(dataset))

    assert len(processor.images) == 1
    assert processor.images[0].getpixel((0, 0)) == (255, 0, 0)
    assert sample["vggt_images"].shape == (2, 3, 518, 518)
    assert sample["vggt_images"][0, 0].mean() > 0.99
    assert sample["vggt_images"][0, 2].mean() < 0.01
    assert sample["vggt_images"][1, 2].mean() > 0.99
    assert sample["planner_view_index"] == 0


def test_materialized_row_is_not_sampled_or_hashed_again():
    class FakeProcessor:
        def apply_chat_template(self, _messages, **_kwargs):
            return "rendered prompt"

        def __call__(self, **_kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.zeros(1, 4),
                "image_grid_thw": torch.tensor([[1, 4, 4]]),
            }

    def image_bytes(color):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color=color).save(buffer, format="PNG")
        return buffer.getvalue()

    row = {
        "primary": image_bytes((255, 0, 0)),
        "wrist": image_bytes((0, 0, 255)),
        "task_name": "close the box",
        "fingerprint": "already-selected",
        "data_partition": "train",
        "annotation": "[[10,20],[20,30],[30,40],[40,50],[50,60]]",
    }
    dataset = MolmoActStage4Dataset(
        [row],
        processor=FakeProcessor(),
        sample_ratio=0.1,
        data_partition="train",
        preselected=True,
    )
    sample = next(iter(dataset))
    assert sample["sample_id"] == "molmoact_already-selected"
    assert sample["task_name"] == "close the box"


def test_materialized_manifest_must_match_training_recipe():
    manifest = {
        "format_version": MATERIALIZED_FORMAT_VERSION,
        "complete": True,
        "sample_ratio": 0.1,
        "seed": 42,
        "split_ratios": [0.70, 0.15, 0.15],
    }
    _validate_materialized_manifest(
        manifest,
        sample_ratio=0.1,
        seed=42,
        split_ratios=(0.70, 0.15, 0.15),
    )

    mismatched = dict(manifest, seed=7)
    try:
        _validate_materialized_manifest(
            mismatched,
            sample_ratio=0.1,
            seed=42,
            split_ratios=(0.70, 0.15, 0.15),
        )
    except ValueError as error:
        assert "seed" in str(error)
    else:
        raise AssertionError("Mismatched materialization seed was accepted")


def test_streaming_partitions_are_deterministic_disjoint_and_approximately_70_15_15():
    fingerprints = [f"record-{index}" for index in range(10_000)]
    assignments = [partition_for_fingerprint(value, seed=42) for value in fingerprints]
    assert assignments == [
        partition_for_fingerprint(value, seed=42) for value in fingerprints
    ]
    counts = {name: assignments.count(name) for name in ("train", "validation", "test")}
    assert 0.68 <= counts["train"] / len(assignments) <= 0.72
    assert 0.13 <= counts["validation"] / len(assignments) <= 0.17
    assert 0.13 <= counts["test"] / len(assignments) <= 0.17
    assert partition_for_fingerprint("exact-duplicate") == partition_for_fingerprint(
        "exact-duplicate"
    )
    assert is_sampled_fingerprint("row", 1.0)


def test_optimizer_uses_requested_layer_dependent_groups():
    class FakeStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList(
                [nn.Linear(1, 1, bias=False) for _ in range(32)]
            )
            self.spatial_mlp = nn.Linear(1, 2)
            self.spatial_tokens = nn.Parameter(
                torch.zeros(5, 1), requires_grad=False
            )

    config = Stage4Config()
    projector = nn.Linear(1, 1)
    optimizer_groups, grouped = _build_optimizer_groups(
        FakeStudent(), projector, config
    )
    by_name = {group["group_name"]: group for group in optimizer_groups}
    assert len(grouped["qwen_layers_0_7"]) == 8
    assert len(grouped["qwen_layers_8_31"]) == 24
    assert by_name["qwen_layers_0_7"]["lr"] == 5e-7
    assert by_name["qwen_layers_8_31"]["lr"] == 5e-8
    assert by_name["waypoint_head"]["lr"] == 5e-7
    assert by_name["sf_projector"]["lr"] == 1e-5
    assert config.alpha == 1.0
    assert config.beta == 3.0
    assert config.gamma == 0.025


def test_relative_kv_drift_uses_frozen_reference_norm():
    reference = torch.tensor([[[3.0, 4.0], [99.0, 99.0]]])
    student = torch.tensor([[[6.0, 8.0], [-50.0, -50.0]]])
    mask = torch.tensor([[True, False]])
    # ||[3,4]|| / ||[3,4]|| = 1; masked padding must not contribute.
    drift = relative_kv_drift(student, reference, mask)
    assert torch.allclose(drift, torch.tensor(1.0))


def test_incomplete_materialized_opt_in_requires_a_local_directory():
    config = Stage4Config(allow_incomplete_materialized=True)
    try:
        _validate_config(config)
    except ValueError as error:
        assert "materialized_data_dir" in str(error)
    else:
        raise AssertionError("Incomplete-data opt-in was accepted without a path")


def test_parameter_group_gradient_norm_is_pre_clip_l2_norm():
    first = nn.Parameter(torch.tensor([0.0]))
    second = nn.Parameter(torch.tensor([0.0]))
    first.grad = torch.tensor([3.0])
    second.grad = torch.tensor([4.0])
    assert _parameter_grad_norm([first, second]) == 5.0


def test_wandb_configuration_validation():
    _validate_config(Stage4Config(wandb_mode="offline"))
    try:
        _validate_config(Stage4Config(wandb_mode="invalid"))
    except ValueError as error:
        assert "wandb_mode" in str(error)
    else:
        raise AssertionError("Invalid W&B mode was accepted")


def test_early_stopping_requires_minimum_improvement_and_resets_patience():
    best, bad, improved = _update_early_stopping(
        validation_loss=0.99,
        best_validation_loss=1.0,
        bad_evaluations=3,
        min_delta=0.001,
    )
    assert improved
    assert best == 0.99
    assert bad == 0

    best, bad, improved = _update_early_stopping(
        validation_loss=0.9895,
        best_validation_loss=best,
        bad_evaluations=bad,
        min_delta=0.001,
    )
    assert not improved
    assert best == 0.99
    assert bad == 1


def test_early_stopping_configuration_validation():
    _validate_config(Stage4Config(eval_steps=100, eval_batches=2))
    try:
        _validate_config(Stage4Config(early_stopping_patience=0))
    except ValueError as error:
        assert "early_stopping_patience" in str(error)
    else:
        raise AssertionError("Zero early-stopping patience was accepted")


def test_stage4_resume_restores_early_stopping_metadata(tmp_path):
    alignment = nn.Linear(2, 2)
    checkpoint_dir = tmp_path / "step_000500"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "step": 500,
            "spatial_alignment": alignment.state_dict(),
            "optimizer": None,
            "scheduler": None,
            "training_metadata": {
                "best_validation_loss": 1.25,
                "early_stopping_bad_evals": 3,
            },
        },
        checkpoint_dir / "stage4_state.pt",
    )
    restored = {}
    step = restore_stage4_training_state(
        checkpoint_dir,
        alignment,
        training_metadata_out=restored,
    )
    assert step == 500
    assert restored["best_validation_loss"] == 1.25
    assert restored["early_stopping_bad_evals"] == 3


def test_no_eval_cli_also_disables_early_stopping(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_stage4.py", "--no_eval", "--no_wandb"],
    )
    config = parse_args()
    assert not config.evaluate
    assert not config.early_stopping
    _validate_config(config)
