"""Fine-tune a latent student with three-model Spatial Forcing.

Models
------
* frozen VGGT geometry teacher;
* frozen Stage 2 latent-student reference;
* trainable Stage 2 latent student with its five spatial slots frozen.

Objective
---------
    alpha * L_latent + beta * L_waypoint + gamma * L_spatial_forcing
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.stage4.checkpointing import (
    assert_spatial_tokens_frozen,
    load_latent_student_checkpoint,
    restore_stage4_training_state,
    save_stage4_checkpoint,
)
from train.stage4.losses import (
    combine_stage4_losses,
    latent_reasoning_preservation_loss,
    waypoint_loss,
)
from train.stage4.models.spatial_forcing import (
    SpatialForcingAlignment,
    VGGTExtractor,
)
from train.stage4.stage4_dataloader import build_stage4_dataloader
from train.stage4.stage4_dataloader import DEFAULT_HF_CONFIG, DEFAULT_HF_REPO

logger = logging.getLogger("stage4")


@dataclass
class Stage4Config:
    student_checkpoint: str = "shreethar/LatentStudent-ckpt-400"
    hf_repo: str = DEFAULT_HF_REPO
    hf_config: str = DEFAULT_HF_CONFIG
    materialized_data_dir: Optional[str] = None
    allow_incomplete_materialized: bool = False
    output_dir: str = "checkpoints/stage4"
    reference_checkpoint: Optional[str] = None
    resume_from: Optional[str] = None
    base_model_name: Optional[str] = None
    processor_name: Optional[str] = None
    vggt_checkpoint: str = "facebook/VGGT-1B"
    split: str = "train"
    data_partition: str = "train"
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    sample_ratio: float = 0.1
    max_seq_len: int = 1024
    batch_size: int = 16
    num_workers: int = 2
    M: int = 6
    K: int = 5
    student_visual_layer: int = 8
    vggt_layer: int = 8
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    qwen_layers_0_7_lr: float = 1e-5
    qwen_layers_8_31_lr: float = 1e-6
    waypoint_head_lr: float = 1e-5
    projector_lr: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 10000
    gradient_accumulation_steps: int = 1
    grad_clip: float = 1.0
    save_steps: int = 500
    log_steps: int = 10
    evaluate: bool = True
    eval_steps: int = 500
    eval_batches: int = 50
    eval_batch_size: Optional[int] = None
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    seed: int = 42
    projector_normalization: str = "batchnorm"
    use_vggt_position_embedding: bool = False
    use_wandb: bool = True
    wandb_project: str = "reasonflow-vla"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "stage4-spatial-forcing"
    wandb_run_id: Optional[str] = None
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ("stage4", "spatial-forcing", "molmoact")


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _move_optional(value, device: torch.device):
    return value.to(device, non_blocking=True) if value is not None else None


def _resume_wandb_run_id(config: Stage4Config) -> Optional[str]:
    """Recover the W&B ID stored in a Stage 4 checkpoint when resuming."""
    if config.wandb_run_id:
        return config.wandb_run_id
    if not config.resume_from:
        return None
    config_path = Path(config.resume_from) / "stage4_config.json"
    if not config_path.is_file():
        return None
    try:
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read W&B run ID from %s: %s", config_path, error)
        return None
    run_id = saved_config.get("wandb_run_id")
    return str(run_id) if run_id else None


def _init_wandb(config: Stage4Config):
    if not config.use_wandb:
        logger.info("W&B logging disabled")
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed. Install "
            "train/stage4/requirements.txt or pass --no_wandb."
        ) from error

    run_id = _resume_wandb_run_id(config)
    init_kwargs = {
        "project": config.wandb_project,
        "entity": config.wandb_entity,
        "name": config.wandb_run_name,
        "tags": list(config.wandb_tags),
        "config": asdict(config),
        "mode": config.wandb_mode,
        "job_type": "stage4-spatial-forcing",
    }
    if run_id:
        init_kwargs.update({"id": run_id, "resume": "allow"})
    run = wandb.init(**init_kwargs)
    if run is None:
        raise RuntimeError("wandb.init() did not return a run")
    config.wandb_run_id = run.id
    run.config.update({"wandb_run_id": run.id}, allow_val_change=True)
    run.define_metric("progress/optimizer_step")
    run.define_metric("*", step_metric="progress/optimizer_step")
    logger.info("W&B run: %s", run.url or f"offline:{run.id}")
    return run


def _parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    """L2 norm for one optimizer group before global clipping."""
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2).item())


def _write_best_checkpoint_pointer(
    output_dir: str,
    *,
    checkpoint_path: str,
    step: int,
    validation_loss: float,
) -> Path:
    """Atomically record which step owns the best validation objective."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "best_checkpoint.json"
    temporary = root / "best_checkpoint.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "checkpoint_path": checkpoint_path,
                "step": step,
                "monitor": "validation/loss/total",
                "validation_loss": validation_loss,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _update_early_stopping(
    *,
    validation_loss: float,
    best_validation_loss: float,
    bad_evaluations: int,
    min_delta: float,
) -> tuple[float, int, bool]:
    """Update a minimization monitor using an absolute improvement threshold."""
    improved = validation_loss < best_validation_loss - min_delta
    if improved:
        return validation_loss, 0, True
    return best_validation_loss, bad_evaluations + 1, False


def _detached_stage4_metrics(
    losses,
    predicted_waypoints: torch.Tensor,
    ground_truth: torch.Tensor,
    config: Stage4Config,
) -> dict[str, float]:
    """Scalar objective and trajectory-quality metrics shared by train/eval."""
    metrics = losses.detached_metrics()
    trajectory_error = (
        predicted_waypoints.detach().float() - ground_truth.detach().float()
    )
    latent_value = metrics["loss/latent"]
    waypoint_value = metrics["loss/waypoint"]
    spatial_value = metrics["loss/spatial_forcing"]
    metrics.update(
        {
            "loss_weighted/latent": config.alpha * latent_value,
            "loss_weighted/waypoint": config.beta * waypoint_value,
            "loss_weighted/spatial_forcing": config.gamma * spatial_value,
            "latent/cosine": 1.0 - latent_value,
            "trajectory/mae_normalized": float(
                trajectory_error.abs().mean().item()
            ),
            "trajectory/rmse_normalized": float(
                trajectory_error.square().mean().sqrt().item()
            ),
            "trajectory/mae_pixels": float(
                trajectory_error.abs().mean().mul(255.0).item()
            ),
            "trajectory/prediction_mean": float(
                predicted_waypoints.detach().float().mean().item()
            ),
            "trajectory/prediction_std": float(
                predicted_waypoints.detach().float().std(unbiased=False).item()
            ),
            "trajectory/target_mean": float(ground_truth.float().mean().item()),
            "trajectory/target_std": float(
                ground_truth.float().std(unbiased=False).item()
            ),
        }
    )
    return metrics


@torch.no_grad()
def _evaluate_stage4(
    dataloader,
    *,
    reference,
    student,
    vggt,
    spatial_alignment,
    device: torch.device,
    config: Stage4Config,
    qwen_spatial_merge_size: int,
) -> dict[str, float]:
    """Evaluate a stable prefix of the validation partition without gradients."""
    student_was_training = student.training
    alignment_was_training = spatial_alignment.training
    student.eval()
    spatial_alignment.eval()
    metric_sums: dict[str, float] = {}
    sample_count = 0
    batch_count = 0
    started = time.perf_counter()

    try:
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= config.eval_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            pixel_values = _move_optional(batch.get("pixel_values"), device)
            image_grid_thw = _move_optional(batch.get("image_grid_thw"), device)
            pixel_values_videos = _move_optional(
                batch.get("pixel_values_videos"), device
            )
            video_grid_thw = _move_optional(batch.get("video_grid_thw"), device)
            ground_truth = batch["gt_waypoints"].to(device, non_blocking=True)
            vggt_images = batch["vggt_images"].to(device, non_blocking=True)
            vggt_view_mask = batch["vggt_view_mask"].to(
                device, non_blocking=True
            )
            planner_view_indices = batch["planner_view_indices"].to(
                device, non_blocking=True
            )

            with _autocast_context(device):
                reference_latents = reference.generate_reasoning_latents(
                    input_ids,
                    pixel_values,
                    image_grid_thw,
                    attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                )
                geometry_features = vggt(vggt_images, vggt_view_mask)
                sf_latents, _, _, predicted_waypoints = student.generate_latents(
                    input_ids,
                    pixel_values,
                    image_grid_thw,
                    attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                )
                student_visual, student_visual_mask = (
                    student.get_mid_layer_visual_features(
                        input_ids,
                        pixel_values,
                        image_grid_thw,
                        attention_mask,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                        layer_idx=config.student_visual_layer,
                        return_mask=True,
                    )
                )

            latent = latent_reasoning_preservation_loss(
                sf_latents, reference_latents
            )
            waypoint = waypoint_loss(predicted_waypoints, ground_truth)
            spatial_forcing, spatial_cosine = spatial_alignment(
                student_visual,
                student_visual_mask,
                geometry_features,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=qwen_spatial_merge_size,
                planner_view_indices=planner_view_indices,
                vggt_patch_size=vggt.patch_size,
            )
            losses = combine_stage4_losses(
                latent=latent,
                waypoint=waypoint,
                spatial_forcing=spatial_forcing,
                spatial_cosine=spatial_cosine,
                alpha=config.alpha,
                beta=config.beta,
                gamma=config.gamma,
            )
            batch_size = int(input_ids.shape[0])
            for key, value in _detached_stage4_metrics(
                losses, predicted_waypoints, ground_truth, config
            ).items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size
            sample_count += batch_size
            batch_count += 1
    finally:
        student.train(student_was_training)
        spatial_alignment.train(alignment_was_training)

    if sample_count == 0:
        raise RuntimeError("Validation dataloader produced no samples")
    elapsed = time.perf_counter() - started
    metrics = {key: value / sample_count for key, value in metric_sums.items()}
    metrics.update(
        {
            "evaluation/batches": float(batch_count),
            "evaluation/samples": float(sample_count),
            "evaluation/seconds": elapsed,
            "evaluation/samples_per_second": sample_count / max(elapsed, 1e-12),
        }
    )
    return metrics


def _build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _resolve_processor_name(config: Stage4Config) -> str:
    if config.processor_name:
        return config.processor_name
    if config.base_model_name:
        return config.base_model_name
    checkpoint_path = Path(config.student_checkpoint)
    if checkpoint_path.exists() and (
        (checkpoint_path / "processor_config.json").is_file()
        or (checkpoint_path / "tokenizer_config.json").is_file()
    ):
        return config.student_checkpoint
    if not checkpoint_path.exists():
        return config.student_checkpoint
    raise ValueError(
        "A local adapter checkpoint requires --base_model_name or --processor_name"
    )


def _token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Tokenizer cannot encode required token {token!r}")
        token_id = encoded[-1]
    return int(token_id)


def _resolve_qwen_spatial_merge_size(student) -> int:
    """Read the vision-to-LLM merge factor from the loaded checkpoint."""
    visual = student._visual_encoder
    candidates = [
        visual,
        getattr(visual, "config", None),
        getattr(getattr(student.vlm, "config", None), "vision_config", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        for name in ("spatial_merge_size", "merge_size"):
            value = getattr(candidate, name, None)
            if value is not None:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"Invalid Qwen {name}={value}")
                return value
    raise RuntimeError(
        "Could not determine Qwen's spatial_merge_size from the loaded model"
    )


_QWEN_LAYER_RE = re.compile(
    r"(?:^|\.)(?:language_model|text_model)\.layers\.(\d+)(?:\.|$)"
)


def _build_optimizer_groups(student, spatial_alignment, config: Stage4Config):
    """Create strict, non-overlapping parameter groups for the requested LRs."""
    grouped: dict[str, list[torch.nn.Parameter]] = {
        "qwen_layers_0_7": [],
        "qwen_layers_8_31": [],
        "waypoint_head": [],
        "sf_projector": list(spatial_alignment.parameters()),
    }
    unclassified: list[str] = []

    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "spatial_tokens":
            raise RuntimeError("Spatial-token embeddings unexpectedly require gradients")
        if name.startswith("spatial_mlp."):
            grouped["waypoint_head"].append(parameter)
            continue
        match = _QWEN_LAYER_RE.search(name)
        if match is None:
            unclassified.append(name)
            continue
        layer_index = int(match.group(1))
        if not 0 <= layer_index < 32:
            unclassified.append(name)
        elif layer_index <= 7:
            grouped["qwen_layers_0_7"].append(parameter)
        else:
            grouped["qwen_layers_8_31"].append(parameter)

    if unclassified:
        preview = "\n  ".join(unclassified[:20])
        raise RuntimeError(
            "Trainable student parameters do not match the requested Qwen-layer/"
            f"waypoint-head LR policy:\n  {preview}"
        )
    empty = [name for name, parameters in grouped.items() if not parameters]
    if empty:
        raise RuntimeError(f"Optimizer parameter groups are empty: {empty}")

    learning_rates = {
        "qwen_layers_0_7": config.qwen_layers_0_7_lr,
        "qwen_layers_8_31": config.qwen_layers_8_31_lr,
        "waypoint_head": config.waypoint_head_lr,
        "sf_projector": config.projector_lr,
    }
    optimizer_groups = [
        {
            "params": parameters,
            "lr": learning_rates[name],
            "group_name": name,
        }
        for name, parameters in grouped.items()
    ]
    return optimizer_groups, grouped


def _validate_config(config: Stage4Config) -> None:
    if config.M != 6:
        raise ValueError("This Stage 4 recipe requires the checkpoint's six latent slots")
    if config.K != 5:
        raise ValueError("MolmoAct Stage 4 requires exactly five spatial slots")
    if not 0.0 < config.sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0,1]")
    split_ratios = (
        config.train_ratio,
        config.validation_ratio,
        config.test_ratio,
    )
    if any(ratio < 0.0 for ratio in split_ratios) or not math.isclose(
        sum(split_ratios), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("train/validation/test ratios must be non-negative and sum to 1")
    if config.data_partition not in {"train", "validation", "test"}:
        raise ValueError("data_partition must be train, validation, or test")
    if config.projector_normalization != "batchnorm":
        raise ValueError("This recipe requires projector_normalization=batchnorm")
    if not 0.0 <= config.adam_beta1 < 1.0 or not 0.0 <= config.adam_beta2 < 1.0:
        raise ValueError("AdamW beta values must be in [0,1)")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.allow_incomplete_materialized and not config.materialized_data_dir:
        raise ValueError(
            "allow_incomplete_materialized requires materialized_data_dir"
        )
    if config.log_steps < 1:
        raise ValueError("log_steps must be positive")
    if config.evaluate and config.eval_steps < 1:
        raise ValueError("eval_steps must be positive when evaluation is enabled")
    if config.evaluate and config.eval_batches < 1:
        raise ValueError("eval_batches must be positive when evaluation is enabled")
    if config.eval_batch_size is not None and config.eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    if config.early_stopping and not config.evaluate:
        raise ValueError("Early stopping requires evaluation")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if config.early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if config.use_wandb and not config.wandb_project.strip():
        raise ValueError("wandb_project cannot be empty when W&B is enabled")
    if config.wandb_mode not in {"online", "offline"}:
        raise ValueError("wandb_mode must be online or offline")


def train(config: Stage4Config, dataloader=None, validation_dataloader=None) -> None:
    _validate_config(config)
    wandb_run = _init_wandb(config)
    if not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; the three-model run will be extremely slow")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    from transformers import AutoTokenizer

    processor_name = _resolve_processor_name(config)
    tokenizer = AutoTokenizer.from_pretrained(
        processor_name, trust_remote_code=True
    )
    end_think_token_id = _token_id(tokenizer, "</think>")

    reference_source = config.reference_checkpoint or config.student_checkpoint
    trainable_source = config.resume_from or config.student_checkpoint

    logger.info("Loading frozen reference student from %s", reference_source)
    reference = load_latent_student_checkpoint(
        checkpoint=reference_source,
        base_model_name=config.base_model_name,
        end_think_token_id=end_think_token_id,
        trainable=False,
        M=config.M,
        K=config.K,
    ).to(device)

    logger.info("Loading trainable student from %s", trainable_source)
    student = load_latent_student_checkpoint(
        checkpoint=trainable_source,
        base_model_name=config.base_model_name,
        end_think_token_id=end_think_token_id,
        trainable=True,
        M=config.M,
        K=config.K,
    ).to(device)
    student.spatial_tokens.requires_grad_(False)
    assert_spatial_tokens_frozen(student)

    logger.info(
        "Loading frozen VGGT %s (aggregator layer %d)",
        config.vggt_checkpoint,
        config.vggt_layer,
    )
    vggt = VGGTExtractor(
        checkpoint=config.vggt_checkpoint,
        layer_index=config.vggt_layer,
    ).to(
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    spatial_alignment = SpatialForcingAlignment(
        student_dim=student.hidden_dim,
        vggt_dim=vggt.output_dim,
        normalization=config.projector_normalization,
        use_vggt_position_embedding=config.use_vggt_position_embedding,
    ).to(device)
    qwen_spatial_merge_size = _resolve_qwen_spatial_merge_size(student)

    optimizer_groups, grouped_parameters = _build_optimizer_groups(
        student, spatial_alignment, config
    )
    student_parameters = (
        grouped_parameters["qwen_layers_0_7"]
        + grouped_parameters["qwen_layers_8_31"]
        + grouped_parameters["waypoint_head"]
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer, config.warmup_steps, config.max_steps
    )

    start_step = 0
    restored_training_metadata: dict = {}
    if config.resume_from:
        start_step = restore_stage4_training_state(
            config.resume_from,
            spatial_alignment,
            optimizer,
            scheduler,
            training_metadata_out=restored_training_metadata,
        )
        logger.info("Resumed Stage 4 optimizer/projector at step %d", start_step)

    processor = None
    if dataloader is None or (config.evaluate and validation_dataloader is None):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            processor_name, trust_remote_code=True
        )

    if dataloader is None:
        dataloader = build_stage4_dataloader(
            processor=processor,
            hf_repo=config.hf_repo,
            hf_config=config.hf_config,
            split=config.split,
            data_partition=config.data_partition,
            split_ratios=(
                config.train_ratio,
                config.validation_ratio,
                config.test_ratio,
            ),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            max_length=config.max_seq_len,
            sample_ratio=config.sample_ratio,
            seed=config.seed,
            materialized_data_dir=config.materialized_data_dir,
            allow_incomplete_materialized=config.allow_incomplete_materialized,
        )
    if config.evaluate and validation_dataloader is None:
        validation_dataloader = build_stage4_dataloader(
            processor=processor,
            hf_repo=config.hf_repo,
            hf_config=config.hf_config,
            split=config.split,
            data_partition="validation",
            split_ratios=(
                config.train_ratio,
                config.validation_ratio,
                config.test_ratio,
            ),
            batch_size=config.eval_batch_size or config.batch_size,
            num_workers=config.num_workers,
            max_length=config.max_seq_len,
            sample_ratio=config.sample_ratio,
            seed=config.seed,
            materialized_data_dir=config.materialized_data_dir,
            allow_incomplete_materialized=config.allow_incomplete_materialized,
            drop_last=False,
        )

    logger.info(
        "Trainable parameters: qwen[0:7]=%s qwen[8:31]=%s waypoint=%s "
        "projector=%s; spatial slots frozen=%s",
        f"{sum(p.numel() for p in grouped_parameters['qwen_layers_0_7']):,}",
        f"{sum(p.numel() for p in grouped_parameters['qwen_layers_8_31']):,}",
        f"{sum(p.numel() for p in grouped_parameters['waypoint_head']):,}",
        f"{sum(p.numel() for p in grouped_parameters['sf_projector']):,}",
        not student.spatial_tokens.requires_grad,
    )
    logger.info(
        "Loss weights: alpha=%g beta=%g gamma=%g; student/VGGT layers=%d/%d",
        config.alpha,
        config.beta,
        config.gamma,
        config.student_visual_layer,
        config.vggt_layer,
    )
    data_source = (
        f"local Parquet {config.materialized_data_dir}"
        if config.materialized_data_dir
        else f"remote stream {config.hf_repo}[{config.hf_config}]"
    )
    logger.info(
        "Dataset: %s partition=%s ratios=%g/%g/%g, valid-row sample "
        "ratio=%g; Qwen spatial merge=%d",
        data_source,
        config.data_partition,
        config.train_ratio,
        config.validation_ratio,
        config.test_ratio,
        config.sample_ratio,
        qwen_spatial_merge_size,
    )
    logger.info(
        "AdamW betas=(%g,%g) weight_decay=%g; LRs qwen[0:7]=%g "
        "qwen[8:31]=%g waypoint=%g projector=%g; cosine warmup=%d/%d",
        config.adam_beta1,
        config.adam_beta2,
        config.weight_decay,
        config.qwen_layers_0_7_lr,
        config.qwen_layers_8_31_lr,
        config.waypoint_head_lr,
        config.projector_lr,
        config.warmup_steps,
        config.max_steps,
    )
    if config.evaluate:
        logger.info(
            "Validation every %d steps: up to %d batches of %d; early "
            "stopping=%s patience=%d min_delta=%g monitor=loss/total",
            config.eval_steps,
            config.eval_batches,
            config.eval_batch_size or config.batch_size,
            config.early_stopping,
            config.early_stopping_patience,
            config.early_stopping_min_delta,
        )

    reference.eval()
    vggt.eval()
    student.train()
    spatial_alignment.train()
    data_iterator = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    last_saved_step = start_step
    samples_seen = int(
        restored_training_metadata.get(
            "samples_seen",
            start_step * config.batch_size * config.gradient_accumulation_steps,
        )
    )
    best_total_loss = float(
        restored_training_metadata.get("best_total_loss", math.inf)
    )
    best_waypoint_loss = float(
        restored_training_metadata.get("best_waypoint_loss", math.inf)
    )
    best_latent_loss = float(
        restored_training_metadata.get("best_latent_loss", math.inf)
    )
    best_spatial_cosine = float(
        restored_training_metadata.get("best_spatial_cosine", -math.inf)
    )
    best_validation_loss = float(
        restored_training_metadata.get("best_validation_loss", math.inf)
    )
    early_stopping_bad_evals = int(
        restored_training_metadata.get("early_stopping_bad_evals", 0)
    )
    best_checkpoint_path = restored_training_metadata.get("best_checkpoint_path")
    stopped_early = False
    final_step = start_step

    def checkpoint_metadata() -> dict:
        return {
            "samples_seen": samples_seen,
            "best_total_loss": best_total_loss,
            "best_waypoint_loss": best_waypoint_loss,
            "best_latent_loss": best_latent_loss,
            "best_spatial_cosine": best_spatial_cosine,
            "best_validation_loss": best_validation_loss,
            "early_stopping_bad_evals": early_stopping_bad_evals,
            "best_checkpoint_path": best_checkpoint_path,
            "stopped_early": stopped_early,
        }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(start_step + 1, config.max_steps + 1):
        final_step = step
        step_started = time.perf_counter()
        step_sample_count = 0
        should_log = step % config.log_steps == 0 or step == start_step + 1
        metric_sums: dict[str, float] = {}
        for _ in range(config.gradient_accumulation_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                batch = next(data_iterator)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            step_sample_count += int(input_ids.shape[0])
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            pixel_values = _move_optional(batch.get("pixel_values"), device)
            image_grid_thw = _move_optional(batch.get("image_grid_thw"), device)
            pixel_values_videos = _move_optional(
                batch.get("pixel_values_videos"), device
            )
            video_grid_thw = _move_optional(batch.get("video_grid_thw"), device)
            ground_truth = batch["gt_waypoints"].to(device, non_blocking=True)
            vggt_images = batch["vggt_images"].to(device, non_blocking=True)
            vggt_view_mask = batch["vggt_view_mask"].to(
                device, non_blocking=True
            )
            planner_view_indices = batch["planner_view_indices"].to(
                device, non_blocking=True
            )

            with torch.no_grad(), _autocast_context(device):
                reference_latents = reference.generate_reasoning_latents(
                    input_ids,
                    pixel_values,
                    image_grid_thw,
                    attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                )
                geometry_features = vggt(vggt_images, vggt_view_mask)

            with _autocast_context(device):
                sf_latents, _, _, predicted_waypoints = student.generate_latents(
                    input_ids,
                    pixel_values,
                    image_grid_thw,
                    attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                )
                student_visual, student_visual_mask = (
                    student.get_mid_layer_visual_features(
                        input_ids,
                        pixel_values,
                        image_grid_thw,
                        attention_mask,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                        layer_idx=config.student_visual_layer,
                        return_mask=True,
                    )
                )

            latent = latent_reasoning_preservation_loss(
                sf_latents, reference_latents
            )
            waypoint = waypoint_loss(predicted_waypoints, ground_truth)
            spatial_forcing, spatial_cosine = spatial_alignment(
                student_visual,
                student_visual_mask,
                geometry_features,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=qwen_spatial_merge_size,
                planner_view_indices=planner_view_indices,
                vggt_patch_size=vggt.patch_size,
            )
            losses = combine_stage4_losses(
                latent=latent,
                waypoint=waypoint,
                spatial_forcing=spatial_forcing,
                spatial_cosine=spatial_cosine,
                alpha=config.alpha,
                beta=config.beta,
                gamma=config.gamma,
            )
            (losses.total / config.gradient_accumulation_steps).backward()

            for key, value in _detached_stage4_metrics(
                losses, predicted_waypoints, ground_truth, config
            ).items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value

        assert_spatial_tokens_frozen(student)
        group_grad_norms = {}
        if should_log:
            group_grad_norms = {
                name: _parameter_grad_norm(parameters)
                for name, parameters in grouped_parameters.items()
            }
        global_grad_norm = torch.nn.utils.clip_grad_norm_(
            student_parameters + list(spatial_alignment.parameters()),
            config.grad_clip,
        )
        global_grad_norm_value = float(global_grad_norm.detach().float().item())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step_time_seconds = time.perf_counter() - step_started
        samples_seen += step_sample_count

        if should_log:
            denom = config.gradient_accumulation_steps
            metrics = {key: value / denom for key, value in metric_sums.items()}
            current_lrs = {
                group["group_name"]: group["lr"] for group in optimizer.param_groups
            }
            logger.info(
                "step=%d total=%.6f latent=%.6f waypoint=%.6f sf=%.6f "
                "sf_cos=%.6f lr_q0_7=%.3e lr_q8_31=%.3e lr_wp=%.3e lr_sf=%.3e",
                step,
                metrics["loss/total"],
                metrics["loss/latent"],
                metrics["loss/waypoint"],
                metrics["loss/spatial_forcing"],
                metrics["spatial/cosine"],
                current_lrs["qwen_layers_0_7"],
                current_lrs["qwen_layers_8_31"],
                current_lrs["waypoint_head"],
                current_lrs["sf_projector"],
            )
            best_total_loss = min(best_total_loss, metrics["loss/total"])
            best_waypoint_loss = min(
                best_waypoint_loss, metrics["loss/waypoint"]
            )
            best_latent_loss = min(best_latent_loss, metrics["loss/latent"])
            best_spatial_cosine = max(
                best_spatial_cosine, metrics["spatial/cosine"]
            )
            if wandb_run is not None:
                wandb_metrics = {
                    **metrics,
                    "progress/optimizer_step": step,
                    "progress/fraction": step / config.max_steps,
                    "data/samples_seen": samples_seen,
                    "data/effective_batch_size": step_sample_count,
                    "performance/step_time_seconds": step_time_seconds,
                    "performance/samples_per_second": step_sample_count
                    / max(step_time_seconds, 1e-12),
                    "grad_norm/global_pre_clip": global_grad_norm_value,
                    "grad_norm/clip_threshold": config.grad_clip,
                    "grad_norm/was_clipped": float(
                        global_grad_norm_value > config.grad_clip
                    ),
                    **{
                        f"grad_norm/{name}": value
                        for name, value in group_grad_norms.items()
                    },
                    **{
                        f"learning_rate/{name}": value
                        for name, value in current_lrs.items()
                    },
                }
                if device.type == "cuda":
                    wandb_metrics.update(
                        {
                            "memory/allocated_gib": torch.cuda.memory_allocated(device)
                            / 1024**3,
                            "memory/reserved_gib": torch.cuda.memory_reserved(device)
                            / 1024**3,
                            "memory/peak_allocated_gib": torch.cuda.max_memory_allocated(
                                device
                            )
                            / 1024**3,
                            "memory/peak_reserved_gib": torch.cuda.max_memory_reserved(
                                device
                            )
                            / 1024**3,
                        }
                    )
                    torch.cuda.reset_peak_memory_stats(device)
                wandb_run.log(wandb_metrics)

                wandb_run.summary.update(
                    {
                        "best/loss_total": best_total_loss,
                        "best/loss_waypoint": best_waypoint_loss,
                        "best/loss_latent": best_latent_loss,
                        "best/spatial_cosine": best_spatial_cosine,
                    }
                )

        should_evaluate = config.evaluate and (
            step % config.eval_steps == 0 or step == config.max_steps
        )
        if should_evaluate:
            validation_metrics = _evaluate_stage4(
                validation_dataloader,
                reference=reference,
                student=student,
                vggt=vggt,
                spatial_alignment=spatial_alignment,
                device=device,
                config=config,
                qwen_spatial_merge_size=qwen_spatial_merge_size,
            )
            validation_loss = validation_metrics["loss/total"]
            (
                best_validation_loss,
                early_stopping_bad_evals,
                improved,
            ) = _update_early_stopping(
                validation_loss=validation_loss,
                best_validation_loss=best_validation_loss,
                bad_evaluations=early_stopping_bad_evals,
                min_delta=config.early_stopping_min_delta,
            )
            if improved:
                best_checkpoint_path = str(
                    Path(config.output_dir) / f"step_{step:06d}"
                )
                saved = save_stage4_checkpoint(
                    config.output_dir,
                    step,
                    student,
                    spatial_alignment,
                    optimizer,
                    scheduler,
                    config,
                    training_metadata=checkpoint_metadata(),
                )
                last_saved_step = step
                best_checkpoint_path = str(saved)
                pointer = _write_best_checkpoint_pointer(
                    config.output_dir,
                    checkpoint_path=best_checkpoint_path,
                    step=step,
                    validation_loss=validation_loss,
                )
                logger.info("New best validation checkpoint: %s (%s)", saved, pointer)
                if wandb_run is not None:
                    wandb_run.summary["checkpoint/latest_step"] = step
                    wandb_run.summary["checkpoint/latest_path"] = str(saved)

            logger.info(
                "validation step=%d total=%.6f latent=%.6f waypoint=%.6f "
                "sf=%.6f sf_cos=%.6f samples=%d improved=%s patience=%d/%d",
                step,
                validation_metrics["loss/total"],
                validation_metrics["loss/latent"],
                validation_metrics["loss/waypoint"],
                validation_metrics["loss/spatial_forcing"],
                validation_metrics["spatial/cosine"],
                int(validation_metrics["evaluation/samples"]),
                improved,
                early_stopping_bad_evals,
                config.early_stopping_patience,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "progress/optimizer_step": step,
                        **{
                            f"validation/{key}": value
                            for key, value in validation_metrics.items()
                        },
                        "early_stopping/best_validation_loss": best_validation_loss,
                        "early_stopping/bad_evaluations": early_stopping_bad_evals,
                        "early_stopping/patience": config.early_stopping_patience,
                        "early_stopping/improved": float(improved),
                    }
                )
                wandb_run.summary["best/validation_loss"] = best_validation_loss
                if best_checkpoint_path:
                    wandb_run.summary["best/checkpoint_path"] = best_checkpoint_path

            if (
                config.early_stopping
                and early_stopping_bad_evals >= config.early_stopping_patience
            ):
                stopped_early = True
                logger.info(
                    "Early stopping at step %d: validation loss did not improve "
                    "by at least %g for %d evaluations. Best=%.6f at %s",
                    step,
                    config.early_stopping_min_delta,
                    config.early_stopping_patience,
                    best_validation_loss,
                    best_checkpoint_path,
                )
                break

        if step % config.save_steps == 0 and last_saved_step != step:
            saved = save_stage4_checkpoint(
                config.output_dir,
                step,
                student,
                spatial_alignment,
                optimizer,
                scheduler,
                config,
                training_metadata=checkpoint_metadata(),
            )
            last_saved_step = step
            logger.info("Saved %s", saved)
            if wandb_run is not None:
                wandb_run.summary["checkpoint/latest_step"] = step
                wandb_run.summary["checkpoint/latest_path"] = str(saved)

    if last_saved_step != final_step:
        saved = save_stage4_checkpoint(
            config.output_dir,
            final_step,
            student,
            spatial_alignment,
            optimizer,
            scheduler,
            config,
            training_metadata=checkpoint_metadata(),
        )
        logger.info("Saved final checkpoint %s", saved)
        last_saved_step = final_step
        if wandb_run is not None:
            wandb_run.summary["checkpoint/latest_step"] = final_step
            wandb_run.summary["checkpoint/latest_path"] = str(saved)

    if wandb_run is not None:
        wandb_run.summary["training/final_step"] = final_step
        wandb_run.summary["training/samples_seen"] = samples_seen
        wandb_run.summary["training/completed"] = True
        wandb_run.summary["training/stopped_early"] = stopped_early
        wandb_run.summary["early_stopping/bad_evaluations"] = (
            early_stopping_bad_evals
        )
        wandb_run.finish()


def parse_args() -> Stage4Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student_checkpoint", default="shreethar/LatentStudent-ckpt-400"
    )
    parser.add_argument("--reference_checkpoint")
    parser.add_argument("--resume_from")
    parser.add_argument("--base_model_name")
    parser.add_argument("--processor_name")
    parser.add_argument("--hf_repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf_config", default=DEFAULT_HF_CONFIG)
    parser.add_argument(
        "--materialized_data_dir",
        help="Completed local output from materialize_molmoact.py (no dataset GETs)",
    )
    parser.add_argument(
        "--allow_incomplete_materialized",
        action="store_true",
        help=(
            "Explicitly train from validated completed Parquet shards left by an "
            "interrupted materialization"
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--data_partition",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--validation_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--output_dir", default="checkpoints/stage4")
    parser.add_argument("--vggt_checkpoint", default="facebook/VGGT-1B")
    parser.add_argument(
        "--sample_ratio", "--subset_ratio", dest="sample_ratio", type=float, default=0.1
    )
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--student_visual_layer", type=int, default=8)
    parser.add_argument("--vggt_layer", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--qwen_layers_0_7_lr", type=float, default=1e-5)
    parser.add_argument("--qwen_layers_8_31_lr", type=float, default=1e-6)
    parser.add_argument("--waypoint_head_lr", type=float, default=1e-5)
    parser.add_argument("--projector_lr", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=50,
        help="Maximum validation batches per evaluation",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        help="Validation batch size (defaults to --batch_size)",
    )
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="reasonflow-vla")
    parser.add_argument("--wandb_entity")
    parser.add_argument(
        "--wandb_run_name",
        "--wandb_run",
        default="stage4-spatial-forcing",
    )
    parser.add_argument("--wandb_run_id")
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline"),
        default="online",
    )
    parser.add_argument(
        "--wandb_tags",
        default="stage4,spatial-forcing,molmoact",
        help="Comma-separated W&B run tags",
    )
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--projector_normalization",
        choices=("batchnorm",),
        default="batchnorm",
    )
    parser.add_argument("--use_vggt_position_embedding", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["use_wandb"] = not values.pop("no_wandb")
    values["evaluate"] = not values.pop("no_eval")
    no_early_stopping = values.pop("no_early_stopping")
    values["early_stopping"] = values["evaluate"] and not no_early_stopping
    values["wandb_tags"] = tuple(
        tag.strip() for tag in values["wandb_tags"].split(",") if tag.strip()
    )
    return Stage4Config(**values)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    train(parse_args())
