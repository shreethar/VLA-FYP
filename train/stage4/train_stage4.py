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
import logging
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
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
    output_dir: str = "checkpoints/stage4"
    reference_checkpoint: Optional[str] = None
    resume_from: Optional[str] = None
    base_model_name: Optional[str] = None
    processor_name: Optional[str] = None
    vggt_checkpoint: str = "facebook/VGGT-1B"
    split: str = "train"
    sample_ratio: float = 0.1
    max_seq_len: int = 1024
    batch_size: int = 1
    num_workers: int = 2
    M: int = 6
    K: int = 5
    student_visual_layer: int = 8
    vggt_layer: int = 8
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    student_lr: float = 1e-5
    projector_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    gradient_accumulation_steps: int = 1
    grad_clip: float = 1.0
    save_steps: int = 500
    log_steps: int = 10
    seed: int = 42
    use_input_norm: bool = False
    use_vggt_position_embedding: bool = False


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _move_optional(value, device: torch.device):
    return value.to(device, non_blocking=True) if value is not None else None


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


def _validate_config(config: Stage4Config) -> None:
    if config.M != 6:
        raise ValueError("This Stage 4 recipe requires the checkpoint's six latent slots")
    if config.K != 5:
        raise ValueError("MolmoAct Stage 4 requires exactly five spatial slots")
    if not 0.0 < config.sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0,1]")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")


def train(config: Stage4Config, dataloader=None) -> None:
    _validate_config(config)
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
        use_input_norm=config.use_input_norm,
        use_vggt_position_embedding=config.use_vggt_position_embedding,
    ).to(device)
    qwen_spatial_merge_size = _resolve_qwen_spatial_merge_size(student)

    student_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    if not student_parameters:
        raise RuntimeError("The trainable student has no trainable parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": student_parameters, "lr": config.student_lr},
            {
                "params": spatial_alignment.parameters(),
                "lr": config.projector_lr,
            },
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer, config.warmup_steps, config.max_steps
    )

    start_step = 0
    if config.resume_from:
        start_step = restore_stage4_training_state(
            config.resume_from,
            spatial_alignment,
            optimizer,
            scheduler,
        )
        logger.info("Resumed Stage 4 optimizer/projector at step %d", start_step)

    if dataloader is None:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            processor_name, trust_remote_code=True
        )
        dataloader = build_stage4_dataloader(
            processor=processor,
            hf_repo=config.hf_repo,
            hf_config=config.hf_config,
            split=config.split,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            max_length=config.max_seq_len,
            sample_ratio=config.sample_ratio,
            seed=config.seed,
        )

    logger.info(
        "Trainable parameters: student=%s projector=%s; spatial slots frozen=%s",
        f"{sum(p.numel() for p in student_parameters):,}",
        f"{sum(p.numel() for p in spatial_alignment.parameters()):,}",
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
    logger.info(
        "Dataset: %s[%s], valid-row sample ratio=%g; Qwen spatial merge=%d",
        config.hf_repo,
        config.hf_config,
        config.sample_ratio,
        qwen_spatial_merge_size,
    )

    reference.eval()
    vggt.eval()
    student.train()
    spatial_alignment.train()
    data_iterator = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    last_saved_step = start_step

    for step in range(start_step + 1, config.max_steps + 1):
        metric_sums: dict[str, float] = {}
        for _ in range(config.gradient_accumulation_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                batch = next(data_iterator)

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

            for key, value in losses.detached_metrics().items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value

        assert_spatial_tokens_frozen(student)
        torch.nn.utils.clip_grad_norm_(
            student_parameters + list(spatial_alignment.parameters()),
            config.grad_clip,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % config.log_steps == 0 or step == start_step + 1:
            denom = config.gradient_accumulation_steps
            metrics = {key: value / denom for key, value in metric_sums.items()}
            logger.info(
                "step=%d total=%.6f latent=%.6f waypoint=%.6f sf=%.6f "
                "sf_cos=%.6f lr=%.3e",
                step,
                metrics["loss/total"],
                metrics["loss/latent"],
                metrics["loss/waypoint"],
                metrics["loss/spatial_forcing"],
                metrics["spatial/cosine"],
                optimizer.param_groups[0]["lr"],
            )

        if step % config.save_steps == 0:
            saved = save_stage4_checkpoint(
                config.output_dir,
                step,
                student,
                spatial_alignment,
                optimizer,
                scheduler,
                config,
            )
            last_saved_step = step
            logger.info("Saved %s", saved)

    if last_saved_step != config.max_steps:
        saved = save_stage4_checkpoint(
            config.output_dir,
            config.max_steps,
            student,
            spatial_alignment,
            optimizer,
            scheduler,
            config,
        )
        logger.info("Saved final checkpoint %s", saved)


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
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="checkpoints/stage4")
    parser.add_argument("--vggt_checkpoint", default="facebook/VGGT-1B")
    parser.add_argument(
        "--sample_ratio", "--subset_ratio", dest="sample_ratio", type=float, default=0.1
    )
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--student_visual_layer", type=int, default=8)
    parser.add_argument("--vggt_layer", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--student_lr", type=float, default=1e-5)
    parser.add_argument("--projector_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_input_norm", action="store_true")
    parser.add_argument("--use_vggt_position_embedding", action="store_true")
    args = parser.parse_args()
    return Stage4Config(**vars(args))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    train(parse_args())
