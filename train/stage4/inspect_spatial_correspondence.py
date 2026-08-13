"""Inspect Qwen/VGGT spatial correspondence before Stage 4 training.

This script runs the real processor, latent-student checkpoint, dataset sample,
and frozen VGGT aggregator. It writes a JSON report that answers the questions
which shape checks alone cannot:

* What grid does Qwen declare before and after spatial merging?
* Does that grid predict the actual placeholder, visual-encoder, and layer-8
  token counts?
* What spatial and temporal grid does VGGT produce?
* Does Stage 4's current count-inferred resampling equal a metadata-driven
  ``(time, height, width)`` interpolation?
* If not, how large is the implied coordinate displacement?

Run this on the training machine and send the generated JSON file back for
review before starting a full Stage 4 run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.stage4.checkpointing import load_latent_student_checkpoint
from train.stage4.models.spatial_forcing import VGGTExtractor, _closest_grid
from train.stage4.stage4_dataloader import Stage4Dataset

logger = logging.getLogger("spatial-correspondence")


@dataclass(frozen=True)
class QwenGridSpec:
    source_thw: tuple[int, int, int]
    merged_thw: tuple[int, int, int]
    expected_tokens: int
    divisible_by_merge: bool


def derive_qwen_grid_specs(
    grid_thw: torch.Tensor | Sequence[Sequence[int]],
    spatial_merge_size: int,
) -> list[QwenGridSpec]:
    """Convert processor ``grid_thw`` rows to post-merge Qwen token grids."""
    rows = grid_thw.tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
    specs: list[QwenGridSpec] = []
    for row in rows:
        if len(row) != 3:
            raise ValueError(f"grid_thw row must contain T,H,W, got {row}")
        time, height, width = (int(value) for value in row)
        divisible = height % spatial_merge_size == 0 and width % spatial_merge_size == 0
        merged_h = height // spatial_merge_size
        merged_w = width // spatial_merge_size
        specs.append(
            QwenGridSpec(
                source_thw=(time, height, width),
                merged_thw=(time, merged_h, merged_w),
                expected_tokens=time * merged_h * merged_w,
                divisible_by_merge=divisible,
            )
        )
    return specs


def metadata_resample_vggt(
    features: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int, int],
) -> torch.Tensor:
    """Resample ``[views, patches, D]`` to Qwen's explicit ``[T,H,W]`` grid."""
    if features.ndim != 3:
        raise ValueError("VGGT features must be [views, patches, dim]")
    views, patch_count, feature_dim = features.shape
    source_h, source_w = source_grid
    if patch_count != source_h * source_w:
        raise ValueError("VGGT patch count does not match source_grid")
    target_t, target_h, target_w = target_grid
    volume = features.permute(2, 0, 1).reshape(
        1, feature_dim, views, source_h, source_w
    )
    resized = F.interpolate(
        volume.float(),
        size=(target_t, target_h, target_w),
        mode="trilinear",
        align_corners=True,
    )
    return resized.reshape(feature_dim, -1).transpose(0, 1)


def current_count_resample_vggt(
    features: torch.Tensor,
    source_grid: tuple[int, int],
    target_token_count: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reproduce the current Stage 4 count/aspect-ratio resampling path."""
    views, patch_count, feature_dim = features.shape
    source_h, source_w = source_grid
    if patch_count != source_h * source_w:
        raise ValueError("VGGT patch count does not match source_grid")
    per_view_count = max(1, round(target_token_count / views))
    target_h, target_w = _closest_grid(
        per_view_count, aspect_ratio=source_w / source_h
    )
    feature_map = features.permute(0, 2, 1).reshape(
        views, feature_dim, source_h, source_w
    )
    resized = F.interpolate(
        feature_map.float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=True,
    )
    flattened = resized.flatten(2).transpose(1, 2).reshape(-1, feature_dim)
    if flattened.shape[0] != target_token_count:
        flattened = F.interpolate(
            flattened.transpose(0, 1).unsqueeze(0),
            size=target_token_count,
            mode="linear",
            align_corners=False,
        ).squeeze(0).transpose(0, 1)
    return flattened, (target_h, target_w)


def make_coordinate_features(
    views: int,
    grid: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Create VGGT-shaped features containing normalized ``(time,y,x)``."""
    height, width = grid
    time_axis = torch.linspace(0.0, 1.0, views, device=device)
    y_axis = torch.linspace(0.0, 1.0, height, device=device)
    x_axis = torch.linspace(0.0, 1.0, width, device=device)
    time, y, x = torch.meshgrid(time_axis, y_axis, x_axis, indexing="ij")
    return torch.stack((time, y, x), dim=-1).reshape(views, height * width, 3)


def contiguous_runs(mask: torch.Tensor) -> list[dict[str, int]]:
    """Return inclusive-exclusive runs of True entries in a 1-D mask."""
    positions = mask.nonzero(as_tuple=False).flatten().tolist()
    if not positions:
        return []
    runs = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            runs.append({"start": start, "end": previous + 1, "length": previous + 1 - start})
            start = position
        previous = position
    runs.append({"start": start, "end": previous + 1, "length": previous + 1 - start})
    return runs


def tensor_summary(tensor: Optional[torch.Tensor]) -> Optional[dict[str, Any]]:
    if tensor is None:
        return None
    floating = tensor.detach().float()
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }
    if floating.numel():
        summary.update(
            {
                "min": float(floating.min().item()),
                "max": float(floating.max().item()),
                "mean": float(floating.mean().item()),
                "std": float(floating.std().item()) if floating.numel() > 1 else 0.0,
            }
        )
    if tensor.ndim >= 2 and tensor.shape[-1] > 1 and floating.numel():
        summary["mean_l2_norm_last_dim"] = float(
            floating.norm(dim=-1).mean().item()
        )
    return summary


def safe_attribute(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
    return None


def resolve_processor_geometry(processor, student) -> dict[str, Any]:
    image_processor = processor.image_processor
    visual_config = getattr(
        getattr(student.vlm, "config", None), "vision_config", None
    )
    if visual_config is None:
        text_config = getattr(student.vlm, "config", None)
        visual_config = getattr(text_config, "vision_config", None)

    merge_size = safe_attribute(
        image_processor, ("merge_size", "spatial_merge_size")
    )
    if merge_size is None and visual_config is not None:
        merge_size = safe_attribute(visual_config, ("spatial_merge_size", "merge_size"))
    merge_size = int(merge_size or 1)

    return {
        "patch_size": safe_attribute(image_processor, ("patch_size",)),
        "spatial_merge_size": merge_size,
        "temporal_patch_size": safe_attribute(
            image_processor, ("temporal_patch_size",)
        ),
        "min_pixels": safe_attribute(image_processor, ("min_pixels",)),
        "max_pixels": safe_attribute(image_processor, ("max_pixels",)),
        "image_processor_class": type(image_processor).__name__,
        "visual_config_class": type(visual_config).__name__ if visual_config is not None else None,
    }


def load_hf_split(repo: str, split: str):
    from datasets import load_dataset

    try:
        files = {split: f"hf://datasets/{repo}/data/{split}-*.parquet"}
        return load_dataset("parquet", data_files=files, split=split)
    except Exception:
        logger.warning("Explicit Parquet load failed; using load_dataset(%s)", repo)
        return load_dataset(repo, split=split)


def modality_details(
    sample: dict[str, Any],
    input_ids: torch.Tensor,
    student,
    merge_size: int,
) -> tuple[str, list[QwenGridSpec], int, list[dict[str, int]]]:
    image_id = getattr(student.vlm.config, "image_token_id", student._image_token_id)
    video_id = getattr(student.vlm.config, "video_token_id", 248057)
    image_count = int((input_ids == image_id).sum().item()) if image_id is not None else 0
    video_count = int((input_ids == video_id).sum().item()) if video_id is not None else 0

    if image_count and video_count:
        raise ValueError("Mixed image/video samples are not supported by this diagnostic")
    if image_count:
        grid = sample.get("image_grid_thw")
        specs = derive_qwen_grid_specs(grid, merge_size)
        return "image", specs, image_count, contiguous_runs(input_ids == image_id)
    if video_count:
        grid = sample.get("video_grid_thw")
        specs = derive_qwen_grid_specs(grid, merge_size)
        return "video", specs, video_count, contiguous_runs(input_ids == video_id)
    raise ValueError("No Qwen image/video placeholder tokens were found")


def inspect_sample(
    sample_index: int,
    dataset: Stage4Dataset,
    processor,
    student,
    vggt: VGGTExtractor,
    student_layer: int,
    device: torch.device,
) -> dict[str, Any]:
    sample = dataset[sample_index]
    original_frame_sizes = [
        list(frame.size) for frame in dataset.samples[sample_index]["frames"]
    ]
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
    pixel_values = sample.get("pixel_values")
    image_grid = sample.get("image_grid_thw")
    pixel_values_videos = sample.get("pixel_values_videos")
    video_grid = sample.get("video_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    if image_grid is not None:
        image_grid = image_grid.to(device)
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.to(device)
    if video_grid is not None:
        video_grid = video_grid.to(device)

    geometry = resolve_processor_geometry(processor, student)
    modality, grid_specs, placeholder_count, placeholder_runs = modality_details(
        sample,
        input_ids[0],
        student,
        geometry["spatial_merge_size"],
    )

    with torch.no_grad():
        if modality == "image":
            encoder_output = student._visual_encoder(
                pixel_values, grid_thw=image_grid
            )
        else:
            encoder_output = student._visual_encoder(
                pixel_values_videos, grid_thw=video_grid
            )
        if not isinstance(encoder_output, torch.Tensor):
            encoder_output = getattr(
                encoder_output, "pooler_output", encoder_output[0]
            )

        layer_features, layer_mask = student.get_mid_layer_visual_features(
            input_ids,
            pixel_values,
            image_grid,
            attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid,
            layer_idx=student_layer,
            return_mask=True,
        )
        vggt_images = sample["vggt_images"].unsqueeze(0).to(device)
        vggt_features = vggt(vggt_images).features[0]

    expected_tokens = sum(spec.expected_tokens for spec in grid_specs)
    actual_layer_tokens = int(layer_mask.sum().item())
    actual_encoder_tokens = int(encoder_output.shape[0])
    vggt_views = int(vggt_features.shape[0])

    comparison: dict[str, Any] = {
        "available": False,
        "reason": None,
    }
    vggt_grid = (
        vggt_images.shape[-2] // vggt.patch_size,
        vggt_images.shape[-1] // vggt.patch_size,
    )
    if len(grid_specs) == 1 and expected_tokens == actual_layer_tokens:
        target_grid = grid_specs[0].merged_thw
        metadata_target = metadata_resample_vggt(
            vggt_features, vggt_grid, target_grid
        )
        current_target, inferred_hw = current_count_resample_vggt(
            vggt_features,
            vggt_grid,
            actual_layer_tokens,
        )
        feature_cosine = F.cosine_similarity(
            current_target.float(), metadata_target.float(), dim=-1
        )

        coordinates = make_coordinate_features(
            vggt_views,
            vggt_grid,
            device,
        )
        metadata_coordinates = metadata_resample_vggt(
            coordinates,
            vggt_grid,
            target_grid,
        )
        current_coordinates, _ = current_count_resample_vggt(
            coordinates,
            vggt_grid,
            actual_layer_tokens,
        )
        displacement = (current_coordinates - metadata_coordinates).norm(dim=-1)
        comparison = {
            "available": True,
            "qwen_metadata_target_thw": list(target_grid),
            "current_inferred_per_view_hw": list(inferred_hw),
            "metadata_resampled_shape": list(metadata_target.shape),
            "current_resampled_shape": list(current_target.shape),
            "mean_feature_cosine_current_vs_metadata": float(feature_cosine.mean().item()),
            "min_feature_cosine_current_vs_metadata": float(feature_cosine.min().item()),
            "mean_normalized_coordinate_displacement": float(displacement.mean().item()),
            "max_normalized_coordinate_displacement": float(displacement.max().item()),
            "direct_temporal_correspondence": vggt_views == target_grid[0],
            "vggt_views": vggt_views,
            "qwen_temporal_positions": target_grid[0],
        }
    elif len(grid_specs) != 1:
        comparison["reason"] = (
            "Multiple Qwen grid rows require per-item/view ownership metadata; "
            "the current Stage 4 dataset normally emits one image or one video."
        )
    else:
        comparison["reason"] = (
            "Qwen grid metadata token count does not match the actual layer-8 "
            "visual-token count, so equal-shape resampling cannot be compared."
        )

    all_grids_divisible = all(spec.divisible_by_merge for spec in grid_specs)
    count_chain_matches = (
        expected_tokens
        == placeholder_count
        == actual_encoder_tokens
        == actual_layer_tokens
    )
    exact_grid_match = False
    if comparison["available"]:
        exact_grid_match = (
            comparison["mean_normalized_coordinate_displacement"] < 1e-6
            and comparison["max_normalized_coordinate_displacement"] < 1e-5
        )
    spatial_aspect_matches = False
    if len(grid_specs) == 1:
        _, qwen_h, qwen_w = grid_specs[0].merged_thw
        spatial_aspect_matches = math.isclose(
            qwen_w / qwen_h,
            vggt_grid[1] / vggt_grid[0],
            rel_tol=1e-6,
            abs_tol=1e-6,
        )

    checks = {
        "all_qwen_grids_divisible_by_merge": all_grids_divisible,
        "qwen_expected_equals_placeholder_count": expected_tokens == placeholder_count,
        "placeholder_equals_visual_encoder_count": placeholder_count == actual_encoder_tokens,
        "visual_encoder_equals_layer_visual_count": actual_encoder_tokens == actual_layer_tokens,
        "entire_qwen_count_chain_matches": count_chain_matches,
        "vggt_patch_count_matches_grid": int(vggt_features.shape[1])
        == (vggt_images.shape[-2] // vggt.patch_size)
        * (vggt_images.shape[-1] // vggt.patch_size),
        "qwen_vggt_spatial_aspect_ratio_matches": spatial_aspect_matches,
        "current_resampling_matches_metadata_grid": exact_grid_match,
    }
    checks["safe_to_use_current_alignment"] = all(checks.values())

    return {
        "sample_index": sample_index,
        "sample_id": str(sample.get("sample_id")),
        "dataset": sample.get("dataset"),
        "modality": modality,
        "image_preprocessing": {
            "original_frame_sizes_wh": original_frame_sizes,
            "qwen_dataset_resize_hw": [448, 448],
            "vggt_dataset_resize_hw": [
                int(vggt_images.shape[-2]),
                int(vggt_images.shape[-1]),
            ],
            "both_paths_use_full_frame_square_resize": True,
        },
        "processor_geometry": geometry,
        "qwen": {
            "grid_specs": [asdict(spec) for spec in grid_specs],
            "expected_post_merge_tokens": expected_tokens,
            "placeholder_token_count": placeholder_count,
            "placeholder_runs": placeholder_runs,
            "visual_encoder_output": tensor_summary(encoder_output),
            "layer_index": student_layer,
            "layer_visual_features": tensor_summary(
                layer_features[0, layer_mask[0]]
            ),
            "qwen_processor_pixel_values": tensor_summary(
                pixel_values if modality == "image" else pixel_values_videos
            ),
        },
        "vggt": {
            "layer_index": vggt.layer_index,
            "input_images": tensor_summary(vggt_images),
            "patch_size": vggt.patch_size,
            "patch_grid_hw": [
                vggt_images.shape[-2] // vggt.patch_size,
                vggt_images.shape[-1] // vggt.patch_size,
            ],
            "features": tensor_summary(vggt_features),
        },
        "resampling_comparison": comparison,
        "checks": checks,
        "verdict": (
            "CURRENT_ALIGNMENT_GEOMETRY_MATCHES"
            if checks["safe_to_use_current_alignment"]
            else "DO_NOT_TRAIN_WITH_CURRENT_ALIGNMENT"
        ),
    }


def parse_indices(value: str) -> list[int]:
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("At least one sample index is required")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--base_model_name")
    parser.add_argument("--processor_name")
    parser.add_argument("--hf_repo", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample_indices", type=parse_indices, default=[0])
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--student_layer", type=int, default=8)
    parser.add_argument("--vggt_layer", type=int, default=8)
    parser.add_argument("--vggt_checkpoint", default="facebook/VGGT-1B")
    parser.add_argument(
        "--output", default="spatial_correspondence_report.json"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA is unavailable; the 4B+VGGT diagnostic will be slow")

    processor_name = args.processor_name or args.base_model_name
    if processor_name is None:
        if Path(args.student_checkpoint).exists():
            parser.error(
                "Local checkpoints require --base_model_name or --processor_name"
            )
        processor_name = args.student_checkpoint

    from transformers import AutoProcessor, AutoTokenizer

    processor = AutoProcessor.from_pretrained(
        processor_name, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        processor_name, trust_remote_code=True
    )
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    if end_think_id is None or end_think_id == tokenizer.unk_token_id:
        end_think_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]

    logger.info("Loading latent student checkpoint")
    student = load_latent_student_checkpoint(
        checkpoint=args.student_checkpoint,
        base_model_name=args.base_model_name,
        end_think_token_id=int(end_think_id),
        trainable=False,
    ).to(device)
    student.eval()

    logger.info("Loading VGGT layer %d", args.vggt_layer)
    vggt = VGGTExtractor(
        checkpoint=args.vggt_checkpoint, layer_index=args.vggt_layer
    ).to(
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    vggt.eval()

    logger.info("Loading dataset %s[%s]", args.hf_repo, args.split)
    hf_split = load_hf_split(args.hf_repo, args.split)
    dataset = Stage4Dataset(
        hf_split, processor=processor, max_length=args.max_seq_len
    )
    for index in args.sample_indices:
        if not 0 <= index < len(dataset):
            parser.error(f"sample index {index} outside dataset of size {len(dataset)}")

    reports = []
    for index in args.sample_indices:
        logger.info("Inspecting sample %d", index)
        reports.append(
            inspect_sample(
                sample_index=index,
                dataset=dataset,
                processor=processor,
                student=student,
                vggt=vggt,
                student_layer=args.student_layer,
                device=device,
            )
        )

    report = {
        "student_checkpoint": args.student_checkpoint,
        "base_model_name": args.base_model_name,
        "processor_name": processor_name,
        "vggt_checkpoint": args.vggt_checkpoint,
        "device": str(device),
        "samples": reports,
        "all_samples_safe_to_use_current_alignment": all(
            sample["checks"]["safe_to_use_current_alignment"]
            for sample in reports
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {output_path.resolve()}")
    if not report["all_samples_safe_to_use_current_alignment"]:
        print("VERDICT: DO NOT START STAGE 4 TRAINING YET.")
    else:
        print("VERDICT: Current alignment geometry matched all inspected samples.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
