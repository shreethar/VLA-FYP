#!/usr/bin/env python3
"""Compare Stage 1, textual teacher, B2, and Spatial-Forcing students.

This is the Stage 4 counterpart of ``train/stage2/evaluate_all.py``. It keeps
the same dataset and split, evaluates 10,000 valid trajectory rows by default,
and samples 50 of those evaluated rows for visualization. The former
checkpoint-619 column is replaced by the merged Spatial Forcing student from
Hugging Face.

Models are evaluated sequentially by default. The original evaluator launched
four 4B/5B models on the same CUDA device concurrently, which can easily OOM
and does not change the evaluation protocol.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.stage4.checkpointing import load_latent_student_checkpoint


DEFAULT_DATASET = "shreethar/FYP-Stage2-dataset"
DEFAULT_PROCESSOR = "shreethar/stage1_unsloth"
DEFAULT_STAGE1 = "shreethar/stage1_unsloth"
DEFAULT_TEACHER = "shreethar/stage2_teacher"
DEFAULT_LATENT_400 = "shreethar/LatentStudent-ckpt-400"
DEFAULT_SPATIAL_FORCING = "shreethar/Latent-Student-Spatial-Forcing"
COORDINATE_SCALE = 1000.0
EXPECTED_WAYPOINTS = 5


def parse_trajectory(text: str) -> Optional[list[list[float]]]:
    """Extract the first coordinate list from generated or reference text."""
    match = re.search(r"\[\[.*?\]\]", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        value = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, (list, tuple)):
        return None

    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        points.append([x, y])
    return points


def _valid_pair(gt, pred) -> bool:
    return (
        gt is not None
        and pred is not None
        and len(gt) == EXPECTED_WAYPOINTS
        and len(pred) == EXPECTED_WAYPOINTS
    )


def calc_waypoint_loss(gt, pred) -> float:
    """Stage 4 waypoint loss: mean_i ||pred_i - gt_i||^2 in [0,1]."""
    if not _valid_pair(gt, pred):
        return float("nan")
    total = 0.0
    for ground_truth, prediction in zip(gt, pred):
        dx = (float(prediction[0]) - float(ground_truth[0])) / COORDINATE_SCALE
        dy = (float(prediction[1]) - float(ground_truth[1])) / COORDINATE_SCALE
        total += dx * dx + dy * dy
    return total / EXPECTED_WAYPOINTS


def calc_l2_distance(gt, pred) -> float:
    """Mean pointwise Euclidean distance in the legacy 0-1000 coordinates."""
    if not _valid_pair(gt, pred):
        return float("nan")
    total = 0.0
    for ground_truth, prediction in zip(gt, pred):
        dx = float(ground_truth[0]) - float(prediction[0])
        dy = float(ground_truth[1]) - float(prediction[1])
        total += math.sqrt(dx * dx + dy * dy)
    return total / EXPECTED_WAYPOINTS


def calc_dtw(gt, pred) -> float:
    """Legacy dynamic-time-warping cost in the 0-1000 coordinates."""
    if not _valid_pair(gt, pred):
        return float("nan")
    n, m = len(gt), len(pred)
    matrix = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    matrix[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dx = float(gt[i - 1][0]) - float(pred[j - 1][0])
            dy = float(gt[i - 1][1]) - float(pred[j - 1][1])
            cost = math.sqrt(dx * dx + dy * dy)
            matrix[i][j] = cost + min(
                matrix[i - 1][j],
                matrix[i][j - 1],
                matrix[i - 1][j - 1],
            )
    return matrix[n][m]


def summarize_predictions(ground_truth, predictions) -> dict[str, Any]:
    per_sample = []
    for gt, pred in zip(ground_truth, predictions):
        waypoint = calc_waypoint_loss(gt, pred)
        l2_distance = calc_l2_distance(gt, pred)
        dtw = calc_dtw(gt, pred)
        per_sample.append(
            {
                "waypoint_loss": None if math.isnan(waypoint) else waypoint,
                "l2_distance_0_1000": None
                if math.isnan(l2_distance)
                else l2_distance,
                "dtw_0_1000": None if math.isnan(dtw) else dtw,
            }
        )

    valid = [item for item in per_sample if item["waypoint_loss"] is not None]
    if not valid:
        aggregate = {
            "valid_samples": 0,
            "failed_samples": len(per_sample),
            "mean_waypoint_loss": None,
            "median_waypoint_loss": None,
            "mean_l2_distance_0_1000": None,
            "mean_dtw_0_1000": None,
        }
    else:
        waypoint_losses = [item["waypoint_loss"] for item in valid]
        aggregate = {
            "valid_samples": len(valid),
            "failed_samples": len(per_sample) - len(valid),
            "mean_waypoint_loss": float(np.mean(waypoint_losses)),
            "median_waypoint_loss": float(np.median(waypoint_losses)),
            "mean_l2_distance_0_1000": float(
                np.mean([item["l2_distance_0_1000"] for item in valid])
            ),
            "mean_dtw_0_1000": float(
                np.mean([item["dtw_0_1000"] for item in valid])
            ),
        }
    return {"aggregate": aggregate, "per_sample": per_sample}


def draw_trajectory(image: Image.Image, trajectory) -> Image.Image:
    if trajectory is None:
        return image
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    width, height = drawn.size
    points = [
        (
            int(float(x) / COORDINATE_SCALE * width),
            int(float(y) / COORDINATE_SCALE * height),
        )
        for x, y in trajectory
    ]
    if len(points) > 1:
        draw.line(points, fill="red", width=3)
    for x, y in points:
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill="blue")
    return drawn


def draw_instruction_image(
    instruction: str, size: tuple[int, int] = (448, 448)
) -> Image.Image:
    image = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(image)
    wrapped = textwrap.fill(instruction, width=35)
    draw.text((20, size[1] // 2 - 40), wrapped, fill="black")
    return image


def _message_text(processor, sample, *, enable_thinking: bool) -> str:
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample["human"]},
            ],
        }
    ]
    return processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _model_device_map(device: str):
    return {"": device}


def process_text_model(
    *,
    model_name: str,
    processor_name: str,
    samples,
    enable_thinking: bool,
    description: str,
    device: str,
    max_new_tokens: int,
):
    processor = AutoProcessor.from_pretrained(
        processor_name, trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=_model_device_map(device),
        trust_remote_code=True,
    )
    model.eval()
    results, timings = [], []
    for sample in tqdm(samples, desc=description):
        prompt = _message_text(
            processor, sample, enable_thinking=enable_thinking
        )
        inputs = processor(
            text=[prompt], images=[sample["frames"][0]], return_tensors="pt"
        ).to(device)
        _synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        _synchronize(device)
        timings.append(time.perf_counter() - started)
        generated = processor.tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :]
        )
        results.append(parse_trajectory(generated))
    return results, timings


def process_latent_model(
    *,
    model_name: str,
    processor_name: str,
    samples,
    end_think_token_id: int,
    description: str,
    device: str,
):
    processor = AutoProcessor.from_pretrained(
        processor_name, trust_remote_code=True
    )
    student = load_latent_student_checkpoint(
        checkpoint=model_name,
        end_think_token_id=end_think_token_id,
        trainable=False,
        M=6,
        K=5,
    ).to(device)
    student.eval()
    results, timings = [], []
    for sample in tqdm(samples, desc=description):
        prompt = _message_text(processor, sample, enable_thinking=True)
        inputs = processor(
            text=[prompt], images=[sample["frames"][0]], return_tensors="pt"
        ).to(device)
        _synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            _, _, _, waypoints = student.generate_latents(
                input_ids=inputs.input_ids,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                attention_mask=inputs.attention_mask,
                pixel_values_videos=inputs.get("pixel_values_videos"),
                video_grid_thw=inputs.get("video_grid_thw"),
            )
        _synchronize(device)
        timings.append(time.perf_counter() - started)
        results.append(
            waypoints[0].detach().float().cpu().mul(COORDINATE_SCALE).tolist()
        )
    return results, timings


def _release_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_instruction(human_prompt: str) -> str:
    match = re.search(r"Task:\s*(.*?)\s*What is", human_prompt, re.DOTALL)
    return match.group(1).strip() if match else human_prompt


def _select_trajectory_rows(dataset, evaluation_rows: int):
    """Select the first N rows whose assistant answer contains five waypoints."""
    if "assistant" not in dataset.column_names:
        raise ValueError("Evaluation dataset must contain an 'assistant' column")
    assistant_rows = dataset.select_columns(["assistant"])
    selected_indices = []
    for row_index, row in enumerate(
        tqdm(assistant_rows, desc="Selecting trajectory rows")
    ):
        trajectory = parse_trajectory(row["assistant"])
        if trajectory is None or len(trajectory) != EXPECTED_WAYPOINTS:
            continue
        selected_indices.append(row_index)
        if len(selected_indices) == evaluation_rows:
            break
    if len(selected_indices) < evaluation_rows:
        raise ValueError(
            f"Requested {evaluation_rows} valid trajectory rows, but found only "
            f"{len(selected_indices)} in {len(dataset)} dataset rows"
        )
    return dataset.select(selected_indices), selected_indices


def _generate_plots(samples, results, metrics, output_dir: Path) -> None:
    column_names = [
        "Instruction",
        "Ground Truth",
        "Stage 1",
        "Teacher",
        "Latent 400",
        "Spatial Forcing",
    ]
    column_keys = [
        "instruction",
        "ground_truth",
        "stage1",
        "teacher",
        "latent400",
        "spatial_forcing",
    ]
    rows_per_page = 10
    pages = math.ceil(len(samples) / rows_per_page)
    for page in range(pages):
        start = page * rows_per_page
        rows = min(rows_per_page, len(samples) - start)
        figure, axes = plt.subplots(rows, len(column_keys), figsize=(26, 4 * rows))
        axes = np.asarray(axes).reshape(rows, len(column_keys))
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        for row in range(rows):
            index = start + row
            image = samples[index]["frames"][0]
            for column, key in enumerate(column_keys):
                axis = axes[row, column]
                if key == "instruction":
                    drawn = draw_instruction_image(
                        results["instruction"][index], size=image.size
                    )
                else:
                    trajectory = results[key][index]
                    drawn = draw_trajectory(image, trajectory)
                    if key not in ("ground_truth",):
                        sample_metrics = metrics[key]["per_sample"][index]
                        waypoint = sample_metrics["waypoint_loss"]
                        if waypoint is None:
                            score = "FAILED TO GENERATE 5 WAYPOINTS"
                        else:
                            score = (
                                f"WP: {waypoint:.5f}\n"
                                f"L2: {sample_metrics['l2_distance_0_1000']:.1f} | "
                                f"DTW: {sample_metrics['dtw_0_1000']:.1f}"
                            )
                        axis.text(
                            0.5,
                            -0.05,
                            score,
                            transform=axis.transAxes,
                            ha="center",
                            va="top",
                            fontsize=11,
                            color="black",
                            bbox={
                                "facecolor": "white",
                                "alpha": 0.8,
                                "edgecolor": "none",
                            },
                        )
                axis.imshow(drawn)
                axis.axis("off")
                if row == 0:
                    axis.set_title(column_names[column], fontsize=18, pad=15)
        figure.tight_layout()
        save_path = output_dir / f"evaluation_grid_{page + 1}.png"
        figure.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(figure)
        print(f"Saved {save_path}")


def _print_summary(metrics, timings) -> None:
    print("\nAggregate evaluation metrics")
    print(
        "model                 valid  failed  waypoint_loss  "
        "L2(0-1000)  DTW(0-1000)  seconds/sample"
    )
    for key in ("stage1", "teacher", "latent400", "spatial_forcing"):
        aggregate = metrics[key]["aggregate"]
        waypoint = aggregate["mean_waypoint_loss"]
        l2_distance = aggregate["mean_l2_distance_0_1000"]
        dtw = aggregate["mean_dtw_0_1000"]
        average_seconds = float(np.mean(timings[key]))
        waypoint_text = "N/A" if waypoint is None else f"{waypoint:.6f}"
        l2_text = "N/A" if l2_distance is None else f"{l2_distance:.2f}"
        dtw_text = "N/A" if dtw is None else f"{dtw:.2f}"
        print(
            f"{key:<22} {aggregate['valid_samples']:>5}  "
            f"{aggregate['failed_samples']:>6}  {waypoint_text:>13}  "
            f"{l2_text:>10}  {dtw_text:>11}  {average_seconds:>14.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--evaluation_rows", type=int, default=10000)
    parser.add_argument("--num_visualizations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache_dir",
        help="Optional Hugging Face dataset cache directory.",
    )
    parser.add_argument("--processor", default=DEFAULT_PROCESSOR)
    parser.add_argument("--stage1_model", default=DEFAULT_STAGE1)
    parser.add_argument("--teacher_model", default=DEFAULT_TEACHER)
    parser.add_argument("--latent400_model", default=DEFAULT_LATENT_400)
    parser.add_argument(
        "--spatial_forcing_model", default=DEFAULT_SPATIAL_FORCING
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", default="evaluation_stage4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluation_rows <= 0:
        raise ValueError("--evaluation_rows must be positive")
    if not 0 < args.num_visualizations <= args.evaluation_rows:
        raise ValueError(
            "--num_visualizations must be positive and no greater than "
            "--evaluation_rows"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}[{args.split}]...")
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    samples, selected_dataset_indices = _select_trajectory_rows(
        dataset, args.evaluation_rows
    )
    random_state = np.random.RandomState(args.seed)
    visualization_positions = random_state.choice(
        args.evaluation_rows,
        args.num_visualizations,
        replace=False,
    )

    processor = AutoProcessor.from_pretrained(
        args.processor, trust_remote_code=True
    )
    end_think_token_id = processor.tokenizer.convert_tokens_to_ids("</think>")

    text_rows = samples.select_columns(["human", "assistant"])
    results: dict[str, Any] = {
        "instruction": [
            _extract_instruction(sample["human"]) for sample in text_rows
        ],
        "ground_truth": [
            parse_trajectory(sample["assistant"]) for sample in text_rows
        ],
    }
    timings: dict[str, list[float]] = {}

    print("Evaluating Stage 1...")
    results["stage1"], timings["stage1"] = process_text_model(
        model_name=args.stage1_model,
        processor_name=args.processor,
        samples=samples,
        enable_thinking=False,
        description="Stage 1",
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    _release_accelerator_memory()

    print("Evaluating textual-thinking teacher...")
    results["teacher"], timings["teacher"] = process_text_model(
        model_name=args.teacher_model,
        processor_name=args.processor,
        samples=samples,
        enable_thinking=True,
        description="Teacher",
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    _release_accelerator_memory()

    print("Evaluating Latent Student checkpoint 400...")
    results["latent400"], timings["latent400"] = process_latent_model(
        model_name=args.latent400_model,
        processor_name=args.processor,
        samples=samples,
        end_think_token_id=end_think_token_id,
        description="Latent 400",
        device=args.device,
    )
    _release_accelerator_memory()

    print("Evaluating Latent Student Spatial Forcing...")
    results["spatial_forcing"], timings["spatial_forcing"] = process_latent_model(
        model_name=args.spatial_forcing_model,
        processor_name=args.processor,
        samples=samples,
        end_think_token_id=end_think_token_id,
        description="Spatial Forcing",
        device=args.device,
    )
    _release_accelerator_memory()

    metrics = {
        key: summarize_predictions(results["ground_truth"], results[key])
        for key in ("stage1", "teacher", "latent400", "spatial_forcing")
    }
    _print_summary(metrics, timings)

    report = {
        "configuration": {
            "dataset": args.dataset,
            "split": args.split,
            "evaluation_rows": args.evaluation_rows,
            "num_visualizations": args.num_visualizations,
            "seed": args.seed,
            "selected_dataset_indices": selected_dataset_indices,
            "visualization_positions": [
                int(position) for position in visualization_positions
            ],
            "visualization_dataset_indices": [
                selected_dataset_indices[int(position)]
                for position in visualization_positions
            ],
            "coordinate_scale": COORDINATE_SCALE,
            "models": {
                "stage1": args.stage1_model,
                "teacher": args.teacher_model,
                "latent400": args.latent400_model,
                "spatial_forcing": args.spatial_forcing_model,
            },
        },
        "metrics": metrics,
        "timings_seconds": timings,
        "results": results,
    }
    report_path = output_dir / "evaluation_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {report_path}")

    print("Generating comparison plots...")
    plot_samples = [samples[int(position)] for position in visualization_positions]
    plot_results = {
        key: [values[int(position)] for position in visualization_positions]
        for key, values in results.items()
    }
    plot_metrics = {
        key: {
            "per_sample": [
                values["per_sample"][int(position)]
                for position in visualization_positions
            ]
        }
        for key, values in metrics.items()
    }
    _generate_plots(
        plot_samples,
        plot_results,
        plot_metrics,
        output_dir,
    )


if __name__ == "__main__":
    main()
