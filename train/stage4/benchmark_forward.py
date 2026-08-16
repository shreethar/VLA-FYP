#!/usr/bin/env python3
"""Benchmark batch-size-1 forward and full prediction latency for all models.

The regular evaluator reports throughput: batch wall time divided by batch
size. This script instead defaults to one sample so its full-prediction timing
can be compared with an observed single-request latency such as 35 seconds.
Models are loaded and benchmarked sequentially to keep accelerator memory use
bounded.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.stage4.checkpointing import load_latent_student_checkpoint
from train.stage4.evaluate_all import (
    DEFAULT_DATASET,
    DEFAULT_LATENT_400,
    DEFAULT_PROCESSOR,
    DEFAULT_SPATIAL_FORCING,
    DEFAULT_STAGE1,
    DEFAULT_TEACHER,
    EXPECTED_WAYPOINTS,
    _load_parquet_split,
    _message_text,
    _model_device_map,
    _synchronize,
    parse_trajectory,
)


def _timing_summary(times: list[float], batch_size: int) -> dict[str, Any]:
    mean = statistics.fmean(times)
    return {
        "runs_seconds": times,
        "mean_batch_seconds": mean,
        "median_batch_seconds": statistics.median(times),
        "min_batch_seconds": min(times),
        "max_batch_seconds": max(times),
        "stdev_batch_seconds": statistics.pstdev(times),
        "mean_seconds_per_sample": mean / batch_size,
    }


def _measure(
    operation: Callable[[], Any],
    *,
    device: str,
    warmup_runs: int,
    measured_runs: int,
    batch_size: int,
):
    for _ in range(warmup_runs):
        warmup_output = operation()
        _synchronize(device)
        del warmup_output

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    times: list[float] = []
    final_output = None
    for _ in range(measured_runs):
        _synchronize(device)
        started = time.perf_counter()
        output = operation()
        _synchronize(device)
        times.append(time.perf_counter() - started)
        final_output = output

    summary = _timing_summary(times, batch_size)
    if device.startswith("cuda"):
        summary["peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3
        )
    else:
        summary["peak_allocated_gib"] = None
    return final_output, summary


def _prepare_inputs(processor, sample, batch_size, enable_thinking, device):
    prompt = _message_text(
        processor, sample, enable_thinking=enable_thinking
    )
    started = time.perf_counter()
    inputs = processor(
        text=[prompt] * batch_size,
        images=[sample["frames"][0]] * batch_size,
        padding=True,
        return_tensors="pt",
    ).to(device)
    _synchronize(device)
    preprocessing_seconds = time.perf_counter() - started
    return inputs, preprocessing_seconds


def _print_measurement(label: str, timing: dict[str, Any]) -> None:
    runs = ", ".join(f"{value:.4f}" for value in timing["runs_seconds"])
    print(
        f"{label}: mean={timing['mean_batch_seconds']:.4f}s/batch, "
        f"{timing['mean_seconds_per_sample']:.4f}s/sample; runs=[{runs}]"
    )


def _benchmark_text_model(
    *,
    key: str,
    model_name: str,
    processor,
    sample,
    enable_thinking: bool,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    warmup_runs: int,
    measured_runs: int,
):
    print(f"\nLoading {key}: {model_name}")
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=_model_device_map(device),
        trust_remote_code=True,
    )
    model.eval()
    _synchronize(device)
    load_seconds = time.perf_counter() - started
    print(f"Loaded {key} in {load_seconds:.2f}s")

    inputs, preprocessing_seconds = _prepare_inputs(
        processor, sample, batch_size, enable_thinking, device
    )

    print(f"Benchmarking {key} prefill forward pass...")

    def prefill():
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, return_dict=True)
        return tuple(output.logits.shape)

    prefill_shape, prefill_timing = _measure(
        prefill,
        device=device,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        batch_size=batch_size,
    )
    _print_measurement(f"{key} prefill", prefill_timing)

    print(
        f"Benchmarking {key} full generation "
        f"(max_new_tokens={max_new_tokens})..."
    )

    def generate():
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    generated, prediction_timing = _measure(
        generate,
        device=device,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        batch_size=batch_size,
    )
    _print_measurement(f"{key} full generation", prediction_timing)
    prompt_tokens = int(inputs.input_ids.shape[1])
    generated_tokens = int(generated.shape[1] - prompt_tokens)
    total_generated_tokens = generated_tokens * batch_size
    prediction_timing["generated_tokens_per_sample"] = generated_tokens
    prediction_timing["generated_tokens_per_second"] = (
        total_generated_tokens / prediction_timing["mean_batch_seconds"]
    )
    decoded = processor.tokenizer.decode(
        generated[0, prompt_tokens:], skip_special_tokens=False
    )

    report = {
        "model": model_name,
        "kind": "autoregressive_text",
        "load_seconds": load_seconds,
        "preprocessing_batch_seconds": preprocessing_seconds,
        "prompt_tokens": prompt_tokens,
        "prefill_logits_shape": list(prefill_shape),
        "prefill": prefill_timing,
        "full_prediction": prediction_timing,
        "output_preview": decoded[:2000],
    }
    del generated, inputs, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _benchmark_latent_model(
    *,
    key: str,
    model_name: str,
    processor,
    sample,
    end_think_token_id: int,
    device: str,
    batch_size: int,
    warmup_runs: int,
    measured_runs: int,
):
    print(f"\nLoading {key}: {model_name}")
    started = time.perf_counter()
    student = load_latent_student_checkpoint(
        checkpoint=model_name,
        end_think_token_id=end_think_token_id,
        trainable=False,
        M=6,
        K=5,
    ).to(device)
    student.eval()
    _synchronize(device)
    load_seconds = time.perf_counter() - started
    print(f"Loaded {key} in {load_seconds:.2f}s")

    inputs, preprocessing_seconds = _prepare_inputs(
        processor, sample, batch_size, True, device
    )

    print(f"Benchmarking {key} VLM prefill forward pass...")

    def prefill():
        with torch.inference_mode():
            output = student.vlm(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
        return tuple(output.logits.shape)

    prefill_shape, prefill_timing = _measure(
        prefill,
        device=device,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        batch_size=batch_size,
    )
    _print_measurement(f"{key} prefill", prefill_timing)

    print(f"Benchmarking {key} six-latent waypoint prediction...")

    def predict():
        with torch.inference_mode():
            _, _, _, waypoints = student.generate_latents(
                input_ids=inputs.input_ids,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                attention_mask=inputs.attention_mask,
                pixel_values_videos=inputs.get("pixel_values_videos"),
                video_grid_thw=inputs.get("video_grid_thw"),
            )
        return waypoints

    waypoints, prediction_timing = _measure(
        predict,
        device=device,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        batch_size=batch_size,
    )
    _print_measurement(f"{key} full prediction", prediction_timing)
    report = {
        "model": model_name,
        "kind": "six_latents_plus_five_spatial_tokens",
        "load_seconds": load_seconds,
        "preprocessing_batch_seconds": preprocessing_seconds,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "prefill_logits_shape": list(prefill_shape),
        "prefill": prefill_timing,
        "full_prediction": prediction_timing,
        "waypoints_0_1": waypoints.detach().float().cpu().tolist(),
    }
    del waypoints, inputs, student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _local_sample(image_path: str, instruction: str):
    image = Image.open(image_path).convert("RGB")
    human = (
        "You are a robot manipulation assistant. Given an observation image "
        "and a task instruction, predict the end-effector's 2D trajectory as "
        "5 waypoints. Output ONLY the coordinate list in this exact format: "
        "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]\n\n"
        f"Task: {instruction}. What is the trajectory that the end effector "
        "should take?"
    )
    return {"human": human, "frames": [image]}, {
        "source": "local_image",
        "image": str(Path(image_path).resolve()),
        "instruction": instruction,
    }


def _dataset_sample(args):
    dataset, parquet_files = _load_parquet_split(
        dataset_repo=args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
        parquet_dir=args.parquet_dir,
        streaming=True,
    )
    row = None
    dataset_index = None
    print("Streaming local Parquet rows until one valid trajectory is found...")
    for index, candidate in enumerate(dataset):
        if candidate.get("dataset") != "molmoact":
            continue
        if candidate.get("type") != "trajectory":
            continue
        trajectory = parse_trajectory(candidate.get("assistant", ""))
        if trajectory is None or len(trajectory) != EXPECTED_WAYPOINTS:
            continue
        row = candidate
        dataset_index = index
        break
    if row is None:
        raise ValueError(
            "No valid five-waypoint MolmoAct trajectory was found in the split"
        )

    image = row["frames"][0].convert("RGB").copy()
    sample = {"human": row["human"], "frames": [image]}
    source = {
        "source": "dataset",
        "dataset": args.dataset,
        "split": args.split,
        "dataset_index": dataset_index,
        "parquet_files": [str(path) for path in parquet_files],
        "instruction": row["human"],
    }
    del dataset
    gc.collect()
    return sample, source


def _print_summary(results):
    print("\nLatency summary")
    print(
        "model             load(s)  prefill(s/sample)  "
        "prediction(s/sample)  generated tokens"
    )
    for key, result in results.items():
        generated = result["full_prediction"].get(
            "generated_tokens_per_sample", "fixed 6+5"
        )
        print(
            f"{key:<17} {result['load_seconds']:>7.2f}  "
            f"{result['prefill']['mean_seconds_per_sample']:>17.4f}  "
            f"{result['full_prediction']['mean_seconds_per_sample']:>20.4f}  "
            f"{str(generated):>16}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", help="Optional local image; avoids dataset loading."
    )
    parser.add_argument(
        "--instruction",
        default="close the box",
        help="Task instruction used with --image.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir")
    parser.add_argument("--parquet_dir")
    parser.add_argument("--processor", default=DEFAULT_PROCESSOR)
    parser.add_argument("--stage1_model", default=DEFAULT_STAGE1)
    parser.add_argument("--teacher_model", default=DEFAULT_TEACHER)
    parser.add_argument("--latent400_model", default=DEFAULT_LATENT_400)
    parser.add_argument(
        "--spatial_forcing_model", default=DEFAULT_SPATIAL_FORCING
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--measured_runs", type=int, default=3)
    parser.add_argument(
        "--output", default="evaluation_stage4/forward_benchmark.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.warmup_runs < 0 or args.measured_runs <= 0:
        raise ValueError("Warmup must be non-negative and measured runs positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    sample, sample_source = (
        _local_sample(args.image, args.instruction)
        if args.image
        else _dataset_sample(args)
    )

    print(f"Loading processor: {args.processor}")
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.processor, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    processor_load_seconds = time.perf_counter() - started
    end_think_token_id = processor.tokenizer.convert_tokens_to_ids("</think>")

    common = {
        "processor": processor,
        "sample": sample,
        "device": args.device,
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
    }
    results = {
        "stage1": _benchmark_text_model(
            key="stage1",
            model_name=args.stage1_model,
            enable_thinking=False,
            max_new_tokens=args.max_new_tokens,
            **common,
        ),
        "teacher": _benchmark_text_model(
            key="teacher",
            model_name=args.teacher_model,
            enable_thinking=True,
            max_new_tokens=args.max_new_tokens,
            **common,
        ),
        "latent400": _benchmark_latent_model(
            key="latent400",
            model_name=args.latent400_model,
            end_think_token_id=end_think_token_id,
            **common,
        ),
        "spatial_forcing": _benchmark_latent_model(
            key="spatial_forcing",
            model_name=args.spatial_forcing_model,
            end_think_token_id=end_think_token_id,
            **common,
        ),
    }
    _print_summary(results)

    report = {
        "configuration": {
            "device": args.device,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "processor": args.processor,
            "processor_load_seconds": processor_load_seconds,
            "sample": sample_source,
            "timing_scope": (
                "CUDA-synchronized model execution; preprocessing and loading "
                "are reported separately"
            ),
        },
        "models": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
