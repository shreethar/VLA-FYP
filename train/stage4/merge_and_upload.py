#!/usr/bin/env python3
"""Merge a Stage 4 student adapter into Stage 2 and upload one HF model repo.

The resulting repository contains:

* the Stage 4 LoRA weights merged into the full Stage 2 VLM;
* the processor/tokenizer from the Stage 2 model;
* the five learned spatial embeddings and trained waypoint head in
  ``spatial_parameters.pt``; and
* enough metadata to identify the exact Stage 4 checkpoint used.

VGGT and the Spatial Forcing projector are training-only teachers and are not
required by the merged inference model.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("stage4.merge")

DEFAULT_STAGE2_MODEL = "shreethar/LatentStudent-ckpt-400"
DEFAULT_REPO_ID = "shreethar/Latent-Student-Spatial-Forcing"
DEFAULT_OUTPUT_DIR = "checkpoints/Latent-Student-Spatial-Forcing-merged"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _is_stage4_checkpoint(path: Path) -> bool:
    adapter = path / "student_lora"
    has_adapter_weights = any(
        (adapter / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    )
    return (
        (adapter / "adapter_config.json").is_file()
        and has_adapter_weights
        and (path / "spatial_parameters.pt").is_file()
    )


def resolve_stage4_checkpoint(path: str | Path) -> tuple[Path, str]:
    """Resolve a direct checkpoint or a run root to the checkpoint to merge.

    Run roots prefer ``best_checkpoint.json``. Only when that file is absent do
    they fall back to the highest-numbered complete ``step_*`` directory.
    """
    requested = Path(path).expanduser()
    if not requested.exists():
        raise FileNotFoundError(f"Stage 4 checkpoint path does not exist: {requested}")
    requested = requested.resolve()

    if _is_stage4_checkpoint(requested):
        return requested, "direct"

    best_pointer = requested / "best_checkpoint.json"
    if best_pointer.is_file():
        pointer = _load_json(best_pointer)
        raw_checkpoint = pointer.get("checkpoint_path")
        if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
            raise ValueError(
                f"{best_pointer} does not contain a non-empty checkpoint_path"
            )

        stored = Path(raw_checkpoint).expanduser()
        candidates = []
        if stored.is_absolute():
            candidates.append(stored)
        else:
            # Training records paths relative to its launch directory. The
            # basename fallback also keeps the pointer usable after moving a run.
            candidates.extend(
                [Path.cwd() / stored, requested / stored, requested / stored.name]
            )
        for candidate in candidates:
            candidate = candidate.resolve()
            if _is_stage4_checkpoint(candidate):
                return candidate, "best_checkpoint.json"
        rendered = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Best-checkpoint pointer {best_pointer} did not resolve to a complete "
            f"Stage 4 checkpoint. Checked: {rendered}"
        )

    step_directories = sorted(
        (
            candidate
            for candidate in requested.glob("step_*")
            if candidate.is_dir() and _is_stage4_checkpoint(candidate)
        ),
        key=lambda candidate: candidate.name,
    )
    if step_directories:
        return step_directories[-1].resolve(), "latest_complete_step_fallback"

    raise FileNotFoundError(
        f"{requested} is neither a complete Stage 4 checkpoint nor a run "
        "directory containing one"
    )


def _prepare_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new directory "
            "so an existing model cannot be overwritten accidentally."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _validate_spatial_parameters(path: Path) -> dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    if "spatial_tokens" not in state or "spatial_mlp" not in state:
        raise ValueError(
            f"{path} must contain spatial_tokens and spatial_mlp from Stage 4"
        )
    tokens = state["spatial_tokens"]
    mlp = state["spatial_mlp"]
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError("spatial_tokens must be a [K, hidden_dim] tensor")
    if tokens.shape[0] != 5:
        raise ValueError(
            f"Expected the trained student's five spatial tokens, got {tokens.shape[0]}"
        )
    if not isinstance(mlp, dict) or not mlp:
        raise ValueError("spatial_mlp must be a non-empty state dictionary")
    tensors = [tokens, *(value for value in mlp.values() if isinstance(value, torch.Tensor))]
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError("Spatial parameters contain NaN or infinity")
    return {
        "num_spatial_tokens": int(tokens.shape[0]),
        "hidden_size": int(tokens.shape[1]),
        "spatial_mlp_tensors": len(tensors) - 1,
    }


def _normalise_model_id(value: str) -> str:
    return value.strip().rstrip("/").casefold()


def _checkpoint_label(checkpoint: Path) -> str:
    """Return useful provenance without publishing an absolute machine path."""
    return f"{checkpoint.parent.name}/{checkpoint.name}"


def _model_card(
    *,
    repo_id: str,
    base_model: str,
    checkpoint: Path,
    checkpoint_selection: str,
    stage4_config: dict[str, Any],
) -> str:
    alpha = stage4_config.get("alpha", "unknown")
    beta = stage4_config.get("beta", "unknown")
    gamma = stage4_config.get("gamma", "unknown")
    step = checkpoint.name.removeprefix("step_")
    checkpoint_label = _checkpoint_label(checkpoint)
    return f"""---
library_name: transformers
pipeline_tag: image-text-to-text
base_model: {base_model}
tags:
- qwen3_5
- vision-language-action
- robotics
- latent-reasoning
- spatial-forcing
---

# Latent Student Spatial Forcing

This is the standalone Stage 4 inference package for the Latent Student. The
Stage 4 LoRA adapter has been merged into `{base_model}`.

## Included components

- Merged Qwen3.5 vision-language model weights
- Processor and tokenizer
- `spatial_parameters.pt`: five learned spatial-slot embeddings and the
  Stage 4 waypoint MLP
- `latent_student_config.json`: packaging and provenance metadata
- `stage4_config.json`: training configuration, when present in the checkpoint

VGGT and the Spatial Forcing projection head were training-only supervision
components. They are not needed for waypoint inference.

## Provenance

- Stage 2 model: `{base_model}`
- Stage 4 checkpoint: `{checkpoint_label}`
- Checkpoint selection: `{checkpoint_selection}`
- Checkpoint step: `{step}`
- Loss weights: alpha={alpha}, beta={beta}, gamma={gamma}

## Loading for waypoint inference

Use the project's `LatentStudent` wrapper so the spatial slots and waypoint
head are restored alongside the merged VLM:

```python
from transformers import AutoTokenizer
from train.stage4.checkpointing import load_latent_student_checkpoint

repo_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
end_think_token_id = tokenizer.convert_tokens_to_ids("</think>")

student = load_latent_student_checkpoint(
    checkpoint=repo_id,
    end_think_token_id=end_think_token_id,
    trainable=False,
    M=6,
    K=5,
)
student.eval()
```

Loading only with `AutoModelForImageTextToText` restores the merged VLM but not
the external spatial slots or waypoint head. Use the wrapper above for the
complete Latent Student behavior.
"""


def merge_stage4_model(args: argparse.Namespace) -> tuple[Path, Path]:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    checkpoint, selection = resolve_stage4_checkpoint(args.stage4_checkpoint)
    adapter_dir = checkpoint / "student_lora"
    spatial_path = checkpoint / "spatial_parameters.pt"
    spatial_metadata = _validate_spatial_parameters(spatial_path)

    stage4_config_path = checkpoint / "stage4_config.json"
    stage4_config = (
        _load_json(stage4_config_path) if stage4_config_path.is_file() else {}
    )

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    adapter_base = getattr(peft_config, "base_model_name_or_path", None)
    if (
        adapter_base
        and _normalise_model_id(str(adapter_base))
        != _normalise_model_id(args.base_model)
        and not args.allow_base_mismatch
    ):
        raise ValueError(
            "The Stage 4 adapter says its base model is "
            f"{adapter_base!r}, but --base_model is {args.base_model!r}. "
            "Use the adapter's actual base, or pass --allow_base_mismatch only "
            "if you have verified the architectures and weights are identical."
        )
    output = _prepare_output_directory(args.output_dir)

    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    device_map: str | dict[str, str]
    if args.device_map == "auto":
        device_map = "auto"
    else:
        device_map = {"": args.device_map}

    LOGGER.info("Selected Stage 4 checkpoint: %s (%s)", checkpoint, selection)
    LOGGER.info("Loading Stage 2 VLM: %s", args.base_model)
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        dtype=dtype_by_name[args.dtype],
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    text_config = getattr(base.config, "text_config", base.config)
    model_hidden_size = getattr(text_config, "hidden_size", None)
    if (
        model_hidden_size is not None
        and int(model_hidden_size) != spatial_metadata["hidden_size"]
    ):
        raise ValueError(
            "Spatial-token hidden size does not match the Stage 2 model: "
            f"{spatial_metadata['hidden_size']} vs {model_hidden_size}"
        )
    LOGGER.info("Loading Stage 4 adapter: %s", adapter_dir)
    adapted = PeftModel.from_pretrained(
        base,
        str(adapter_dir),
        is_trainable=False,
    )
    adapted.eval()

    LOGGER.info("Safely merging LoRA weights into the Stage 2 VLM")
    merged = adapted.merge_and_unload(safe_merge=True, progressbar=True)
    remaining_lora = [
        name for name, _ in merged.named_parameters() if "lora_" in name.casefold()
    ]
    if remaining_lora:
        raise RuntimeError(
            "LoRA parameters remain after merge: " + ", ".join(remaining_lora[:5])
        )
    merged.config.use_cache = True

    LOGGER.info("Saving merged model to %s", output)
    merged.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor_source = args.processor or args.base_model
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
    )
    processor.save_pretrained(output)

    shutil.copy2(spatial_path, output / "spatial_parameters.pt")
    if stage4_config_path.is_file():
        shutil.copy2(stage4_config_path, output / "stage4_config.json")

    package_metadata = {
        "format_version": 1,
        "model_type": "latent_student_spatial_forcing",
        "base_model": args.base_model,
        "stage4_checkpoint": _checkpoint_label(checkpoint),
        "checkpoint_selection": selection,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "M": int(stage4_config.get("M", 6)),
        "K": int(stage4_config.get("K", spatial_metadata["num_spatial_tokens"])),
        "student_visual_layer": stage4_config.get("student_visual_layer"),
        "vggt_layer": stage4_config.get("vggt_layer"),
        "spatial_parameters": spatial_metadata,
        "sf_projector_required_for_inference": False,
        "vggt_required_for_inference": False,
    }
    (output / "latent_student_config.json").write_text(
        json.dumps(package_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        _model_card(
            repo_id=args.repo_id,
            base_model=args.base_model,
            checkpoint=checkpoint,
            checkpoint_selection=selection,
            stage4_config=stage4_config,
        ),
        encoding="utf-8",
    )
    return output, checkpoint


def upload_model(output: Path, args: argparse.Namespace) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    LOGGER.info("Creating or reusing Hugging Face model repo %s", args.repo_id)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    LOGGER.info("Uploading %s to %s", output, args.repo_id)
    commit = api.upload_folder(
        folder_path=str(output),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    return str(commit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage4_checkpoint",
        default="checkpoints/stage4_partial_run_2",
        help=(
            "Stage 4 step directory or run root. A run root uses "
            "best_checkpoint.json when available."
        ),
    )
    parser.add_argument("--base_model", default=DEFAULT_STAGE2_MODEL)
    parser.add_argument(
        "--processor",
        help="Processor source; defaults to --base_model.",
    )
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device_map",
        default="cpu",
        help="Use 'cpu' for the safest merge or 'auto' for Accelerate placement.",
    )
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--no_upload", action="store_true")
    parser.add_argument("--allow_base_mismatch", action="store_true")
    parser.add_argument(
        "--commit_message",
        default="Upload merged Latent Student Spatial Forcing model",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    output, checkpoint = merge_stage4_model(args)
    LOGGER.info("Merged %s into %s", checkpoint, output)
    if args.no_upload:
        LOGGER.info("Upload skipped because --no_upload was supplied")
        return
    commit = upload_model(output, args)
    LOGGER.info("Upload complete: %s", commit)


if __name__ == "__main__":
    main()
