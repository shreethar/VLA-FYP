"""Checkpoint loading/saving helpers for Stage 4.

Stage 2 stores LoRA weights and custom spatial parameters separately. These
helpers deliberately require both pieces so a Stage 4 run can never silently
start with newly initialized waypoint slots or a newly initialized SpatialMLP.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from train.stage2.models.latent_student import LatentStudent


def _adapter_directory(checkpoint: str | Path) -> Optional[Path]:
    path = Path(checkpoint)
    if not path.exists():
        return None
    if (path / "student_lora" / "adapter_config.json").is_file():
        return path / "student_lora"
    if (path / "adapter_config.json").is_file():
        return path
    return None


def _load_spatial_state(checkpoint: str | Path) -> dict[str, Any]:
    path = Path(checkpoint)
    candidates = []
    if path.exists():
        candidates.extend(
            [
                path / "spatial_parameters.pt",
                path / "training_state.pt",
                path.parent / "spatial_parameters.pt",
                path.parent / "training_state.pt",
            ]
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            if "spatial_tokens" in state and "spatial_mlp" in state:
                return state
    else:
        from huggingface_hub import hf_hub_download

        try:
            filename = hf_hub_download(
                repo_id=str(checkpoint), filename="spatial_parameters.pt"
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Checkpoint {checkpoint!r} has no spatial_parameters.pt"
            ) from exc
        state = torch.load(filename, map_location="cpu", weights_only=False)
        if "spatial_tokens" in state and "spatial_mlp" in state:
            return state

    raise FileNotFoundError(
        "Could not find learned spatial_tokens and spatial_mlp for checkpoint "
        f"{checkpoint!r}. Stage 4 refuses to use random spatial parameters."
    )


def _restore_spatial_parameters(student: LatentStudent, state: dict[str, Any]) -> None:
    tokens = state["spatial_tokens"].to(
        device=student.spatial_tokens.device,
        dtype=student.spatial_tokens.dtype,
    )
    if tokens.shape != student.spatial_tokens.shape:
        raise ValueError(
            f"Spatial-token shape mismatch: checkpoint {tuple(tokens.shape)}, "
            f"model {tuple(student.spatial_tokens.shape)}"
        )
    student.spatial_tokens.data.copy_(tokens)
    student.spatial_mlp.load_state_dict(state["spatial_mlp"], strict=True)


def load_latent_student_checkpoint(
    checkpoint: str,
    end_think_token_id: int,
    trainable: bool,
    base_model_name: Optional[str] = None,
    M: int = 6,
    K: int = 5,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
) -> LatentStudent:
    """Load either a Stage 2 adapter checkpoint or a merged model.

    Adapter checkpoints continue training their learned Stage 2 LoRA weights.
    Merged checkpoints receive a fresh Stage 4 LoRA adapter when trainable.
    Frozen reference models are loaded without a new adapter.
    """
    adapter_dir = _adapter_directory(checkpoint)
    spatial_state = _load_spatial_state(checkpoint)

    if adapter_dir is not None:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(str(adapter_dir))
        resolved_base = base_model_name or peft_config.base_model_name_or_path
        if not resolved_base:
            raise ValueError(
                "base_model_name is required because the adapter config does not "
                "identify its base model"
            )
        student = LatentStudent(
            model_name=resolved_base,
            M=M,
            K=K,
            end_think_token_id=end_think_token_id,
            use_lora=False,
        )
        student.vlm = PeftModel.from_pretrained(
            student.vlm,
            str(adapter_dir),
            is_trainable=trainable,
        )
        student.vlm.config.use_cache = False
        if trainable:
            student.vlm.enable_input_require_grads()
            student.vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    else:
        student = LatentStudent(
            model_name=checkpoint,
            M=M,
            K=K,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            end_think_token_id=end_think_token_id,
            use_lora=trainable,
        )

    _restore_spatial_parameters(student, spatial_state)

    if trainable:
        # Slot meanings are fixed by Stage 2. The SpatialMLP may continue to
        # adapt, but the five learned input embeddings must remain immutable.
        student.spatial_tokens.requires_grad_(False)
        for parameter in student.spatial_mlp.parameters():
            parameter.requires_grad_(True)
        student.train()
    else:
        student.requires_grad_(False)
        student.eval()

    return student


def assert_spatial_tokens_frozen(student: LatentStudent) -> None:
    if student.spatial_tokens.requires_grad:
        raise RuntimeError("Stage 4 spatial tokens must be frozen")


def save_stage4_checkpoint(
    output_dir: str | Path,
    step: int,
    student: LatentStudent,
    spatial_alignment,
    optimizer,
    scheduler,
    config,
) -> Path:
    checkpoint_dir = Path(output_dir) / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    student.vlm.save_pretrained(checkpoint_dir / "student_lora")
    torch.save(
        {
            "spatial_tokens": student.spatial_tokens.detach().cpu(),
            "spatial_mlp": {
                key: value.detach().cpu()
                for key, value in student.spatial_mlp.state_dict().items()
            },
        },
        checkpoint_dir / "spatial_parameters.pt",
    )
    torch.save(
        {
            "step": step,
            "spatial_alignment": spatial_alignment.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        checkpoint_dir / "stage4_state.pt",
    )

    config_dict = asdict(config) if is_dataclass(config) else dict(config)
    with (checkpoint_dir / "stage4_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2, default=str)
    return checkpoint_dir


def restore_stage4_training_state(
    checkpoint_dir: str | Path,
    spatial_alignment,
    optimizer=None,
    scheduler=None,
) -> int:
    state_path = Path(checkpoint_dir) / "stage4_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing Stage 4 state: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    spatial_alignment.load_state_dict(state["spatial_alignment"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])
