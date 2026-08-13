"""MolmoAct dataloader for Stage 4 Spatial Forcing.

Each selected sample has two deliberately different visual paths:

* the reference and trainable latent students receive only ``primary``;
* frozen VGGT jointly receives ``[primary, wrist]`` in that order.

The dataset is streamed because the full MolmoAct mixture is very large.  Rows
are first validated (task, two images, and five waypoints), then retained by a
seeded Bernoulli sample.  Thus ``sample_ratio=0.1`` is a reproducible random
approximately-10% sample of usable rows without materialising the dataset.
"""

from __future__ import annotations

import ast
import hashlib
import io
import logging
import random
import re
from typing import Any, Iterable, Iterator, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from train.stage2.stage2_dataloader import IMAGE_SIZE, collate_stage2_batch

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "allenai/MolmoAct-Midtraining-Mixture"
DEFAULT_HF_CONFIG = "molmoact_tabletop_primary"
VGGT_IMAGE_SIZE = 518
K_WAYPOINTS = 5
ANNOTATION_MIN = 1.0
ANNOTATION_MAX = 256.0

TRAJECTORY_PROMPT = (
    "You are a robot manipulation assistant. Given an observation image and a "
    "task instruction, predict the end-effector's 2D trajectory as 5 waypoints. "
    "Output ONLY the coordinate list in this exact format: "
    "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]\n\n"
    "Task: {task_name}. What is the trajectory that the end effector should take?"
)

_TASK_RE = re.compile(
    r"\bThe\s+task\s+is\s+(.+?)(?=(?:[.!?]\s)|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _to_rgb_image(value: Any) -> Image.Image:
    """Decode the common Hugging Face Image representations."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (str, bytes)):
        source = io.BytesIO(value) if isinstance(value, bytes) else value
        return Image.open(source).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value).__name__}")


def _to_vggt_tensor(image: Image.Image) -> torch.Tensor:
    resized = image.resize(
        (VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE), Image.Resampling.BICUBIC
    )
    array = np.asarray(resized, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0)


def _conversation_values(conversation: Any) -> list[tuple[str, str]]:
    """Normalize either HF dict-of-lists or list-of-message conversations."""
    if isinstance(conversation, dict):
        senders = conversation.get("from", [])
        values = conversation.get("value", [])
        if isinstance(senders, str):
            senders = [senders]
        if isinstance(values, str):
            values = [values]
        return [(str(sender), str(value)) for sender, value in zip(senders, values)]
    if isinstance(conversation, list):
        result = []
        for message in conversation:
            if isinstance(message, dict):
                result.append(
                    (str(message.get("from", "")), str(message.get("value", "")))
                )
        return result
    if isinstance(conversation, str):
        return [("human", conversation)]
    return []


def extract_task_name(conversation: Any) -> Optional[str]:
    """Extract only ``{task}`` from the first human ``The task is {task}.``."""
    messages = _conversation_values(conversation)
    human_text = next(
        (value for sender, value in messages if sender.lower() in {"human", "user"}),
        messages[0][1] if messages else "",
    )
    match = _TASK_RE.search(human_text)
    if match is None:
        return None
    task_name = " ".join(match.group(1).split()).strip(" .!?\t\r\n")
    return task_name or None


def build_trajectory_prompt(task_name: str) -> str:
    clean_name = " ".join(task_name.split()).strip(" .!?\t\r\n")
    if not clean_name:
        raise ValueError("task_name cannot be empty")
    return TRAJECTORY_PROMPT.format(task_name=clean_name)


def parse_molmoact_annotation(annotation: Any) -> Optional[torch.Tensor]:
    """Return the first five MolmoAct points normalized from [1,256] to [0,1]."""
    if annotation is None:
        return None
    if isinstance(annotation, str):
        if not annotation.strip():
            return None
        try:
            annotation = ast.literal_eval(annotation)
        except (SyntaxError, ValueError):
            return None
    if isinstance(annotation, np.ndarray):
        annotation = annotation.tolist()
    if not isinstance(annotation, (list, tuple)) or len(annotation) < K_WAYPOINTS:
        return None

    points: list[list[float]] = []
    for point in annotation[:K_WAYPOINTS]:
        if not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if not (
            np.isfinite(x)
            and np.isfinite(y)
            and ANNOTATION_MIN <= x <= ANNOTATION_MAX
            and ANNOTATION_MIN <= y <= ANNOTATION_MAX
        ):
            return None
        points.append([x, y])

    waypoints = torch.tensor(points, dtype=torch.float32)
    return (waypoints - ANNOTATION_MIN) / (ANNOTATION_MAX - ANNOTATION_MIN)


def _sharded_rows(rows: Iterable[dict], seed: int) -> tuple[Iterable[dict], random.Random]:
    """Shard an HF iterable across distributed ranks and DataLoader workers."""
    worker = get_worker_info()
    workers = worker.num_workers if worker is not None else 1
    worker_id = worker.id if worker is not None else 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size, rank = 1, 0

    shard_count = world_size * workers
    shard_index = rank * workers + worker_id
    # Hugging Face IterableDataset automatically assigns its internal data
    # shards to PyTorch workers. Applying .shard() here as well would discard
    # another fraction of the data. The fallback is for plain Python iterables
    # used by tests/custom callers.
    is_hf_iterable = type(rows).__module__.startswith("datasets.")
    if is_hf_iterable and world_size > 1:
        from datasets.distributed import split_dataset_by_node

        rows = split_dataset_by_node(rows, rank=rank, world_size=world_size)
    elif shard_count > 1 and not is_hf_iterable:
        rows = (
            row for index, row in enumerate(rows) if index % shard_count == shard_index
        )
    return rows, random.Random(seed + 1_000_003 * shard_index)


class MolmoActStage4Dataset(IterableDataset):
    """Streaming, filtered MolmoAct samples with primary/wrist view ownership."""

    def __init__(
        self,
        hf_split,
        processor,
        max_length: int = 1024,
        sample_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in (0,1]")
        self.hf_split = hf_split
        self.processor = processor
        self.max_length = max_length
        self.sample_ratio = sample_ratio
        self.seed = seed

    def _convert_row(self, row: dict, row_index: int) -> Optional[dict]:
        waypoints = parse_molmoact_annotation(row.get("annotation"))
        if waypoints is None:
            return None
        conversation = row.get("conversations", row.get("conversation"))
        task_name = extract_task_name(conversation)
        if task_name is None:
            return None

        # Official schema is ``wrist``; tolerate the typo used in early notes.
        wrist_value = row.get("wrist", row.get("wrirst"))
        if row.get("primary") is None or wrist_value is None:
            return None
        try:
            primary = _to_rgb_image(row["primary"])
            wrist = _to_rgb_image(wrist_value)
        except (OSError, TypeError, ValueError):
            logger.warning("Skipping row %d with an unreadable primary/wrist image", row_index)
            return None

        prompt = build_trajectory_prompt(task_name)
        qwen_primary = primary.resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC
        )
        content = [
            {"type": "image", "image": qwen_primary},
            {"type": "text", "text": prompt},
        ]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.processor(
            text=[text],
            images=[qwen_primary],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        sample_id = row.get("id", row.get("uuid"))
        if sample_id is None:
            fingerprint = hashlib.blake2b(
                f"{task_name}|{row.get('annotation')}".encode("utf-8"),
                digest_size=8,
            ).hexdigest()
            sample_id = f"molmoact_{fingerprint}"
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "pixel_values_videos": None,
            "video_grid_thw": None,
            "gt_waypoints": waypoints,
            "task_type": "trajectory",
            "qa_answer": None,
            "dataset": DEFAULT_HF_CONFIG,
            "sample_id": sample_id,
            "task_name": task_name,
            # View order is part of the training contract: 0=planner, 1=context.
            "vggt_images": torch.stack(
                [_to_vggt_tensor(primary), _to_vggt_tensor(wrist)], dim=0
            ),
            "planner_view_index": 0,
            "original_frame_sizes": [list(primary.size), list(wrist.size)],
        }

    def __iter__(self) -> Iterator[dict]:
        rows, rng = _sharded_rows(self.hf_split, self.seed)
        for row_index, row in enumerate(rows):
            # Validate cheap fields before sampling and image preprocessing.
            if parse_molmoact_annotation(row.get("annotation")) is None:
                continue
            conversation = row.get("conversations", row.get("conversation"))
            if extract_task_name(conversation) is None:
                continue
            if row.get("primary") is None or row.get("wrist", row.get("wrirst")) is None:
                continue
            if self.sample_ratio < 1.0 and rng.random() >= self.sample_ratio:
                continue
            sample = self._convert_row(row, row_index)
            if sample is not None:
                yield sample


# Backward-compatible import name. It now has iterable/streaming semantics.
Stage4Dataset = MolmoActStage4Dataset


def collate_stage4_batch(samples: List[dict]) -> dict:
    if not samples:
        raise ValueError("Cannot collate an empty Stage 4 batch")
    if any(sample["vggt_images"].shape[0] != 2 for sample in samples):
        raise ValueError("MolmoAct Stage 4 requires exactly [primary, wrist] for VGGT")
    if any(sample["planner_view_index"] != 0 for sample in samples):
        raise ValueError("MolmoAct planner_view_index must be 0 (primary)")

    batch = collate_stage2_batch(samples)
    batch["vggt_images"] = torch.stack(
        [sample["vggt_images"] for sample in samples], dim=0
    )
    batch["vggt_view_mask"] = torch.ones(len(samples), 2, dtype=torch.bool)
    batch["planner_view_indices"] = torch.zeros(len(samples), dtype=torch.long)
    batch["task_names"] = [sample["task_name"] for sample in samples]
    batch["original_frame_sizes"] = [
        sample["original_frame_sizes"] for sample in samples
    ]
    return batch


def build_stage4_dataloader(
    processor,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_HF_CONFIG,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 2,
    max_length: int = 1024,
    sample_ratio: float = 0.1,
    seed: int = 42,
    hf_split=None,
) -> DataLoader:
    if hf_split is None:
        from datasets import load_dataset

        hf_split = load_dataset(
            hf_repo,
            hf_config,
            split=split,
            streaming=True,
        )

    dataset = MolmoActStage4Dataset(
        hf_split,
        processor=processor,
        max_length=max_length,
        sample_ratio=sample_ratio,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset sampling is already random and seeded.
        num_workers=num_workers,
        collate_fn=collate_stage4_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
