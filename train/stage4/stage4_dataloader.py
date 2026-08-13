"""MolmoAct dataloader for Stage 4 Spatial Forcing.

Each selected sample has two deliberately different visual paths:

* the reference and trainable latent students receive only ``primary``;
* frozen VGGT jointly receives ``[primary, wrist]`` in that order.

The dataset is streamed because the full MolmoAct mixture is very large. Rows
are first validated, then content-hashed for reproducible 10% sampling and a
leakage-resistant 70/15/15 train/validation/test partition. No full local copy
or preliminary counting pass is required.
"""

from __future__ import annotations

import ast
import hashlib
import io
import logging
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
PARTITIONS = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)

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


def _update_hash_with_image(hasher, value: Any) -> None:
    """Hash image content without depending on cache paths or worker ordering."""
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            hasher.update(value["bytes"])
            return
        if value.get("path"):
            with open(value["path"], "rb") as image_file:
                for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return
    if isinstance(value, Image.Image):
        hasher.update(f"{value.mode}:{value.size}".encode("utf-8"))
        hasher.update(value.tobytes())
        return
    if isinstance(value, np.ndarray):
        hasher.update(f"{value.dtype}:{value.shape}".encode("utf-8"))
        hasher.update(value.tobytes())
        return
    if isinstance(value, bytes):
        hasher.update(value)
        return
    if isinstance(value, str):
        with open(value, "rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                hasher.update(chunk)
        return
    raise TypeError(f"Unsupported image value for fingerprint: {type(value).__name__}")


def molmoact_row_fingerprint(row: dict, task_name: str) -> str:
    """Stable content ID used for sampling, partitioning, and logging.

    The planner image is included so repeated task/trajectory annotations on
    different observations remain distinct. Exact duplicates intentionally get
    the same ID and therefore cannot leak across dataset partitions.
    """
    explicit_id = row.get("id", row.get("uuid"))
    hasher = hashlib.blake2b(digest_size=16, person=b"stage4-row-v1")
    if explicit_id is not None:
        hasher.update(f"id:{explicit_id}".encode("utf-8"))
    else:
        hasher.update(task_name.encode("utf-8"))
        hasher.update(repr(row.get("annotation")).encode("utf-8"))
        _update_hash_with_image(hasher, row["primary"])
    return hasher.hexdigest()


def _hash_fraction(fingerprint: str, seed: int, purpose: str) -> float:
    digest = hashlib.blake2b(
        f"{seed}:{purpose}:{fingerprint}".encode("utf-8"),
        digest_size=8,
        person=b"stage4-split",
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def partition_for_fingerprint(
    fingerprint: str,
    seed: int = 42,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> str:
    """Assign a stable content fingerprint to train/validation/test."""
    if len(split_ratios) != 3 or any(ratio < 0.0 for ratio in split_ratios):
        raise ValueError("split_ratios must contain three non-negative values")
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("split_ratios must sum to 1.0")
    value = _hash_fraction(fingerprint, seed, "partition")
    if value < split_ratios[0]:
        return "train"
    if value < split_ratios[0] + split_ratios[1]:
        return "validation"
    return "test"


def is_sampled_fingerprint(
    fingerprint: str,
    sample_ratio: float,
    seed: int = 42,
) -> bool:
    return sample_ratio >= 1.0 or (
        _hash_fraction(fingerprint, seed, "sample") < sample_ratio
    )


def _sharded_rows(rows: Iterable[dict]) -> Iterable[dict]:
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
    return rows


class MolmoActStage4Dataset(IterableDataset):
    """Streaming, filtered MolmoAct samples with primary/wrist view ownership."""

    def __init__(
        self,
        hf_split,
        processor,
        max_length: int = 1024,
        sample_ratio: float = 0.1,
        seed: int = 42,
        data_partition: str = "train",
        split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    ) -> None:
        super().__init__()
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in (0,1]")
        if data_partition not in PARTITIONS:
            raise ValueError(f"data_partition must be one of {PARTITIONS}")
        # Validate once at construction rather than on every row.
        partition_for_fingerprint("validation", seed, split_ratios)
        self.hf_split = hf_split
        self.processor = processor
        self.max_length = max_length
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.data_partition = data_partition
        self.split_ratios = split_ratios

    def _convert_row(
        self,
        row: dict,
        row_index: int,
        task_name: str,
        fingerprint: str,
    ) -> Optional[dict]:
        waypoints = parse_molmoact_annotation(row.get("annotation"))
        if waypoints is None:
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
            "data_partition": self.data_partition,
            "sample_id": f"molmoact_{fingerprint}",
            "task_name": task_name,
            # View order is part of the training contract: 0=planner, 1=context.
            "vggt_images": torch.stack(
                [_to_vggt_tensor(primary), _to_vggt_tensor(wrist)], dim=0
            ),
            "planner_view_index": 0,
            "original_frame_sizes": [list(primary.size), list(wrist.size)],
        }

    def __iter__(self) -> Iterator[dict]:
        rows = _sharded_rows(self.hf_split)
        for row_index, row in enumerate(rows):
            # Validate cheap fields before sampling and image preprocessing.
            if parse_molmoact_annotation(row.get("annotation")) is None:
                continue
            conversation = row.get("conversations", row.get("conversation"))
            task_name = extract_task_name(conversation)
            if task_name is None:
                continue
            if row.get("primary") is None or row.get("wrist", row.get("wrirst")) is None:
                continue
            try:
                fingerprint = molmoact_row_fingerprint(row, task_name)
            except (OSError, TypeError, ValueError):
                logger.warning("Skipping row %d that cannot be fingerprinted", row_index)
                continue
            if not is_sampled_fingerprint(fingerprint, self.sample_ratio, self.seed):
                continue
            if partition_for_fingerprint(
                fingerprint, self.seed, self.split_ratios
            ) != self.data_partition:
                continue
            sample = self._convert_row(row, row_index, task_name, fingerprint)
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
    batch["data_partitions"] = [sample["data_partition"] for sample in samples]
    batch["original_frame_sizes"] = [
        sample["original_frame_sizes"] for sample in samples
    ]
    return batch


def build_stage4_dataloader(
    processor,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_config: str = DEFAULT_HF_CONFIG,
    split: str = "train",
    data_partition: str = "train",
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    batch_size: int = 16,
    num_workers: int = 2,
    max_length: int = 1024,
    sample_ratio: float = 0.1,
    seed: int = 42,
    hf_split=None,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    if hf_split is None:
        from datasets import load_dataset
        from datasets import Image as HFImage

        hf_split = load_dataset(
            hf_repo,
            hf_config,
            split=split,
            streaming=True,
        )
        # Preserve compressed image bytes until the selected row is decoded.
        hf_split = hf_split.cast_column("primary", HFImage(decode=False))
        hf_split = hf_split.cast_column("wrist", HFImage(decode=False))

    dataset = MolmoActStage4Dataset(
        hf_split,
        processor=processor,
        max_length=max_length,
        sample_ratio=sample_ratio,
        seed=seed,
        data_partition=data_partition,
        split_ratios=split_ratios,
    )
    if drop_last is None:
        drop_last = data_partition == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset sampling is already random and seeded.
        num_workers=num_workers,
        collate_fn=collate_stage4_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def build_stage4_partition_dataloaders(**kwargs) -> dict[str, DataLoader]:
    """Build independent streaming loaders for the fixed 70/15/15 partitions."""
    return {
        partition: build_stage4_dataloader(
            data_partition=partition,
            drop_last=partition == "train",
            **kwargs,
        )
        for partition in PARTITIONS
    }
