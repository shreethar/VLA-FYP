"""Trajectory dataloader for Stage 4 Spatial Forcing.

The batch contract extends Stage 2 with raw RGB images for VGGT:

``vggt_images``: ``[B, V_max, 3, 518, 518]`` in ``[0,1]``
``vggt_view_mask``: ``[B, V_max]``

Qwen continues to receive its own processor-generated packed patch tensors.
VGGT must not consume those normalized/packed tensors.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from train.stage2.stage2_dataloader import (
    IMAGE_SIZE,
    Stage2Dataset,
    collate_stage2_batch,
)

logger = logging.getLogger(__name__)

VGGT_IMAGE_SIZE = 518


def _to_vggt_tensor(image: Image.Image) -> torch.Tensor:
    resized = image.convert("RGB").resize(
        (VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE), Image.Resampling.BICUBIC
    )
    array = np.asarray(resized, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0)


class Stage4Dataset(Stage2Dataset):
    """Stage 2 trajectory samples with a parallel VGGT image path."""

    def __init__(self, hf_split, processor, max_length: int = 1024):
        super().__init__(hf_split, processor=processor, max_length=max_length)
        self.samples = [
            sample for sample in self.samples if sample["task_type"] == "trajectory"
        ]
        logger.info("Stage4Dataset: retained %s trajectory samples", len(self.samples))

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        frames: List[Image.Image] = sample["frames"]

        if len(frames) > 1:
            max_vision_tokens = self.max_length - 400
            max_frames = max(1, max_vision_tokens // 256)
            start_idx = 1 if len(frames) % 2 == 0 else 0
            frames = frames[start_idx::2]
            if len(frames) > max_frames:
                step = len(frames) / max_frames
                frames = [frames[int(i * step)] for i in range(max_frames)]

        frames = [frame.convert("RGB") for frame in frames]
        vggt_images = torch.stack([_to_vggt_tensor(frame) for frame in frames])
        qwen_frames = [
            frame.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
            if frame.size != (IMAGE_SIZE, IMAGE_SIZE)
            else frame
            for frame in frames
        ]

        if len(qwen_frames) == 1:
            content = [
                {"type": "image", "image": qwen_frames[0]},
                {"type": "text", "text": sample["human"]},
            ]
        else:
            content = [
                {"type": "video", "video": qwen_frames},
                {"type": "text", "text": sample["human"]},
            ]

        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.processor(
            text=[text],
            images=qwen_frames if len(qwen_frames) == 1 else None,
            videos=qwen_frames if len(qwen_frames) > 1 else None,
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
            "pixel_values_videos": inputs.get("pixel_values_videos"),
            "video_grid_thw": inputs.get("video_grid_thw"),
            "gt_waypoints": sample["gt_wpts"],
            "task_type": sample["task_type"],
            "qa_answer": None,
            "dataset": sample["dataset"],
            "sample_id": sample["id"],
            "vggt_images": vggt_images,
        }


def collate_stage4_batch(samples: List[dict]) -> dict:
    batch = collate_stage2_batch(samples)
    max_views = max(sample["vggt_images"].shape[0] for sample in samples)
    _, channels, height, width = samples[0]["vggt_images"].shape
    images = torch.zeros(
        len(samples), max_views, channels, height, width, dtype=torch.float32
    )
    view_mask = torch.zeros(len(samples), max_views, dtype=torch.bool)
    for batch_idx, sample in enumerate(samples):
        view_count = sample["vggt_images"].shape[0]
        images[batch_idx, :view_count] = sample["vggt_images"]
        view_mask[batch_idx, :view_count] = True
    batch["vggt_images"] = images
    batch["vggt_view_mask"] = view_mask
    return batch


def build_stage4_dataloader(
    processor,
    hf_repo: str,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 2,
    max_length: int = 1024,
    subset_ratio: float = 1.0,
    shuffle: bool = True,
    hf_split=None,
) -> DataLoader:
    if hf_split is None:
        from datasets import load_dataset

        try:
            data_files = {split: f"hf://datasets/{hf_repo}/data/{split}-*.parquet"}
            hf_split = load_dataset(
                "parquet", data_files=data_files, split=split
            )
        except Exception:
            logger.warning("Parquet loading failed; trying load_dataset(%s)", hf_repo)
            hf_split = load_dataset(hf_repo, split=split)

    if not 0.0 < subset_ratio <= 1.0:
        raise ValueError("subset_ratio must be in (0,1]")
    if subset_ratio < 1.0:
        count = max(1, int(len(hf_split) * subset_ratio))
        hf_split = hf_split.shuffle(seed=42).select(range(count))

    dataset = Stage4Dataset(hf_split, processor=processor, max_length=max_length)
    if not dataset:
        raise ValueError("Stage 4 dataset contains no trajectory samples")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_stage4_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
