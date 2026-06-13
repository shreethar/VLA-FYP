"""
stage2_dataloader.py
--------------------
Stage 2 dataset and dataloader.

Loads the pre-materialized HF subset (shreethar/FYP-Stage2-dataset) and
filters it to trajectory-type records only (MolmoAct), since Stage 2 requires
ground-truth waypoints for both the reward function and L_ans.

Batch format (matches train_stage2.py expectations exactly):
    input_ids        [B, seq]           — tokenized prompt (vision + instruction)
    pixel_values     [total_patches, C, H, W] or None
    image_grid_thw   [total_images, 3]  or None
    attention_mask   [B, seq]
    gt_waypoints     [B, K, 2]          — normalised [0, 1], K=5
    ground_truth     dict               — {"gt_waypoints": [B, K, 2]}

Usage:
    from stage2_dataloader import build_stage2_dataloader
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
    loader = build_stage2_dataloader(
        hf_repo="shreethar/FYP-Stage2-dataset",
        processor=processor,
        split="train",
        batch_size=4,
        num_workers=2,
    )
    for batch in loader:
        ...  # pass directly into train_stage2.py's training loop
"""

import ast
import re
import logging
from typing import Optional, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
K_WAYPOINTS = 5
IMAGE_SIZE  = 448   # must match stage_1_datasets_static.py

# ── Stage 2 system prompt for trajectory tasks ────────────────────────────────
# WHY this is needed:
#   Stage 1 SFT trained the model with TRAJ_SYSTEM which instructs:
#       "Output ONLY the coordinate list in this exact format: [[x1,y1],...]"
#       with 0–1000 scale integers.
#
#   Stage 2's reward function (action_reward.py) explicitly validates:
#       if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0): return None
#   and parses from <ans>...</ans> tags, not [[...]] brackets.
#
#   Without replacing the prompt:
#       → Teacher generates [[500,300],...] in 0-1000 scale
#       → reward function rejects all of them (out-of-range)
#       → every rollout gets r_visual = 0
#       → GRPO advantage = 0 for all rollouts → no learning signal
#
#   Regex to strip Stage 1's system prefix (everything up to "\n\nTask:"):
_STAGE1_PREFIX_RE = re.compile(
    r'^You are a robot manipulation assistant\..*?\n\nTask:\s*',
    re.DOTALL,
)

STAGE2_TRAJ_SYSTEM = (
    "You are a robot manipulation assistant. "
    "Given an observation image and a task instruction, predict the "
    f"end-effector's 2D trajectory as {K_WAYPOINTS} "
    "distinct waypoints showing the continuous movement from the start to the target. "
    "If reasoning, provide a single concise plan: identify the start, locate the target, "
    "and interpolate the path. Do NOT loop, endlessly refine, or re-evaluate. "
    "Once you get a feasible trajectory output, finish reasoning, DO NOT RE-EVALUATE."
    "Finally, output the coordinate list exactly once in this exact format: "
    "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]"
)


def _reformat_traj_prompt(human_text: str) -> str:
    """
    Replace the Stage 1 trajectory system prefix with the Stage 2 one.

    Stage 1 human field:
        "You are a robot manipulation assistant...\n\nTask: pick up the cup"
    Stage 2 output:
        "[STAGE2_TRAJ_SYSTEM]\n\nTask: pick up the cup"

    The Stage 2 system prompt instructs the model to:
      1. Think via Qwen3's native <think>...</think> block
      2. Output [[x,y],...] in its naturally learned 0-1000 scale
    The reward function finds </think> and parses the [[...]] list after it.
    """
    m = _STAGE1_PREFIX_RE.match(human_text)
    if m:
        task_text = human_text[m.end():]   # just the task description
        return f"{STAGE2_TRAJ_SYSTEM}\n\nTask: {task_text.strip()}"
    # Fallback: prepend Stage 2 system prompt if pattern doesn't match
    return f"{STAGE2_TRAJ_SYSTEM}\n\n{human_text.strip()}"


# ── Waypoint parsing ──────────────────────────────────────────────────────────
# MolmoAct assistant format stored in HF dataset (from build_molmoact_records):
#   "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]"  — 0-1000 scale integers
_WPT_PATTERN = re.compile(r'\[\s*\[[\d\s,\[\]]+\]\s*\]')


def parse_waypoints(assistant_text: str, K: int = K_WAYPOINTS) -> Optional[torch.Tensor]:
    """
    Parse K waypoints from a MolmoAct assistant string.

    Expects format: [[x1,y1],[x2,y2],...,[xK,yK]]
    Coordinates are in 0-1000 scale → normalised to [0, 1] for Stage 2.

    Returns [K, 2] float32 tensor, or None if parsing fails.
    """
    m = _WPT_PATTERN.search(assistant_text)
    if not m:
        return None
    try:
        raw = ast.literal_eval(m.group(0))     # list of [x, y] pairs
        if len(raw) < K:
            return None
        pts = raw[:K]
        wpts = torch.tensor([[x / 1000.0, y / 1000.0] for x, y in pts],
                            dtype=torch.float32)   # [K, 2] in [0, 1]
        return wpts
    except Exception:
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────

class Stage2Dataset(Dataset):
    """
    Filters the pre-built HF subset to trajectory records and tokenizes
    each sample with the Qwen processor.

    Only samples with successfully parsed GT waypoints are kept.
    Tokenisation is done lazily in __getitem__ for memory efficiency.

    Parameters
    ----------
    hf_split   : A HuggingFace Dataset split object (already downloaded).
    processor  : Qwen AutoProcessor — handles both vision and text tokenization.
    max_length : Maximum token sequence length. Longer prompts are truncated.
    """

    def __init__(self, hf_split, processor, max_length: int = 1024):
        self.processor  = processor
        self.max_length = max_length

        # Filter to trajectory records and verify waypoints can be parsed
        logger.info("Scanning HF split for records…")
        self.samples = []
        skipped = 0
        for row in hf_split:
            task_type = row.get("type", "trajectory")
            qa_answer = None
            if task_type == "trajectory":
                wpts = parse_waypoints(row["assistant"])
                if wpts is None:
                    skipped += 1
                    continue
                human_text = _reformat_traj_prompt(row["human"])
            else:
                wpts = torch.zeros((K_WAYPOINTS, 2), dtype=torch.float32)
                human_text = row["human"]
                qa_answer = row.get("qa_answer", row.get("assistant"))

            self.samples.append({
                "frames":    row["frames"],
                "human":     human_text,
                "assistant": row["assistant"],
                "dataset":   row["dataset"],
                "gt_wpts":   wpts,
                "task_type": task_type,
                "qa_answer": qa_answer,
            })

        logger.info(
            f"Stage2Dataset: kept {len(self.samples):,} samples "
            f"(skipped {skipped:,} trajectory samples with unparseable waypoints)."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # ── Build chat messages (Qwen multimodal format) ──────────────────
        frames: List[Image.Image] = s["frames"]

        # Ensure 448×448 RGB (already done during materialisation, but guard)
        frames = [
            f.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            if f.size != (IMAGE_SIZE, IMAGE_SIZE) or f.mode != "RGB" else f
            for f in frames
        ]

        # Qwen expects a single image or a list for video-style multi-frame
        if len(frames) == 1:
            content = [
                {"type": "image", "image": frames[0]},
                {"type": "text",  "text":  s["human"]},
            ]
        else:
            # Multiple frames → treat as video sequence
            content = [
                {"type": "video", "video": frames},
                {"type": "text",  "text":  s["human"]},
            ]

        messages = [{"role": "user", "content": content}]

        # ── Apply chat template (text only, no tokenization yet) ──────────
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # ── Tokenize with processor (handles image patches) ───────────────
        inputs = self.processor(
            text=[text],
            images=frames if len(frames) == 1 else None,
            videos=frames if len(frames) > 1  else None,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,   # collate_fn handles padding
        )

        return {
            "input_ids":      inputs["input_ids"].squeeze(0),          # [seq]
            "attention_mask": inputs["attention_mask"].squeeze(0),     # [seq]
            "pixel_values":   inputs.get("pixel_values"),              # [patches,C,H,W] or None
            "image_grid_thw": inputs.get("image_grid_thw"),            # [n_imgs, 3] or None
            "gt_waypoints":   s["gt_wpts"],                            # [K, 2]
            "task_type":      s["task_type"],
            "qa_answer":      s["qa_answer"],
        }


# ── Collate function ──────────────────────────────────────────────────────────

def collate_stage2_batch(samples: List[dict]) -> dict:
    """
    Pad variable-length sequences to the maximum length in the batch.
    Stacks pixel_values and image_grid_thw if present.
    """
    # Pad input_ids and attention_mask (left-pad to max sequence length)
    max_len = max(s["input_ids"].shape[0] for s in samples)

    input_ids_padded  = torch.zeros(len(samples), max_len, dtype=torch.long)
    attn_mask_padded  = torch.zeros(len(samples), max_len, dtype=torch.long)

    for i, s in enumerate(samples):
        seq_len = s["input_ids"].shape[0]
        # Left-pad with zeros (pad token id = 0 for Qwen)
        input_ids_padded[i, max_len - seq_len:]  = s["input_ids"]
        attn_mask_padded[i, max_len - seq_len:]  = s["attention_mask"]

    # Stack waypoints [B, K, 2]
    gt_waypoints = torch.stack([s["gt_waypoints"] for s in samples], dim=0)

    # pixel_values: concatenate along batch dimension (Qwen packs patches)
    pv_list = [s["pixel_values"] for s in samples if s["pixel_values"] is not None]
    pixel_values = torch.cat(pv_list, dim=0) if pv_list else None

    # image_grid_thw: concatenate — each row is one image's [T, H, W]
    gt_list = [s["image_grid_thw"] for s in samples if s["image_grid_thw"] is not None]
    image_grid_thw = torch.cat(gt_list, dim=0) if gt_list else None

    task_types = [s["task_type"] for s in samples]
    qa_answers = [s["qa_answer"] for s in samples]

    return {
        "input_ids":      input_ids_padded,
        "attention_mask": attn_mask_padded,
        "pixel_values":   pixel_values,
        "image_grid_thw": image_grid_thw,
        "gt_waypoints":   gt_waypoints,
        # ground_truth dict stays on CPU — used by reward functions
        "ground_truth":   {
            "gt_waypoints": gt_waypoints.clone(),
            "task_type": task_types,
            "qa_answer": qa_answers,
        },
    }


# ── Factory ───────────────────────────────────────────────────────────────────

def build_stage2_dataloader(
    processor,
    hf_repo:      str = "shreethar/FYP-Stage2-dataset",
    split:        str = "train",
    batch_size:   int = 4,
    num_workers:  int = 2,
    max_length:   int = 1024,
    shuffle:      bool = True,
    hf_split=None,        # pass a pre-loaded split to skip download
    subset_ratio: float = 1.0, # Optionally use a percentage of the dataset (e.g. 0.15 for 15%)
) -> DataLoader:
    """
    Build the Stage 2 DataLoader.

    Parameters
    ----------
    processor   : AutoProcessor for Qwen3.5-4B (handles vision + text).
    hf_repo     : HuggingFace dataset repo ID.
    split       : Which split to load ("train" or "test").
    batch_size  : Samples per GPU step.
    num_workers : DataLoader worker processes (set 0 for debugging).
    max_length  : Max token length — longer prompts truncated.
    shuffle     : Shuffle the dataset each epoch.
    hf_split    : Pass a pre-loaded HF split to skip re-downloading.

    Returns
    -------
    DataLoader that yields batches compatible with train_stage2.py.

    Example
    -------
        from transformers import AutoProcessor
        from stage2_dataloader import build_stage2_dataloader

        processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B",
                                                   trust_remote_code=True)
        loader = build_stage2_dataloader(
            processor=processor,
            hf_repo="shreethar/FYP-Stage2-dataset",
            split="train",
            batch_size=4,
        )
    """
    if hf_split is None:
        from datasets import load_dataset
        logger.info(f"Loading HF dataset directly via Parquet: {hf_repo} [{split}] …")
        try:
            # Explicitly load just the parquet files for this split, ignoring dataset metadata
            # which might complain if other splits (e.g. 'train') are missing.
            data_files = {split: f"hf://datasets/{hf_repo}/data/{split}-*.parquet"}
            hf_split = load_dataset("parquet", data_files=data_files, split=split)
        except Exception as e:
            logger.warning(f"Failed to load via parquet explicitly: {e}. Falling back to default load_dataset...")
            hf_split = load_dataset(hf_repo, split=split)

    if subset_ratio < 1.0:
        subset_size = int(len(hf_split) * subset_ratio)
        logger.info(f"Subsetting dataset to {subset_ratio*100:.1f}% ({subset_size} samples)")
        # Make sure to shuffle the subset selection so we don't just take the first N
        # Seed ensures reproducibility if needed, but here we just take a random slice
        hf_split = hf_split.shuffle(seed=42).select(range(subset_size))

    dataset = Stage2Dataset(hf_split, processor=processor, max_length=max_length)

    if len(dataset) == 0:
        raise ValueError(
            f"Stage2Dataset is empty after filtering. "
            f"Check that {hf_repo} [{split}] contains records with type='trajectory' "
            f"and parseable waypoints in the 'assistant' column."
        )

    logger.info(
        f"Building DataLoader: {len(dataset):,} samples, "
        f"batch_size={batch_size}, num_workers={num_workers}"
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_stage2_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,     # avoids variable-batch issues with GRPO
    )
