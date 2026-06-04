#!/usr/bin/env python3
"""
push_from_cache.py — Recovery script.

Phase 2 (materialization) already completed. HF stores the result as
Arrow shards (NOT in save_to_disk format), so we load them directly
via load_dataset("arrow", ...), then run Phase 3 (split) + Phase 4 (push).

Usage:
    python data/push_from_cache.py --repo_id YOUR_USERNAME/vla-stage1-subset
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from datasets import load_dataset, DatasetDict, Features, Value, Sequence
from datasets import Image as HFImage

CACHE_DIR = (
    Path.home()
    / ".cache/huggingface/datasets/generator/default-67cba23759e52e8a/0.0.0"
)

# Must match the features used when the dataset was originally built
HF_FEATURES = Features({
    "dataset":    Value("string"),
    "type":       Value("string"),
    "human":      Value("string"),
    "assistant":  Value("string"),
    "split":      Value("string"),
    "frames":     Sequence(HFImage()),
    "media_type": Value("string"),
})


def main():
    p = argparse.ArgumentParser(description="Push cached VLA subset to HuggingFace Hub")
    p.add_argument("--repo_id", required=True,
                   help="HF repo (e.g. your-username/vla-stage1-subset)")
    p.add_argument("--public", action="store_true",
                   help="Make the repo public (default: private)")
    args = p.parse_args()

    private = not args.public

    # ── Load cached Arrow shards ──────────────────────────────────────────────
    arrow_files = sorted(str(f) for f in CACHE_DIR.glob("generator-train-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No arrow shards found in:\n  {CACHE_DIR}")

    print(f"📂  Loading {len(arrow_files)} Arrow shards from:\n    {CACHE_DIR}\n")
    flat_ds = load_dataset(
        "arrow",
        data_files={"train": arrow_files},
        features=HF_FEATURES,
        split="train",
    )
    print(f"  Loaded: {len(flat_ds):,} rows")
    print(f"  Columns: {flat_ds.column_names}\n")

    # ── Phase 3: Split (85 / 15) ──────────────────────────────────────────────
    print("📊  Phase 3 — Splitting dataset (85 / 15)")
    splits = {}
    for split_name in ("train", "test"):
        split_ds = flat_ds.filter(
            lambda rows, s=split_name: [x == s for x in rows["split"]],
            batched=True,
        )
        split_ds = split_ds.remove_columns("split")
        splits[split_name] = split_ds

        # Column-level access — no image decoding, no PIL crashes
        media_types = split_ds["media_type"]
        n_img = media_types.count("image")
        n_vid  = len(media_types) - n_img
        print(f"  {split_name:5s}: {len(split_ds):>8,}  "
              f"(image={n_img:,}  video={n_vid:,})")

    dataset_dict = DatasetDict(splits)
    print()

    # ── Phase 4: Push ─────────────────────────────────────────────────────────
    print(f"🚀  Phase 4 — Pushing to hub: {args.repo_id}  (private={private})")
    dataset_dict.push_to_hub(args.repo_id, private=private)
    print("✅  Done!\n")


if __name__ == "__main__":
    main()

