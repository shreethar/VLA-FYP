#!/usr/bin/env python3
"""
build_hf_subset.py — Pre-sample a compact VLA subset and push to HuggingFace Hub.

Run ONCE on a machine where the full datasets are downloaded. The resulting
HF dataset (~3-8 GB) can then be pulled on any cloud instance with a single
`load_dataset()` call — no need to store the full ~100+ GB sources.

Usage (CLI):
    python data/build_hf_subset.py --repo_id YOUR_USER/vla-stage2-subset

Usage (Python):
    from datasets import load_dataset
    molmoact   = load_dataset("allenai/MolmoAct-Pretraining-Mixture", "auxiliary_trace")["train"]
    pixmocap   = load_dataset("allenai/pixmo-cap")["train"]
    pixmoama   = load_dataset("allenai/pixmo-ask-model-anything")["train"]
    pixmocapqa = load_dataset("allenai/pixmo-cap-qa")["train"]

    from build_hf_subset import build_and_push
    build_and_push("YOUR_USER/vla-stage2-subset",
                   molmoact_ds=molmoact, pixmocap_ds=pixmocap,
                   pixmoama_ds=pixmoama, pixmocapqa_ds=pixmocapqa)

Subset target counts (total ~52.5 K):
    MolmoAct               10 000
    ShareRobot Affordance  all  (~6.5 K)
    ShareRobot Planning    10 000
    RoboVQA                10 000
    Pixmo Cap               2 000
    Pixmo AMA               2 000
    Pixmo Cap-QA            2 000
    RoboFAC                10 000
"""

import sys
from pathlib import Path

# Ensure sibling imports work when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import Dataset, DatasetDict, Features, Value, Sequence
from datasets import Image as HFImage

from stage_1_datasets_static import (
    build_molmoact_records,
    build_sharerobot_affordance_records,
    build_sharerobot_planning_records,
    build_robovqa_records,
    build_pixmocap_records,
    build_pixmoama_records,
    build_pixmocapqa_records,
    build_robofac_records,
    VLAStaticDataset,
    _get_split,
    _keep,
)

# ── Target subset sizes ─────────────────────────────────────────────────────
SUBSET_SIZES = {
    "molmoact":              10_000,
    "sharerobot_affordance": None,     # keep all (~6.5 K)
    "sharerobot_planning":   10_000,
    "robovqa":               10_000,
    "pixmocap":               2_000,
    "pixmoama":               2_000,
    "pixmocapqa":             2_000,
    "robofac":               10_000,
}


# ── Phase 1: Build lightweight metadata records ─────────────────────────────

def _build_subset_records(molmoact_ds=None, pixmocap_ds=None,
                          pixmoama_ds=None, pixmocapqa_ds=None):
    """Call each record builder with reduced sample counts."""
    all_records: list[dict] = []
    hf_cache: dict = {}

    # MolmoAct (HF)
    if molmoact_ds is not None:
        hf_cache["molmoact"] = molmoact_ds
        all_records += build_molmoact_records(molmoact_ds,
                                              n_samples=SUBSET_SIZES["molmoact"])
    else:
        print("  ⚠  MolmoAct HF dataset not provided — skipping")

    # ShareRobot (local)
    all_records += build_sharerobot_affordance_records()
    all_records += build_sharerobot_planning_records(
        n_samples=SUBSET_SIZES["sharerobot_planning"])

    # RoboVQA (local TFRecords)
    all_records += build_robovqa_records(n_samples=SUBSET_SIZES["robovqa"])

    # Pixmo (HF)
    if pixmocap_ds is not None:
        all_records += build_pixmocap_records(pixmocap_ds,
                                              n_samples=SUBSET_SIZES["pixmocap"])
    else:
        print("  ⚠  Pixmo Cap not provided — skipping")

    if pixmoama_ds is not None:
        all_records += build_pixmoama_records(pixmoama_ds,
                                              n_samples=SUBSET_SIZES["pixmoama"])
    else:
        print("  ⚠  Pixmo AMA not provided — skipping")

    if pixmocapqa_ds is not None:
        all_records += build_pixmocapqa_records(pixmocapqa_ds,
                                                n_samples=SUBSET_SIZES["pixmocapqa"])
    else:
        print("  ⚠  Pixmo Cap-QA not provided — skipping")

    # RoboFAC (local) — builder has no n_samples param, sub-sample after
    robofac = build_robofac_records()
    target = SUBSET_SIZES["robofac"]
    if target and len(robofac) > target:
        robofac = [r for i, r in enumerate(robofac)
                   if _keep(i, len(robofac), target)]
        print(f"  RoboFAC (sub-sampled): {len(robofac):,} records")
    all_records += robofac

    return all_records, hf_cache


# ── Phase 2: Eagerly materialize media via generator ─────────────────────────

def _sample_generator(records, hf_cache):
    """
    Pure yielder — no prints here.
    HF calls this function multiple times internally (sharding/checkpointing),
    so any stateful printing inside would fire repeatedly. Stats are computed
    from the finished flat_ds after from_generator() returns.
    """
    loader = VLAStaticDataset(records, hf_cache)
    for rec in records:
        media = loader._load_media(rec)
        if media is None:
            continue
        frames     = media if isinstance(media, list) else [media]
        media_type = "image" if len(frames) == 1 else "video"
        yield {
            "dataset":    rec["dataset"],
            "type":       rec["type"],
            "human":      rec["human"],
            "assistant":  rec["assistant"],
            "split":      _get_split(rec),
            "frames":     frames,
            "media_type": media_type,
        }


def _print_materialization_stats(flat_ds, records):
    """Print per-dataset ok/fail summary using fast HF column reads."""
    from collections import Counter
    ok_counts    = Counter(flat_ds["dataset"])          # yielded (successful) rows
    total_counts = Counter(rec["dataset"] for rec in records)  # all attempted

    for ds_name in total_counts:   # preserve insertion order
        ok   = ok_counts.get(ds_name, 0)
        fail = total_counts[ds_name] - ok
        print(f"  ✓ {ds_name:<28s}  ok={ok:>6,}  fail={fail:>5,}  "
              f"total={total_counts[ds_name]:>6,}")

    total_ok   = len(flat_ds)
    total_fail = len(records) - total_ok
    print(f"\n  Materialization done: {total_ok:,} ok, {total_fail:,} failed")


# ── Phase 3 & 4: Build DatasetDict and push ─────────────────────────────────

HF_FEATURES = Features({
    "dataset":    Value("string"),
    "type":       Value("string"),
    "human":      Value("string"),
    "assistant":  Value("string"),
    "split":      Value("string"),
    "frames":     Sequence(HFImage()),
    "media_type": Value("string"),   # "image" (1 frame) | "video" (>1 frames)
})


def build_and_push(
    repo_id: str,
    molmoact_ds=None,
    pixmocap_ds=None,
    pixmoama_ds=None,
    pixmocapqa_ds=None,
    private: bool = True,
):
    """End-to-end: build records → materialize media → split → push to HF Hub."""
    print("=" * 60)
    print("  Building VLA Stage 2 Subset")
    print("=" * 60)

    # 1. Metadata
    print("\n📋  Phase 1 — Building metadata records")
    records, hf_cache = _build_subset_records(
        molmoact_ds, pixmocap_ds, pixmoama_ds, pixmocapqa_ds)
    print(f"  Total records to materialize: {len(records):,}\n")

    # 2. Materialize into a single flat HF Dataset
    print("📸  Phase 2 — Materializing images / video frames")
    flat_ds = Dataset.from_generator(
        _sample_generator,
        gen_kwargs={"records": records, "hf_cache": hf_cache},
        features=HF_FEATURES,
    )
    _print_materialization_stats(flat_ds, records)
    print(f"\n  Flat dataset size: {len(flat_ds):,} rows\n")

    # 3. Split into train / test
    print("📊  Phase 3 — Splitting dataset (85 / 15)")
    splits = {}
    for split_name in ("train", "test"):
        split_ds = flat_ds.filter(
            lambda rows, s=split_name: [x == s for x in rows["split"]],
            batched=True,
        )
        split_ds = split_ds.remove_columns("split")
        splits[split_name] = split_ds
        # Use column-level access — avoids decoding the image bytes entirely
        media_types = split_ds["media_type"]   # plain list[str], no PIL involved
        n_img = media_types.count("image")
        n_vid = len(media_types) - n_img
        print(f"  {split_name:5s}: {len(split_ds):>8,}  "
              f"(image={n_img:,}  video={n_vid:,})")

    dataset_dict = DatasetDict(splits)

    # 4. Push
    print(f"\n🚀  Phase 4 — Pushing to hub: {repo_id}  (private={private})")
    dataset_dict.push_to_hub(repo_id, private=private)
    print("✅  Done!\n")

    return dataset_dict


# ── CLI entrypoint ───────────────────────────────────────────────────────────

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Build a compact VLA subset and push to HuggingFace Hub")
    p.add_argument("--repo_id", required=True,
                   help="HF repo (e.g. your-username/vla-stage2-subset)")
    p.add_argument("--private", action="store_true", default=True,
                   help="Make the repo private (default)")
    p.add_argument("--public", action="store_true",
                   help="Make the repo public instead")
    p.add_argument("--skip_hf", action="store_true",
                   help="Skip HF datasets (MolmoAct + Pixmo) — "
                        "only process local datasets")
    args = p.parse_args()

    private = not args.public
    molmoact = pixmocap = pixmoama = pixmocapqa = None

    if not args.skip_hf:
        from datasets import load_dataset

        print("Downloading HuggingFace source datasets …")
        print("  Loading MolmoAct …")
        molmoact = load_dataset(
            "allenai/MolmoAct-Pretraining-Mixture", "auxiliary_trace"
        )["train"]
        print("  Loading Pixmo Cap …")
        pixmocap = load_dataset("allenai/pixmo-cap")["train"]
        print("  Loading Pixmo AMA …")
        pixmoama = load_dataset("allenai/pixmo-ask-model-anything")["train"]
        print("  Loading Pixmo Cap-QA …")
        pixmocapqa = load_dataset("allenai/pixmo-cap-qa")["train"]

    build_and_push(
        repo_id=args.repo_id,
        molmoact_ds=molmoact,
        pixmocap_ds=pixmocap,
        pixmoama_ds=pixmoama,
        pixmocapqa_ds=pixmocapqa,
        private=private,
    )


if __name__ == "__main__":
    main()
