"""Materialize the filtered Stage 4 MolmoAct subset into local Parquet shards.

This performs exactly one streaming pass over the remote source. It preserves
the source's compressed primary/wrist image bytes, selects approximately 10%
of valid rows by the same deterministic content hash used by Stage 4, and
writes separate 70/15/15 train/validation/test directories. Training can then
stream the Parquet shards from local disk without dataset HTTP requests.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.stage4.stage4_dataloader import (
    DEFAULT_HF_CONFIG,
    DEFAULT_HF_REPO,
    DEFAULT_SPLIT_RATIOS,
    MATERIALIZED_FORMAT_VERSION,
    PARTITIONS,
    extract_task_name,
    is_sampled_fingerprint,
    molmoact_row_fingerprint,
    parse_molmoact_annotation,
    partition_for_fingerprint,
)

logger = logging.getLogger("materialize_molmoact")


def _image_bytes(value: Any) -> bytes:
    """Return original compressed bytes when available; encode only fallbacks."""
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            payload = bytes(value["bytes"])
            if payload:
                return payload
        if value.get("path"):
            return Path(value["path"]).read_bytes()
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return Path(value).read_bytes()
    if isinstance(value, np.ndarray):
        value = Image.fromarray(value)
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        value.convert("RGB").save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    raise TypeError(f"Unsupported image value: {type(value).__name__}")


def _annotation_json(annotation: Any) -> str:
    """Store the validated first five points in the source [1,256] scale."""
    normalized = parse_molmoact_annotation(annotation)
    if normalized is None:
        raise ValueError("annotation is not a valid five-waypoint trajectory")
    points = (normalized.mul(255.0).add(1.0)).tolist()
    return json.dumps(points, separators=(",", ":"))


class _ParquetShardWriter:
    def __init__(self, root: Path, rows_per_shard: int) -> None:
        import pyarrow as pa

        self.root = root
        self.rows_per_shard = rows_per_shard
        self.buffers: dict[str, list[dict]] = {name: [] for name in PARTITIONS}
        self.shard_counts: dict[str, int] = {name: 0 for name in PARTITIONS}
        self.row_counts: dict[str, int] = {name: 0 for name in PARTITIONS}
        self.schema = pa.schema(
            [
                pa.field("fingerprint", pa.string(), nullable=False),
                pa.field("task_name", pa.string(), nullable=False),
                pa.field("annotation", pa.string(), nullable=False),
                pa.field("primary", pa.binary(), nullable=False),
                pa.field("wrist", pa.binary(), nullable=False),
                pa.field("data_partition", pa.string(), nullable=False),
            ]
        )
        for partition in PARTITIONS:
            (root / partition).mkdir(parents=True, exist_ok=False)

    def add(self, partition: str, row: dict) -> None:
        buffer = self.buffers[partition]
        buffer.append(row)
        if len(buffer) >= self.rows_per_shard:
            self._flush(partition)

    def _flush(self, partition: str) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = self.buffers[partition]
        if not rows:
            return
        shard_index = self.shard_counts[partition]
        final_path = self.root / partition / f"part-{shard_index:05d}.parquet"
        temporary_path = final_path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(rows, schema=self.schema)
        pq.write_table(
            table,
            temporary_path,
            compression="zstd",
            compression_level=3,
            use_dictionary=["task_name", "data_partition"],
            row_group_size=len(rows),
        )
        os.replace(temporary_path, final_path)
        self.shard_counts[partition] += 1
        self.row_counts[partition] += len(rows)
        rows.clear()
        logger.info(
            "Wrote %s (%d rows)", final_path.relative_to(self.root), len(table)
        )

    def close(self) -> None:
        for partition in PARTITIONS:
            self._flush(partition)


def materialize(args: argparse.Namespace) -> Path:
    from datasets import Image as HFImage
    from datasets import load_dataset

    if args.rows_per_shard < 1:
        raise ValueError("rows_per_shard must be positive")
    if args.progress_rows < 1:
        raise ValueError("progress_rows must be positive")
    if not 0.0 < args.sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0,1]")
    split_ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    # Reuse the runtime validator for non-negative values and a unit sum.
    partition_for_fingerprint("validate-ratios", args.seed, split_ratios)

    output_dir = Path(args.output_dir).expanduser().resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    if staging_dir.exists():
        raise FileExistsError(
            "An incomplete materialization already exists. Inspect or remove it "
            f"before retrying: {staging_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()

    logger.info(
        "Streaming %s[%s] split=%s once",
        args.hf_repo,
        args.hf_config,
        args.split,
    )
    source = load_dataset(
        args.hf_repo,
        args.hf_config,
        split=args.split,
        streaming=True,
    )
    source = source.cast_column("primary", HFImage(decode=False))
    source = source.cast_column("wrist", HFImage(decode=False))

    writer = _ParquetShardWriter(staging_dir, args.rows_per_shard)
    stats: defaultdict[str, int] = defaultdict(int)

    for row_index, row in enumerate(source, start=1):
        stats["source_rows"] += 1
        if row_index % args.progress_rows == 0:
            logger.info(
                "Scanned %s source rows; valid=%s selected=%s",
                f"{row_index:,}",
                f"{stats['valid_rows']:,}",
                f"{stats['selected_rows']:,}",
            )
        annotation = row.get("annotation")
        if parse_molmoact_annotation(annotation) is None:
            stats["rejected_annotation"] += 1
            continue
        conversation = row.get("conversations", row.get("conversation"))
        task_name = extract_task_name(conversation)
        if task_name is None:
            stats["rejected_task"] += 1
            continue
        primary = row.get("primary")
        wrist = row.get("wrist", row.get("wrirst"))
        if primary is None or wrist is None:
            stats["rejected_images"] += 1
            continue

        stats["valid_rows"] += 1
        try:
            fingerprint = molmoact_row_fingerprint(row, task_name)
        except (OSError, TypeError, ValueError):
            stats["rejected_fingerprint"] += 1
            continue
        if not is_sampled_fingerprint(fingerprint, args.sample_ratio, args.seed):
            stats["not_sampled"] += 1
            continue
        partition = partition_for_fingerprint(
            fingerprint, args.seed, split_ratios
        )
        try:
            materialized_row = {
                "fingerprint": fingerprint,
                "task_name": task_name,
                "annotation": _annotation_json(annotation),
                "primary": _image_bytes(primary),
                "wrist": _image_bytes(wrist),
                "data_partition": partition,
            }
        except (OSError, TypeError, ValueError):
            stats["rejected_image_serialization"] += 1
            continue
        writer.add(partition, materialized_row)
        stats["selected_rows"] += 1
        stats[f"selected_{partition}"] += 1

    writer.close()
    empty_partitions = [
        partition for partition, count in writer.row_counts.items() if count == 0
    ]
    if empty_partitions:
        raise RuntimeError(
            "Materialization produced empty partitions; output remains incomplete: "
            + ", ".join(empty_partitions)
        )
    materialized_bytes = sum(
        path.stat().st_size for path in staging_dir.rglob("*.parquet")
    )
    manifest = {
        "format_version": MATERIALIZED_FORMAT_VERSION,
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": args.hf_repo,
        "source_config": args.hf_config,
        "source_split": args.split,
        "sample_ratio": args.sample_ratio,
        "seed": args.seed,
        "split_ratios": list(split_ratios),
        "rows_per_shard": args.rows_per_shard,
        "counts": dict(sorted(stats.items())),
        "partition_rows": writer.row_counts,
        "partition_shards": writer.shard_counts,
        "materialized_bytes": materialized_bytes,
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging_dir, output_dir)
    logger.info(
        "Materialization complete: %s | rows=%s | size=%.2f GiB",
        output_dir,
        writer.row_counts,
        materialized_bytes / 1024**3,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hf_repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf_config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample_ratio", type=float, default=0.1)
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_SPLIT_RATIOS[0])
    parser.add_argument(
        "--validation_ratio", type=float, default=DEFAULT_SPLIT_RATIOS[1]
    )
    parser.add_argument("--test_ratio", type=float, default=DEFAULT_SPLIT_RATIOS[2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows_per_shard", type=int, default=256)
    parser.add_argument("--progress_rows", type=int, default=5_000)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    materialize(parse_args())
