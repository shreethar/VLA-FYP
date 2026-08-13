# Stage 4 - Spatial Forcing

Stage 4 fine-tunes the learned Stage 2 latent student with three models:

- frozen VGGT geometry teacher;
- frozen Stage 2 latent-student reference;
- trainable Stage 2 latent student.

The trainable student's five learned spatial-token embeddings are restored from
the Stage 2 checkpoint and then frozen. Its LoRA parameters, SpatialMLP, and the
new Spatial Forcing alignment projector remain trainable.

## Objective

```text
L = alpha * L_latent + beta * L_waypoint + gamma * L_SF

L_latent   = mean_b,m(1 - cos(z_SF[b,m], stopgrad(z_ref[b,m])))
L_waypoint = mean_b,i(||predicted[b,i] - ground_truth[b,i]||_2^2)
L_SF       = -mean_b,n(cos(P(H_visual,SF[b,n]), G_VGGT[b,n]))
```

Both the trainable and reference students run the same six-step continuous
latent loop. VGGT runs under `no_grad`, and the reference student is fully
frozen and held in evaluation mode.

The defaults align Qwen visual features from layer index 8 with VGGT aggregator
features from layer index 8. Both indices are zero-based and independently
configurable. Official VGGT does not cache layer 8 by default; the extractor
enables that cache entry explicitly.

## Installation

Install the repository's normal dependencies, followed by VGGT:

```bash
pip install -r train/stage4/requirements.txt
```

## Dataset contract

The source is
`allenai/MolmoAct-Midtraining-Mixture[molmoact_tabletop_primary]`. The
recommended training path is to materialize the selected subset once and then
stream local Parquet shards. Direct Hugging Face streaming remains available
as a fallback. Rows are processed in this order:

1. reject null/malformed annotations and annotations with fewer than five pairs;
2. reject rows without both `primary` and `wrist` images;
3. extract only the task name from the first human `The task is ...` sentence;
4. content-hash the task, annotation, and primary image;
5. retain approximately 10% by a seeded hash threshold;
6. assign retained rows by a second hash to train/validation/test with
   probabilities 70%/15%/15%;
7. normalize the first five coordinates from `[1,256]` to `[0,1]` using
   `(coordinate - 1) / 255`.

Sampling and partition membership are stable across worker counts and runs.
Exact duplicate planner samples receive the same fingerprint, so they cannot
leak between partitions. Both the 10% sample and 70/15/15 proportions are
approximate: exact counts would require a full preliminary scan/materialization.
The resulting training partition therefore contains approximately 7% of all
usable rows; validation and test each contain approximately 1.5%.

### One-time local materialization

Choose a fast local disk with enough free space, then run:

```bash
python train/stage4/materialize_molmoact.py \
  --output_dir /path/to/fast-local-disk/molmoact_stage4_10pct
```

This makes one remote streaming pass, preserves the two compressed source
images, and writes:

```text
molmoact_stage4_10pct/
  manifest.json
  train/part-*.parquet
  validation/part-*.parquet
  test/part-*.parquet
```

The final disk footprint is data-dependent. The script logs the actual GiB and
records byte/row/shard counts in `manifest.json`. It writes to a sibling
`.incomplete` directory and only renames it to the requested output after a
complete source scan, so training cannot mistake a partial run for a complete
dataset.

Train from it with:

```bash
python train/stage4/train_stage4.py \
  --materialized_data_dir /path/to/fast-local-disk/molmoact_stage4_10pct \
  --output_dir checkpoints/stage4
```

The loader still uses iterable/streaming reads to keep RAM bounded, but every
read is from local Parquet: there are no Hugging Face dataset GET requests
during training. The manifest is checked against the requested sample ratio,
seed, and 70/15/15 split before training starts. If materialization is
interrupted, inspect or remove the `.incomplete` directory before retrying.

### Training from an interrupted materialization

Completed Parquet shards survive `Ctrl+C`: each was written to a temporary
file and atomically renamed. Only the final in-memory buffers are lost (at most
`rows_per_shard - 1`, or 255 by default, in each partition). To deliberately
use those completed shards, keep the `.incomplete` directory and opt in:

```bash
python train/stage4/train_stage4.py \
  --materialized_data_dir /path/to/molmoact_stage_10pct.incomplete \
  --allow_incomplete_materialized \
  --alpha 100 \
  --beta 100 \
  --gamma 25 \
  --output_dir checkpoints/stage4_partial
```

When constructing the data loader, the opt-in path reads every Parquet footer,
checks the required schema, reports the exact number of usable rows in the
selected partition, and ignores any unfinished `*.tmp` file. It does not
pretend that the scan covered the whole source dataset: the resulting training
set is the selected subset encountered before interruption.

The students receive only the resized `primary` image. VGGT receives exactly
`[primary, wrist]` jointly and must return `[B,2,N,D]`; only `features[:,0]` is
used as the Spatial Forcing target. The view-0 representation has nevertheless
been enriched by VGGT's joint/global cross-view processing.

Qwen/VGGT correspondence is metadata-driven. Qwen's target height and width are
read from `image_grid_thw` and divided by its actual `spatial_merge_size`. The
single VGGT primary-view patch map is then bilinearly resized to that explicit
grid. Token-count factorization is not used by training.

The SF projector applies `BatchNorm1d` to the concatenated valid visual tokens
from the complete optimizer batch, followed by `Linear -> GELU -> Linear` and
tokenwise cosine alignment.

## Optimization recipe

The trainable Qwen parameters are LoRA adapters and are separated by decoder
layer. Unrecognized trainable parameters abort startup rather than receiving an
accidental learning rate.

```text
optimizer                         AdamW
betas                             (0.9, 0.95)
weight decay                      0.01
Qwen decoder layers 0-7           1e-5
Qwen decoder layers 8-31          1e-6
waypoint head (SpatialMLP)         1e-5
SF projector + BatchNorm           1e-4
five learned spatial embeddings    frozen
scheduler                          cosine decay
warmup                             500 / 10,000 steps (5%)
optimizer batch size               16
```

## Checkpoint formats

`--student_checkpoint` can be:

- a Stage 2 checkpoint directory containing `student_lora/` and
  `training_state.pt`;
- a standalone PEFT adapter, with spatial parameters in its parent directory;
- a merged local/Hugging Face latent-student model with
  `spatial_parameters.pt`.

Loading fails if learned spatial parameters are missing. It never silently uses
random spatial slots.

## Example

```bash
python train/stage4/train_stage4.py \
  --materialized_data_dir /path/to/fast-local-disk/molmoact_stage4_10pct \
  --output_dir checkpoints/stage4
```

This uses the requested defaults:

```text
checkpoint    shreethar/LatentStudent-ckpt-400
dataset       local materialized MolmoAct Parquet shards
source        allenai/MolmoAct-Midtraining-Mixture[molmoact_tabletop_primary]
sample ratio  0.1 after validation
partitions     70/15/15 materialized hash split (training selects train)
layers        Qwen 8 / VGGT 8 (zero-based)
loss weights  alpha=1.0, beta=1.0, gamma=0.5
steps/batch    10000 / 16
```

To resume, retain the original Stage 2 checkpoint as
`--student_checkpoint` and pass the Stage 4 checkpoint separately:

```bash
python train/stage4/train_stage4.py \
  --resume_from checkpoints/stage4/step_001000 \
  --materialized_data_dir /path/to/fast-local-disk/molmoact_stage4_10pct \
  --output_dir checkpoints/stage4
```

The frozen reference continues to load from the original Stage 2 checkpoint,
while the trainable student, projector, optimizer, and scheduler resume from
Stage 4.

## Validate token correspondence first

Before training, run the correspondence inspector on several samples. It
compares Qwen's real `grid_thw`, placeholder count, vision-encoder output, and
layer-8 token count against VGGT's two-view patch grid. It verifies that only
the primary feature slice is resized to the explicit Qwen merged grid.

```bash
python train/stage4/inspect_spatial_correspondence.py \
  --sample_indices 0,1,2,10 \
  --student_layer 8 \
  --vggt_layer 8 \
  --output spatial_correspondence_report.json
```

Do not start Stage 4 when any sample reports
`DO_NOT_TRAIN_WITH_CURRENT_ALIGNMENT`. Send
`spatial_correspondence_report.json` back for review; in particular, inspect
the grid/count chain and the primary/wrist view ownership.
