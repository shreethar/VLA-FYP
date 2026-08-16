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
L_SF       = mean_b,n(1 - cos(P(H_visual,SF[b,n]), G_VGGT[b,n]))
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
  --alpha 1 \
  --beta 3 \
  --gamma 0.025 \
  --eval_steps 500 \
  --eval_batches 50 \
  --early_stopping_patience 5 \
  --early_stopping_min_delta 1e-4 \
  --wandb_run_name stage4-sf-partial-a1-b3-g0025 \
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

## Weights & Biases metrics

W&B logging is enabled by default and uses project `reasonflow-vla` with run
name `stage4-spatial-forcing`. Authenticate once on the training machine:

```bash
wandb login
```

Stage 4 logs the following every `--log_steps` optimizer steps:

- total and raw latent/waypoint/Spatial Forcing losses;
- alpha/beta/gamma-weighted contribution from each loss;
- latent cosine preservation and VGGT spatial cosine alignment;
- relative layer-8 visual K/V drift between B3 and frozen B2 during validation;
- normalized and pixel-space waypoint MAE, normalized RMSE, and prediction/
  target distribution statistics;
- pre-clipping global and per-optimizer-group gradient norms;
- all four parameter-group learning rates and whether clipping occurred;
- optimizer progress, samples processed, step time, throughput, and CUDA
  allocated/reserved/peak memory.

Run summaries retain the best logged total/waypoint/latent losses, best spatial
cosine, latest checkpoint path, final step, and sample count. Model checkpoint
directories are not uploaded as W&B artifacts, avoiding large background
uploads. The W&B run ID is stored in `stage4_config.json` and automatically
reused with `--resume_from`.

Useful controls:

```bash
--wandb_project reasonflow-vla
--wandb_run_name stage4-sf-partial
--wandb_tags stage4,spatial-forcing,molmoact,partial
--wandb_mode offline      # retain local W&B logs without network sync
--no_wandb                # explicitly disable tracking
```

## Validation and early stopping

Evaluation is enabled by default. Every 500 optimizer steps, Stage 4 evaluates
a fixed, deterministic prefix of 50 batches from the materialized
`validation/` partition (800 samples with batch size 16). The student and SF
projector switch to evaluation mode, so dropout is disabled and BatchNorm uses
its training running statistics; all three models run without gradients.

Validation reports the same raw and weighted losses, cosine metrics, and
trajectory errors as training under `validation/*` W&B keys. The monitored
quantity is the weighted validation objective:

```text
validation/loss/total = alpha * L_latent
                      + beta  * L_waypoint
                      + gamma * L_spatial_forcing
```

Validation additionally reports the relative layer-8 visual key/value drift:

```text
validation/representation/relative_kv_drift
    = mean_sample(
        ||KV_B3^8 - KV_B2^8||_2 / max(||KV_B2^8||_2, epsilon)
      )
```

Qwen3.5 layer 8 is a GatedDeltaNet linear-attention layer. Here `KV^8` means
the actual K and V activation slices produced by its combined `in_proj_qkv`,
restricted to primary-image visual-token positions; it does not mean a
transformer KV cache. B2 is frozen and supplies the denominator/reference.

An improvement must exceed `--early_stopping_min_delta` (default `1e-4`). A
new best checkpoint is saved immediately and recorded in
`<output_dir>/best_checkpoint.json`. Training stops after five consecutive
validation checks without sufficient improvement, equivalent to at most 2,500
non-improving training steps with the defaults. The best loss, patience count,
sample count, and best-checkpoint path are included in every Stage 4 checkpoint
and restored by `--resume_from`.

Controls:

```bash
--eval_steps 500
--eval_batches 50
--eval_batch_size 16
--early_stopping_patience 5
--early_stopping_min_delta 1e-4
--no_early_stopping       # keep validation, never stop early
--no_eval                 # disable validation and early stopping
```

## Optimization recipe

The trainable Qwen parameters are LoRA adapters and are separated by decoder
layer. Unrecognized trainable parameters abort startup rather than receiving an
accidental learning rate.

```text
optimizer                         AdamW
betas                             (0.9, 0.95)
weight decay                      0.01
Qwen decoder layers 0-7           5e-7
Qwen decoder layers 8-31          5e-8
waypoint head (SpatialMLP)         5e-7
SF projector + BatchNorm           1e-5
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
loss weights  alpha=1.0, beta=3.0, gamma=0.025
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

## Merge the best Stage 4 student and upload it

The merge utility accepts either a concrete `step_*` checkpoint or the Stage 4
run directory. When given the run directory it follows `best_checkpoint.json`,
so an early-stopped run publishes the best validation checkpoint rather than
the final checkpoint by accident.

Authenticate once on the training machine, then run:

```bash
hf auth login

python train/stage4/merge_and_upload.py \
  --stage4_checkpoint checkpoints/stage4_partial_run_2 \
  --base_model shreethar/LatentStudent-ckpt-400 \
  --repo_id shreethar/Latent-Student-Spatial-Forcing
```

By default the merge runs on CPU in bfloat16 and writes its upload bundle to
`checkpoints/Latent-Student-Spatial-Forcing-merged`. This requires enough CPU
RAM for the full Stage 2 model and merge operation. Use `--device_map auto` if
the training machine has enough accelerator memory. To inspect the local
bundle before publishing, add `--no_upload`.

The uploaded repository contains the merged VLM, processor/tokenizer, and the
Stage 4 `spatial_parameters.pt` holding the frozen five spatial slots and
updated waypoint head. VGGT and the SF projector are training-only and are not
included in the inference path.

## Compare Stage 4 against the previous models

The Stage 4 evaluator reproduces `train/stage2/evaluate_all.py` on the same
`shreethar/FYP-Stage2-dataset` train split, sampling 50 rows from the first
8,000 with NumPy seed 42. It compares the dataset ground truth, Stage 1,
textual-thinking teacher, Latent Student checkpoint 400, and the merged Spatial
Forcing student:

```bash
python train/stage4/evaluate_all.py \
  --spatial_forcing_model shreethar/Latent-Student-Spatial-Forcing \
  --output_dir evaluation_stage4
```

Models run sequentially to avoid placing four large models on `cuda:0` at the
same time. The evaluator retains the old 0-1000 pointwise L2 and DTW metrics
and also calculates the actual Stage 4 waypoint objective:

```text
mean_i ||(prediction_i - ground_truth_i) / 1000||_2^2
```

Aggregate metrics, per-sample predictions, timings, selected dataset indices,
and provenance are saved in `evaluation_stage4/evaluation_results.json`.
Comparison grids are written to the same directory.

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
