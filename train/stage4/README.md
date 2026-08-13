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

The default source is
`allenai/MolmoAct-Midtraining-Mixture[molmoact_tabletop_primary]`, loaded in
streaming mode. Rows are processed in this order:

1. reject null/malformed annotations and annotations with fewer than five pairs;
2. reject rows without both `primary` and `wrist` images;
3. extract only the task name from the first human `The task is ...` sentence;
4. retain each valid row with probability `--sample_ratio` (default `0.1`);
5. normalize the first five coordinates from `[1,256]` to `[0,1]` using
   `(coordinate - 1) / 255`.

Sampling is seeded and reproducible for a fixed worker/distributed setup. It is
a Bernoulli sample of approximately 10% of usable rows; computing an exact 10%
would require a full counting pass over this very large dataset.

The students receive only the resized `primary` image. VGGT receives exactly
`[primary, wrist]` jointly and must return `[B,2,N,D]`; only `features[:,0]` is
used as the Spatial Forcing target. The view-0 representation has nevertheless
been enriched by VGGT's joint/global cross-view processing.

Qwen/VGGT correspondence is metadata-driven. Qwen's target height and width are
read from `image_grid_thw` and divided by its actual `spatial_merge_size`. The
single VGGT primary-view patch map is then bilinearly resized to that explicit
grid. Token-count factorization is not used by training.

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
  --output_dir checkpoints/stage4 \
  --batch_size 1 \
  --max_steps 5000
```

This uses the requested defaults:

```text
checkpoint    shreethar/LatentStudent-ckpt-400
dataset       allenai/MolmoAct-Midtraining-Mixture
config        molmoact_tabletop_primary
sample ratio  0.1 after validation
layers        Qwen 8 / VGGT 8 (zero-based)
loss weights  alpha=1.0, beta=1.0, gamma=0.5
```

To resume, retain the original Stage 2 checkpoint as
`--student_checkpoint` and pass the Stage 4 checkpoint separately:

```bash
python train/stage4/train_stage4.py \
  --resume_from checkpoints/stage4/step_001000 \
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
