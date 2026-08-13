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

`stage4_dataloader.py` currently accepts the same Hugging Face record schema as
Stage 2 and keeps trajectory examples only. It produces two visual paths:

- Qwen processor tensors (`pixel_values`, `image_grid_thw`, and video variants);
- raw `[0,1]` RGB images at 518x518 for VGGT.

For a new dataset, either adapt `Stage4Dataset` or call
`train(config, dataloader=your_loader)` with a loader yielding the same batch
keys. Ground-truth waypoints must be `[B, 5, 2]` normalized to `[0,1]`.

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
  --student_checkpoint checkpoints/stage2/step_004500 \
  --base_model_name shreethar/stage1_unsloth \
  --hf_repo YOUR_ORG/YOUR_TRAJECTORY_DATASET \
  --output_dir checkpoints/stage4 \
  --student_visual_layer 8 \
  --vggt_layer 8 \
  --alpha 1.0 \
  --beta 1.0 \
  --gamma 0.5
```

To resume, retain the original Stage 2 checkpoint as
`--student_checkpoint` and pass the Stage 4 checkpoint separately:

```bash
python train/stage4/train_stage4.py \
  --student_checkpoint checkpoints/stage2/step_004500 \
  --resume_from checkpoints/stage4/step_001000 \
  --base_model_name shreethar/stage1_unsloth \
  --hf_repo YOUR_ORG/YOUR_TRAJECTORY_DATASET
```

The frozen reference continues to load from the original Stage 2 checkpoint,
while the trainable student, projector, optimizer, and scheduler resume from
Stage 4.

## Validate token correspondence first

Before training, run the correspondence inspector on several image and video
samples. It compares Qwen's real `grid_thw`, placeholder count, vision-encoder
output, and layer-8 token count against VGGT's view/patch grid. It also compares
the current count-inferred resize with an explicit `(time, height, width)`
resize using a synthetic coordinate field.

```bash
python train/stage4/inspect_spatial_correspondence.py \
  --student_checkpoint checkpoints/stage2/step_004500 \
  --base_model_name shreethar/stage1_unsloth \
  --hf_repo YOUR_ORG/YOUR_TRAJECTORY_DATASET \
  --split train \
  --sample_indices 0,1,2,10 \
  --student_layer 8 \
  --vggt_layer 8 \
  --output spatial_correspondence_report.json
```

Do not start Stage 4 when any sample reports
`DO_NOT_TRAIN_WITH_CURRENT_ALIGNMENT`. Send
`spatial_correspondence_report.json` back for review; in particular, inspect
temporal view mismatches and normalized coordinate displacement rather than
accepting equal token counts alone.
