# ThinkFlow-VLA — Stage 2: Teacher-Student Distillation

## Overview

Three models train jointly:

| Model | Base | Role |
|---|---|---|
| Teacher (F_θT) | Qwen2.5-VL-4B + LoRA r=64 | GRPO policy, generates CoT text traces |
| Student (F_θ)  | Qwen2.5-VL-4B + LoRA r=64 | Generates M=6 continuous latent vectors + K=5 spatial tokens |
| Verbalizer (Vψ)| Qwen3-0.6B + LoRA r=32 + per-layer CA | Translates Student latents → text for DPO gradient signal |

Teacher and Student are initialised from the **same Stage 1 checkpoint**.

---

## Training Phases

### Warm-up (Steps 0 – 3000)
```
Teacher : GRPO rollouts → score → update → extract h_T
Verbalizer: LM loss on τ+ (latents DETACHED)  → verbalizer_optimizer
Student : L_distill + L_ans + L_spatial       → student_optimizer
```
Verbalizer learns to read whatever the Student currently produces.
Student learns spatial grounding via distillation and waypoint supervision only.

### Frozen Verbalizer (Steps 3000 – 4500)
```
Teacher : GRPO (continues)
Verbalizer: FROZEN — no parameter updates
Student : L_verb + L_distill + L_ans + L_spatial → student_optimizer
```
DPO gradient flows through frozen Verbalizer CA blocks → Student latents.
Student learns to produce latents that naturally decode as τ+ reasoning.

---

## Loss Functions

| Loss | Formula | λ | Purpose |
|---|---|---|---|
| L_verb | DPO(τ+, τ−) via Verbalizer | 1.0 | High-level reasoning alignment |
| L_distill | MSE(h_S, h_T) | 1.0 | Spatial hidden state alignment |
| L_ans | MSE(pred_waypoints, gt_waypoints) | 1.0 | Physical waypoint grounding |
| L_spatial | −CosSim(MLP(x_V), VGGT(I)) | 0.1 | Visual feature alignment |

---

## Reward Function

```
r = 0.9 * r_visual + 0.1 * r_format

r_visual = 0.5 * r_goal + 0.5 * r_traj
r_goal   = 0.5 * (f(p1,p̂1) + f(pK,p̂K)),  f(p,p') = max(0, 1 − ‖p−p'‖²₂)
r_traj   = max(0, 1 − DTW(τ, τ̂))
r_format = 1.0 (exact structure) | ROUGE avg (QA tasks)
```

---

## File Map

```
stage2/
├── models/
│   ├── latent_student.py   — Student VLM: latent loop + spatial tokens + SpatialMLP
│   ├── verbalizer.py       — 0.6B LM with per-layer CA blocks + DPO/LM losses
│   └── spatial_forcing.py  — Frozen DINOv2/VGGT extractor + ProjectionMLP
├── training/
│   ├── grpo_teacher.py     — Teacher: rollouts, scoring, GRPO update, h_T extraction
│   ├── student_losses.py   — All four Student loss terms, phase-aware
│   └── train_stage2.py     — Main loop, three optimizers, checkpointing
├── rewards/
│   ├── action_reward.py    — r_visual: goal reward + DTW trajectory reward
│   └── qa_reward.py        — r_format: structural check + ROUGE for QA tasks
└── requirements.txt
```

---

## Launch

```bash
# Install dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Run Stage 2
python training/train_stage2.py \
    --stage1_ckpt   checkpoints/stage1 \
    --output_dir    checkpoints/stage2 \
    --total_steps   4500 \
    --answer_token_id <ID_FROM_TOKENIZER>

# Resume from checkpoint
python training/train_stage2.py \
    --stage1_ckpt   checkpoints/stage1 \
    --output_dir    checkpoints/stage2 \
    --resume_from   checkpoints/stage2/step_003000 \
    --total_steps   4500 \
    --answer_token_id <ID_FROM_TOKENIZER>
```

---

## Key Implementation Notes

**Latent loop** — Student bypasses the vocabulary entirely for M=6 steps.
`z_m = last_hidden_state[:, 0, :]` fed directly as `inputs_embeds` for step m+1.

**h_T extraction timing** — Teacher's `<ans>` hidden state is extracted in a
SEPARATE forward pass AFTER the GRPO optimizer step. This matches Algorithm 1's
sequential ordering and ensures h_T reflects post-update weights.

**Gradient flow in frozen phase** — Verbalizer parameters are frozen but
`latents` is NOT detached. DPO loss backward deposits gradients at `latents`
which propagates into Student LoRA weights through the CA computation graph.

**VGGT checkpoint** — `VGGTExtractor` in `spatial_forcing.py` has a TODO
comment at the output parsing step. Update `extract()` once the checkpoint
is confirmed and its output format is known.