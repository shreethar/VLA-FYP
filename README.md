# ReasonFlow VLA

> **Final Year Project — Universiti Teknikal Malaysia Melaka (UTeM)**
> A multi-stage Vision-Language-Action system built on top of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (natively multimodal), drawing inspiration from [Fast Think, Then Act](https://arxiv.org/abs/2601.09708) (NVIDIA) with significant architectural extensions.

---

## Abstract

**ReasonFlow VLA** is a research project that implements a phased training pipeline to endow a natively multimodal 4B-parameter language model with robot manipulation capabilities. The system is designed around a **latent reasoning** paradigm: rather than generating action tokens verbosely in text space, an inner Student model produces a compressed sequence of continuous latent vectors that are subsequently decoded into physical actions by a Conditional Flow Matching (CFM) adapter. A recurrent memory module and a spatial-forcing objective ground the model's internal representations to the visual scene geometry, while a GRPO-trained Teacher model provides high-quality reasoning traces to guide distillation.

Key architectural contributions over the base paper:
- **CFM Action Expert** — replaces discrete action token regression with continuous flow matching
- **Goal-Delta conditioning** — injects a goal–current-state delta as a conditioning token into the first Cross-Attention layer
- **Recurrent Memory Module** — N=8 persistent query vectors `m_t` maintain episodic context across timesteps (inspired by [RememVLA](https://arxiv.org/abs/2505.07218))
- **Spatial Forcing** — a frozen visual encoder (VGGT/DINOv2) supervises the Student's spatial representations via cosine similarity loss
- **Two-phase Teacher-Student Distillation** — warm-up then frozen-Verbalizer phases enable DPO-gradient flow through a cross-attention Verbalizer (inspired by [SmolVLA](https://arxiv.org/abs/2506.01844))

---

## Training Pipeline

| Stage | Name | Status | Description |
|:-----:|------|:------:|-------------|
| **1** | Robot Grounding SFT | ✅ **Done** | Multi-source SFT to inject robot manipulation knowledge into the base VLM |
| **2** | GRPO Teacher · Student Distillation | 🔄 In Progress | Teacher generates CoT traces via GRPO; Student learns M=6 latent vectors aligned to Teacher's hidden states |
| **3** | Action Expert — CFM Adapter | 📋 Planned | Conditional Flow Matching head trained on Student latents → physical action tokens |
| **4** | Partial VLM Coupling · Spatial Forcing | 📋 Planned | Fine-tune selected VLM layers with spatial alignment loss (VGGT feature supervision) |
| **5** | LIBERO Evaluation · RL Fine-Tuning | ⚗️ Optional | Downstream task evaluation on LIBERO benchmark + optional RL fine-tuning via [RLInf](https://arxiv.org/abs/2505.01821) |

---

## Stage 1 — Robot Grounding SFT ✅

### Motivation
A general-purpose VLM has limited understanding of robot-specific concepts: end-effector trajectories, affordance regions, task planning in manipulation contexts, and failure analysis. Stage 1 performs supervised fine-tuning across eight curated datasets to establish this foundational knowledge before any RL or distillation is applied.

### Datasets

| Dataset | Task | Sampling | Source |
|---------|------|----------|--------|
| [MolmoAct](https://huggingface.co/datasets/allenai/MolmoAct-Pretraining-Mixture) | 2D trajectory prediction | 10% (200K) | HuggingFace |
| [RoboVQA](https://huggingface.co/datasets/google/robovqa) | Robot visual QA | 10% (100K) | TFRecord (local) |
| [RoboFAC](https://huggingface.co/datasets/RoboFAC/RoboFAC) | Failure analysis QA | 100% (~64K) | Local video |
| [ShareRobot Affordance](https://huggingface.co/datasets/ShareRobot/ShareRobot) | Affordance bbox prediction | 100% (~6.5K) | Local image |
| [ShareRobot Planning](https://huggingface.co/datasets/ShareRobot/ShareRobot) | Multi-step task planning | 10% (100K) | Local frames |
| [Pixmo Cap](https://huggingface.co/datasets/allenai/pixmo-cap) | Dense image captioning | 10% (50K) | HuggingFace |
| [Pixmo Cap-QA](https://huggingface.co/datasets/allenai/pixmo-cap-qa) | Caption-grounded QA | 10% (50K) | HuggingFace |
| [Pixmo AMA](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything) | Open-ended visual QA | 10% (50K) | HuggingFace |

> **Total:** ~560K samples. A compact 51K-sample pre-materialized subset is available at [`shreethar/FYP-Stage2-dataset`](https://huggingface.co/datasets/shreethar/FYP-Stage2-dataset) for cloud training.

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Framework | [Unsloth](https://github.com/unslothai/unsloth) |
| Base Model | Qwen3.5-4B (natively multimodal) |
| Learning Rate | 1e-5 |
| Batch Size | 1 |
| Gradient Accumulation | 8 (effective batch = 8) |
| Total Steps | 750K (~1 epoch over 600K samples) |
| Estimated Duration | ~10 days |

---

## Stage 2 — GRPO Teacher · Student Distillation 🔄

Three models train jointly from the same **Stage 1 checkpoint**:

| Model | Base | LoRA | Role |
|-------|------|------|------|
| Teacher `F_θT` | Qwen3.5-4B | r=64 | GRPO policy — generates CoT reasoning traces |
| Student `F_θ` | Qwen3.5-4B | r=64 | Produces M=6 continuous latent vectors + K=5 spatial tokens |
| Verbalizer `V_ψ` | Qwen3.5-0.8B | r=32 + per-layer CA | Decodes Student latents → text for DPO gradient signal |

### Two-Phase Training

**Phase 1 — Warm-up** (steps 0–3000): Verbalizer learns to interpret whatever latents the Student currently produces. Student learns spatial grounding from distillation and waypoint supervision alone.

**Phase 2 — Frozen Verbalizer** (steps 3000–4500): Verbalizer parameters are frozen; DPO loss gradients propagate *through* the frozen CA blocks back into Student LoRA weights, teaching the Student to produce latents that naturally decode as high-quality reasoning traces.

### Loss Functions

$$\mathcal{L} = \lambda_\text{verb} \mathcal{L}_\text{DPO} + \lambda_\text{distill} \mathcal{L}_\text{MSE}(h_S, h_T) + \lambda_\text{ans} \mathcal{L}_\text{MSE}(\hat{\tau}, \tau) + \lambda_\text{spatial} \mathcal{L}_\text{spatial}$$

| Term | Formula | λ | Purpose |
|------|---------|---|---------|
| `L_verb` | DPO(τ⁺, τ⁻) via Verbalizer | 1.0 | High-level reasoning alignment |
| `L_distill` | MSE(h_S, h_T) on `<ans>` hidden states | 1.0 | Spatial hidden-state alignment |
| `L_ans` | MSE(predicted waypoints, GT waypoints) | 1.0 | Physical waypoint grounding |
| `L_spatial` | −CosSim(MLP(x_V), VGGT(I)) | 0.1 | Visual geometry alignment |

### Reward Function (GRPO Teacher)

```
r = 0.9 × r_visual + 0.1 × r_format

r_visual = 0.5 × r_goal  +  0.5 × r_traj
r_goal   = 0.5 × ( f(p₁,p̂₁) + f(pₖ,p̂ₖ) )   where f(p,p') = max(0, 1 − ‖p−p'‖²)
r_traj   = max(0, 1 − DTW(τ, τ̂))
r_format = 1.0 (exact structure) | ROUGE avg (QA tasks)
```

See [`train/stage2/README.md`](train/stage2/README.md) for full implementation details.

---

## Stage 3 — Action Expert · CFM Adapter 📋

A **Conditional Flow Matching** head is trained on top of the frozen Student model's latent outputs. The CFM adapter learns to map the M=6 latent vectors produced by the Student into a sequence of physical action tokens (e.g. end-effector delta poses or joint targets).

- **Input:** Student latent sequence `[z₁, …, z_M]` + goal-delta conditioning token
- **Output:** Continuous action trajectory decoded via flow matching
- **Conditioning:** Goal-delta token is injected into the first Cross-Attention layer of the adapter

---

## Stage 4 — Partial VLM Coupling · Spatial Forcing 📋

Selective unfreezing of VLM layers combined with a spatial alignment loss using frozen visual features from **VGGT** / **DINOv2** as supervision signal. Spatial Forcing encourages the VLM's internal representations to remain geometrically consistent with the raw scene structure.

See [`Spatial-Forcing/`](Spatial-Forcing/) for the standalone module.

---

## Stage 5 — LIBERO Evaluation · RL Fine-Tuning ⚗️ *(Optional)*

Final downstream evaluation on the [LIBERO](https://libero-project.github.io) manipulation benchmark suite. Optional RL fine-tuning stage using inference-time reward feedback following the [RLInf](https://arxiv.org/abs/2505.01821) paradigm.

---

## Repository Structure

```
ReasonFlow-VLA/
├── data/
│   ├── stage_1_datasets_static.py   # Lazy-loading map-style Dataset (8 sources)
│   ├── build_hf_subset.py           # Materialize & push compact subset to HF Hub
│   ├── hf_subset_dataset.py         # Cloud-side loader (same interface as above)
│   └── push_from_cache.py           # Recovery: push from existing HF Arrow cache
├── train/
│   ├── stage_1_sft_training.py      # Stage 1 Unsloth SFT entry point
│   └── stage2/
│       ├── models/
│       │   ├── latent_student.py    # Student VLM: latent loop + spatial tokens
│       │   ├── verbalizer.py        # 0.8B LM + per-layer CA + DPO/LM losses
│       │   └── spatial_forcing.py  # Frozen VGGT extractor + ProjectionMLP
│       ├── training/
│       │   ├── grpo_teacher.py      # Teacher: GRPO rollouts, scoring, h_T extraction
│       │   ├── student_losses.py    # All four Student loss terms, phase-aware
│       │   └── train_stage2.py      # Main loop: three optimizers, checkpointing
│       └── rewards/
│           ├── action_reward.py     # r_visual: goal reward + DTW trajectory reward
│           └── qa_reward.py         # r_format: structure check + ROUGE
└── Spatial-Forcing/                 # Standalone spatial forcing module
```

---

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX A4000 (16 GB VRAM) |
| RAM | 128 GB @ 4400 MT/s |
| CPU | Intel Xeon w3-2425 |

---

## References

This project draws from the following works:

| Paper | Relevance |
|-------|-----------|
| [Fast Think, Then Act](https://arxiv.org/abs/2601.09708) — NVIDIA | Primary architectural inspiration |
| [SmolVLA](https://arxiv.org/abs/2506.01844) — HuggingFace | Compact VLA design; Verbalizer distillation concept |
| [RememVLA](https://arxiv.org/abs/2505.07218) | Recurrent memory module (N=8 query vectors) |
| [Spatial Forcing](https://arxiv.org/abs/2501.09808) | Spatial grounding via frozen visual encoder supervision |
| [RLInf](https://arxiv.org/abs/2505.01821) | Inference-time RL feedback (Stage 5) |
| [GRPO](https://arxiv.org/abs/2402.03300) — DeepSeek | Group Relative Policy Optimisation for Teacher training |
| [DPO](https://arxiv.org/abs/2305.18290) | Verbalizer preference optimisation |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | CFM Action Expert (Stage 3) |

---

## Acknowledgements

This project is submitted in partial fulfilment of the requirements for the Bachelor of Computer Science degree at **Universiti Teknikal Malaysia Melaka (UTeM)**.
