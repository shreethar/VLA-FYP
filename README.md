# VLA-FYP

## This project aims to re-create the [**Fast Think Act**](https://arxiv.org/pdf/2601.09708) paper released by NVIDIA but with some tweaks.

### It relies on Conditional Flow Matching Adapter for Action Token Generation. It also has a Goal-Delta injected as conditioning token in first Cross Attention Layer and Memory Module with N=8 recurrent query vectors m_t

---
This project is broken into 7 stages

- [x] Stage 1: Robot Grounding SFT
- [ ] Stage 2: CoT-SFT (Teacher Reasoning Warmup)
- [ ] Stage 3: GRPO Teacher Training
- [ ] Stage 4: Joint Student Distillation + Spatial Forcing
- [ ] Stage 5: Action Expert Training
- [ ] Stage 6: Partial VLM Coupling
- [ ] Stage 7: LIBERO Fine-Tuning + RL (Optional)

---
Stage Details:
1. Stage 1: Robot Grounding SFT
* This stage is to give the robotic foundational knowledge to the the VLM as a genral VLM do not have good robotic knowledge
* The datasets used in this stage are [MolmoAct Trajectory](https://huggingface.co), [RoboVQA](), [RoboFAC](), [PixmoCap](), [PixmoCapQA](), [PixmoAMA](), [ShareRobot]()
* I sampled around 10% of the data (If the full sample from the dataset is less than 100K samples, I take 100% of that dataset)
* Total came up to be around 560K samples
* I trained for 750K steps (600K samples / (Batch Size = 1 * Gradient Accum = 8) - Rounded up to 600K samples for calculation, or 1 epoch if you're not streaming
* Learning Rate: 1e-5, Batch size: 1 due to GPU limitations
* It's taking around 10 days to train Stage 1 using Unsloth
---
Hardware used:
1. GPU: NVIDIA RTX A4000
2. RAM: 128GB @ 4400 MT/s
3. CPU: Intel(R) Xeon(R) w3-2425
