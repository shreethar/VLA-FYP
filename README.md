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
Hardware used:
1. GPU: NVIDIA RTX A4000
2. RAM: 128GB @ 4400 MT/s
3. CPU: Intel(R) Xeon(R) w3-2425
