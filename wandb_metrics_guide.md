# ThinkFlow-VLA Stage 2: WandB Metrics Guide

During Stage 2 training, the pipeline logs several key metrics to WandB. Understanding these graphs is critical for diagnosing model convergence and identifying bugs. 

Here is a detailed breakdown of every metric, what direction it should trend, and what it means if it misbehaves.

---

## 1. `distill/l_distill`
**What it is:** The Mean Squared Error (MSE) between the Teacher's hidden state at the `<answer>` token and the Student's hidden state at the `</think>` token.
**Expected Trend:** 📉 **DOWN**
- **What it means:** The Student is successfully learning to compress the Teacher's deep reasoning trace into its internal latent space.
- **If it goes UP or stays flat:** The Teacher's hidden states might be too complex, or the Student's learning rate is too low. Check if `h_T` and `h_S` are being extracted at the correct, consistent token positions.

## 2. `distill/cosine_sim`
**What it is:** The Cosine Similarity between the Teacher's `h_T` and the Student's `h_S`.
**Expected Trend:** 📈 **UP (towards 1.0)**
- **What it means:** Similar to `l_distill`, but measures the *directional* alignment of the hidden states rather than magnitude. A value near 1.0 means the Student's latents perfectly mimic the Teacher's reasoning vector.
- **If it goes DOWN or stays near 0:** The Student is failing to align with the Teacher. This often happens if the spatial forcing loss is overpowering the distillation loss, or if gradients are vanishing in the `LatentStudent`'s DeltaNet layers.

## 3. `loss/l_ans`
**What it is:** The L1/MSE loss between the physical waypoints predicted by the Student (via its spatial MLP) and the ground truth waypoints.
**Expected Trend:** 📉 **DOWN**
- **What it means:** The Student's spatial tokens are accurately decoding into the correct robotic actions/coordinates.
- **If it goes UP or fluctuates wildly:** The Student is losing physical grounding. Ensure that `human` prompts are formatted correctly (i.e. replacing the Stage 1 system prompt with the Stage 2 `<think>/<ans>` format). 

## 4. `dpo/loss` (Frozen Phase Only)
**What it is:** Direct Preference Optimization loss. It trains the Student to generate latents that the Verbalizer prefers to decode into high-reward traces (τ+) over low-reward traces (τ−).
**Expected Trend:** 📉 **DOWN**
- **What it means:** The Student is learning to produce latents that the Verbalizer interprets as successful reasoning.
- **If it goes UP:** The Teacher's rewards might be too noisy, causing τ+ and τ− to be virtually identical, giving the DPO algorithm no clear preference gradient.

## 5. `dpo/reward_margin`
**What it is:** The difference in implicit log-probabilities the Verbalizer assigns to τ+ vs τ−.
**Expected Trend:** 📈 **UP**
- **What it means:** The Verbalizer is becoming increasingly confident that the latents associated with τ+ are better than the latents associated with τ−.
- **If it goes DOWN or goes Negative:** The Student is actively confusing the Verbalizer, or the KL penalty is too high.

## 6. `loss/l_verb`
**What it is:** The overall Language Modeling Cross-Entropy loss of the Verbalizer trying to reconstruct the Teacher's text from the Student's latents.
**Expected Trend:** 📉 **DOWN**
- **What it means:** The Cross-Attention layers in the Verbalizer are successfully injecting the Student's latents into the language model.
- **If it goes UP:** The Verbalizer's Cross-Attention mechanism is failing, or the Student's latents are collapsing to zero (making it impossible to reconstruct text).

## 7. `teacher/reward_mean` & `teacher/advantage_mean`
**What it is:** The raw scores from the custom Reward Functions (Action Reward + Format Reward).
**Expected Trend:** 📈 **UP (Rewards) / Flat near 0 (Advantages)**
- **What it means:** The Teacher is exploring the action space and finding better, more accurate robotic trajectories. GRPO advantages will naturally center around 0 because they are group-relative (normalized).
- **If Rewards stay low:** The Teacher policy has collapsed, or the generation temperature is too low/high, preventing it from finding the correct waypoints.

## 8. `grad/lora_total` & `grad/spatial_total`
**What it is:** The global L2 norm of the gradients flowing through the LoRA adapters and the Spatial layers.
**Expected Trend:** 🌊 **STABLE (No massive spikes or drops to 0)**
- **What it means:** Gradients are flowing healthily through the network.
- **If it Spikes to >1000:** Gradient explosion. You may need to lower your learning rate or increase gradient clipping.
- **If it Drops to exactly 0.0:** Vanishing gradients or a broken computational graph (e.g. detaching tensors prematurely).
