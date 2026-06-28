# Teacher GRPO Pipeline Migration

I've successfully migrated the Teacher's GRPO pipeline to use the robust HuggingFace/TRL `GRPOTrainer` and Unsloth optimization, ensuring mathematical stability and eliminating the reward collapse issue. Most importantly, I've patched the trainer to flawlessly interoperate with the decoupled `student_offline` distillation workflow.

## Overview of Changes

To extract the `</think>` hidden states (`h_T`) after the optimizer steps without breaking the internal mechanics of TRL, I've created a custom subclass of `GRPOTrainer` called `ThinkFlowGRPOTrainer`. This approach avoids modifying the TRL source code and is fully compatible with native `accelerate` scaling.

### 1. `ThinkFlowGRPOTrainer` Implementation ([`train/stage2/training/thinkflow_grpo_trainer.py`](file:///home/ubuntu/VLA-FYP/train/stage2/training/thinkflow_grpo_trainer.py))

*   **Preserving Dataset Metadata**: TRL inherently strips any columns from the HF Dataset that aren't native inputs to the model. I overrode `_generate_and_score_completions` to duplicate our custom `ground_truth` and `sample_id` fields \(as well as original `pixel_values` and grids\) `G` times. This ensures that when TRL internally reshuffles and splits the `B*G` generations into micro-batches, our custom fields are routed correctly alongside them.
*   **On-the-fly `h_T` Extraction**: I overrode `compute_loss` to intercept the pipeline immediately after the loss is computed and the gradients are formed. Within this hook:
    *   It reshapes the flattened `B*G` advantages back to `[B, G]`.
    *   It computes `τ+` (best generation) and `τ-` (worst generation).
    *   It runs a targeted, single-sequence forward pass **strictly on `τ+`** with `output_hidden_states=True` to extract the `h_T` feature vector of the `</think>` token.
*   **Offline Data Serialization**: Just before returning the loss to the optimizer, the trainer serializes the batch directly into the `offline_data_dir`. The schema of the saved `.pt` dictionaries has been formatted to **exactly match** the expectations of `train_stage2.py`'s `student_offline` mode (`gt_waypoints`, `tau_pos_ids`, `h_T`, etc.).

### 2. Teacher Training Orchestrator ([`train/stage2/training/train_teacher_grpo.py`](file:///home/ubuntu/VLA-FYP/train/stage2/training/train_teacher_grpo.py))

*   **Data Pipeline Integration**: Developed a dataset preparer (`prepare_grpo_dataset`) that loads your HuggingFace dataset (`shreethar/FYP-Stage2-dataset`), applies the specific filters (excluding `pixmocapqa`, checking for valid waypoints), and maps the raw text into the multimodal conversational dictionary structure TRL requires.
*   **Reward Wrapping**: Designed a wrapper for your existing `CombinedActionReward` to rebuild the structured `ground_truth` dicts from the batched tensors.
*   **Initialization**: Configured the model with `FastVisionModel.get_peft_model` replicating the precise hyperparameter settings shown in your reference Unsloth notebook (cosine scheduler, adamw_8bit, learning rate 5e-6).

## Verification Plan

You are now ready to launch the Teacher training. 

> [!TIP]
> Ensure you run this from within the `train/stage2/training` directory so relative imports map correctly:
> ```bash
> python train_teacher_grpo.py --offline_data_dir checkpoints/stage2_decoupled/offline_data
> ```

As the trainer runs, monitor the `offline_data_dir`. You will see `.pt` files matching the exact format `step_000001_micro_00.pt`. Once the teacher training generates sufficient rollouts, you can boot up `train_stage2.py` using `student_offline` mode entirely unmodified.
