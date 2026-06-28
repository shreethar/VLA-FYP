"""
train_teacher_grpo.py
---------------------
Trains the Teacher model using Unsloth + TRL GRPOTrainer.
Automatically extracts `h_T` using `ThinkFlowGRPOTrainer` and saves to the offline directory.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
from datasets import load_dataset

from unsloth import FastVisionModel
from trl import GRPOConfig

from training.thinkflow_grpo_trainer import ThinkFlowGRPOTrainer
from stage2_dataloader import _reformat_traj_prompt, _reformat_qa_prompt, parse_waypoints, K_WAYPOINTS
from rewards.action_reward import CombinedActionReward, ActionAlignedReward
from rewards.qa_reward import FormatReward


def prepare_grpo_dataset(hf_repo, split, tokenizer, subset_ratio=1.0):
    dataset = load_dataset(hf_repo, split=split)
    
    if subset_ratio < 1.0:
        subset_size = int(len(dataset) * subset_ratio)
        dataset = dataset.shuffle(seed=42).select(range(subset_size))

    def filter_fn(example):
        if example.get("dataset") in ["pixmocapqa", "pixmocap", "pixmoama"]:
            return False
        if example.get("type", "trajectory") == "trajectory":
            wpts = parse_waypoints(example["assistant"])
            if wpts is None:
                return False
        return True

    dataset = dataset.filter(filter_fn)

    from torch.utils.data import Dataset as TorchDataset
    
    class GRPOMappedDataset(TorchDataset):
        def __init__(self, hf_dataset, tokenizer):
            self.dataset = hf_dataset
            self.tokenizer = tokenizer
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            example = self.dataset[idx]
            task_type = example.get("type", "trajectory")
            
            if task_type == "trajectory":
                human_text = _reformat_traj_prompt(example["human"])
                wpts = parse_waypoints(example["assistant"])
                gt_waypoints = wpts.tolist()
                qa_answer = ""
            else:
                human_text = _reformat_qa_prompt(example["human"])
                gt_waypoints = [[0.0, 0.0]] * K_WAYPOINTS
                qa_answer = example.get("qa_answer", example.get("assistant"))

            frames = example["frames"]
            if len(frames) > 1:
                max_frames = 8
                if len(frames) > max_frames:
                    step = len(frames) / max_frames
                    frames = [frames[int(i * step)] for i in range(max_frames)]
            
            IMAGE_SIZE = 448
            frames = [
                f.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                if f.size != (IMAGE_SIZE, IMAGE_SIZE) or f.mode != "RGB" else f
                for f in frames
            ]

            prompt_list = [
                {"role": "user", "content": [
                    *([{"type": "image"}] * len(frames)),
                    {"type": "text", "text": human_text}
                ]}
            ]

            prompt = self.tokenizer.apply_chat_template(
                prompt_list, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=True
            )

            sample_id = example.get("id", example.get("uuid", f"sample_{hash(human_text)}"))

            return {
                "prompt": prompt,
                "images": frames,
                "sample_id": sample_id,
                "ground_truth": {
                    "gt_waypoints": gt_waypoints,
                    "task_type": task_type,
                    "qa_answer": qa_answer,
                    "dataset": example.get("dataset", "unknown")
                }
            }

    return GRPOMappedDataset(dataset, tokenizer)


def get_reward_wrapper():
    visual_reward = ActionAlignedReward()
    format_reward = FormatReward()
    combined_reward_fn = CombinedActionReward(visual_reward, format_reward)

    def thinkflow_reward_func(completions, prompts, **kwargs):
        B = len(completions)
        # Handle completions being either a list of strings or list of msg dicts
        extracted_completions = []
        for c in completions:
            if isinstance(c, list):
                extracted_completions.append(c[-1]["content"] if c else "")
            else:
                extracted_completions.append(c)

        batched_gt = {}
        if "ground_truth" in kwargs:
            gt_list = kwargs["ground_truth"]
            batched_gt["task_type"] = [gt["task_type"] for gt in gt_list]
            batched_gt["qa_answer"] = [gt["qa_answer"] for gt in gt_list]
            batched_gt["dataset"] = [gt["dataset"] for gt in gt_list]
            
            gt_waypoints = []
            for gt in gt_list:
                gt_waypoints.append(torch.tensor(gt["gt_waypoints"], dtype=torch.float32))
            batched_gt["gt_waypoints"] = torch.stack(gt_waypoints, dim=0)
            
        rewards = combined_reward_fn(
            rollout_ids=None,
            rollout_text=extracted_completions,
            ground_truth=batched_gt
        )
        return rewards.tolist()
        
    return thinkflow_reward_func


def main():
    parser = argparse.ArgumentParser("GRPO Teacher Args")
    parser.add_argument("--model_name", type=str, default="shreethar/stage1_unsloth")
    parser.add_argument("--offline_data_dir", type=str, default="checkpoints/stage2_decoupled/offline_data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage2_decoupled/teacher_lora")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    args = parser.parse_args()

    max_seq_length = 4096

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
    )

    train_dataset = prepare_grpo_dataset("shreethar/FYP-Stage2-dataset", "test", tokenizer, subset_ratio=1.0)
    
    think_end_token_id = tokenizer.tokenizer.encode("</think>")[-1]

    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        log_completions=False,
        per_device_train_batch_size=args.batch_size * args.num_generations,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=max_seq_length - 2048,
        max_completion_length=2048,
        max_steps=args.max_steps,
        save_steps=50,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=args.output_dir,
        
        # GSPO settings from notebook
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type="dr_grpo",
        
        # Required to fix memory layout issues in some versions of TRL
        remove_unused_columns=False,
    )

    trainer = ThinkFlowGRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[get_reward_wrapper()],
        train_dataset=train_dataset,
        thinkflow_offline_dir=args.offline_data_dir,
        think_end_token_id=think_end_token_id,
        tf_tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    main()
