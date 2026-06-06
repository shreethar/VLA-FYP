"""
stage1_5_answer_format_sft.py
------------------------------
Stage 1.5: Teach the model to wrap outputs in <answer>...</answer> tags.

Uses the SAME dataset as Stage 1 (VLAStaticDataset from data/stage_1_datasets_static.py),
but wraps every assistant response with <answer>...</answer>.

WHY SFT:  Format learning = output structure, not reasoning quality.
          SFT demonstrates the target directly.  ~1-3 hrs vs days with GRPO.

WHY <answer> not </think>:
          After </think> the model may emit filler text before [[x,y],...].
          <answer>...</answer> is an unambiguous extraction boundary.

Architecture after this stage:
    </think>  →  h_T anchor in grpo_teacher.py   (native Qwen3 single token)
    <answer>  →  waypoint extraction in action_reward.py

Before:
    assistant: "[[450,300],[500,350],[560,410],[620,450],[680,490]]"
After:
    assistant: "<answer>[[450,300],[500,350],[560,410],[620,450],[680,490]]</answer>"

Same wrapping is applied to QA / planning / affordance records so the model
learns a consistent "always use <answer>" habit across all task types.

Usage
-----
# Sanity-check the data pipeline without loading the model:
python train/stage1_5_answer_format_sft.py --dry_run

# Full training (~1-3 hrs on RTX A4000 16 GB):
python train/stage1_5_answer_format_sft.py \\
    --model        shreethar/stage1_unsloth \\
    --output_dir   checkpoints/stage1_5 \\
    --sample_frac  0.10 \\
    --epochs       2 \\
    --batch_size   2 \\
    --grad_accum   8

# Push to Hub after training:
python train/stage1_5_answer_format_sft.py \\
    --model        shreethar/stage1_unsloth \\
    --output_dir   checkpoints/stage1_5 \\
    --push_to_hub  shreethar/stage1_5_unsloth \\
    --sample_frac  0.10 --epochs 2
"""

import os
import sys
import random
import logging
import argparse

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
# stage_1_datasets_static.py lives in  <repo>/data/
# this script lives in                 <repo>/train/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, "data")
sys.path.insert(0, DATA_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Stage 1.5 dataset wrapper ─────────────────────────────────────────────────

class AnswerTagDataset(torch.utils.data.Dataset):
    """
    Wraps VLAStaticDataset and adds <answer>...</answer> around every
    assistant response.

    Also updates the user's system prompt for trajectory records to
    instruct the model to use answer tags (so prompt and target match).

    Parameters
    ----------
    vla_dataset  : a VLAStaticDataset instance (already built / split)
    indices      : list of integer indices to use (for 10% sub-sampling)
    """

    TRAJ_SYSTEM_15 = (
        "You are a robot manipulation assistant. "
        "Given an observation image and a task instruction, predict the "
        "end-effector's 2D trajectory as 5 waypoints.\n\n"
        "Think through the task, then output ONLY the coordinate list "
        "wrapped in answer tags:\n"
        "<answer>[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]</answer>\n\n"
        "Coordinates are 0-1000 integers (0=top-left, 1000=bottom-right)."
    )

    QA_SYSTEM_15 = (
        "You are a robot manipulation assistant. Answer questions about robot tasks, "
        "object affordances, spatial relationships, and manipulation strategies based "
        "on the provided image or video frame.\n\n"
        "Think through the question, then output your final response wrapped in answer tags:\n"
        "<answer>your response here</answer>"
    )

    # Regex to strip Stage 1 trajectory system prefix
    import re as _re
    _TRAJ_PREFIX_RE = _re.compile(
        r"^You are a robot manipulation assistant\. Given an observation image and a "
        r"task instruction, predict the end-effector's 2D trajectory.*?\n\n(Task:\s*)?",
        _re.DOTALL,
    )

    # Regex to strip Stage 1 QA system prefix
    _QA_PREFIX_RE = _re.compile(
        r"^You are a robot manipulation assistant\. Answer questions about robot tasks, "
        r"object affordances, spatial relationships, and manipulation strategies based "
        r"on the provided image or video frame\.\n\n",
        _re.DOTALL,
    )

    def __init__(self, vla_dataset, indices):
        self.vla   = vla_dataset
        self.idxs  = indices

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        item = self.vla[self.idxs[i]]
        msgs = item["messages"]          # list: [user_msg, assistant_msg]

        # Wrap assistant content with <answer> tags
        assert msgs[-1]["role"] == "assistant", "Last message must be assistant"
        orig_answer = msgs[-1]["content"]
        msgs[-1] = {
            "role":    "assistant",
            "content": f"<answer>{orig_answer}</answer>",
        }

        # Also update the user system prompt so it instructs <answer> format.
        # This prevents prompt-target mismatch.
        user_text_blocks = msgs[0]["content"]
        for j, block in enumerate(user_text_blocks):
            if isinstance(block, dict) and block.get("type") == "text":
                old_text = block["text"]
                
                if item.get("type") == "trajectory":
                    m = self._TRAJ_PREFIX_RE.match(old_text)
                    if m:
                        task_text = old_text[m.end():]
                        # Restore "Task: " if it was eaten or omitted
                        if not task_text.startswith("Task:"):
                            task_text = f"Task: {task_text}"
                        user_text_blocks[j] = {
                            "type": "text",
                            "text": f"{self.TRAJ_SYSTEM_15}\n\n{task_text.strip()}",
                        }
                else:
                    # Non-trajectory (QA, Affordance, etc.)
                    m = self._QA_PREFIX_RE.match(old_text)
                    if m:
                        question_text = old_text[m.end():]
                        user_text_blocks[j] = {
                            "type": "text",
                            "text": f"{self.QA_SYSTEM_15}\n\n{question_text.strip()}",
                        }
                break

        return {"messages": msgs}


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_stage1_datasets():
    """
    Load all HF datasets required by build_static_dataset.
    Mirrors the loading block from the Stage 1 training notebook.
    """
    from datasets import load_dataset

    logger.info("Loading HF datasets for Stage 1 sources ...")

    molmoact = load_dataset(
        "allenai/MolmoAct-Pretraining-Mixture",
        "auxiliary_trace", split="train",
    )
    pixmocap   = load_dataset("allenai/pixmo-cap",              split="train")
    pixmoama   = load_dataset("allenai/pixmo-ask-model-anything", split="train")
    pixmocapqa = load_dataset("allenai/pixmo-cap-qa",           split="train")

    logger.info("HF datasets loaded.")
    return molmoact, pixmocap, pixmoama, pixmocapqa


def build_dataset(sample_frac: float, seed: int) -> AnswerTagDataset:
    """
    1. Load HF datasets
    2. build_static_dataset() → VLAStaticDataset (train split)
    3. Take sample_frac random subset
    4. Wrap with AnswerTagDataset
    """
    from stage_1_datasets_static import build_static_dataset

    molmoact, pixmocap, pixmoama, pixmocapqa = load_stage1_datasets()

    splits = build_static_dataset(
        molmoact_hf_ds   = molmoact,
        pixmocap_hf_ds   = pixmocap,
        pixmoama_hf_ds   = pixmoama,
        pixmocapqa_hf_ds = pixmocapqa,
    )
    train_vla = splits["train"]
    total     = len(train_vla)

    n_keep = max(1, int(total * sample_frac))
    random.seed(seed)
    indices = random.sample(range(total), n_keep)

    logger.info(
        f"Stage 1 train split: {total:,} records  →  "
        f"10%% sample: {n_keep:,} records"
    )
    return AnswerTagDataset(train_vla, indices)


# ── Dry-run (no GPU, no model) ────────────────────────────────────────────────

def dry_run(args):
    """Verify data pipeline quickly — no model loading."""
    logger.info("=== DRY RUN: checking data pipeline only ===")
    ds = build_dataset(args.sample_frac, args.seed)

    print(f"\nTotal samples: {len(ds):,}")
    print("\n── First 3 samples ──")
    for i in range(min(3, len(ds))):
        item = ds[i]
        msgs = item["messages"]

        user_content  = msgs[0]["content"]
        user_text     = next(
            (b["text"] for b in user_content if isinstance(b, dict) and b.get("type") == "text"),
            ""
        )
        has_image     = any(isinstance(b, dict) and b.get("type") in ("image","video")
                            for b in user_content)
        assistant_out = msgs[1]["content"]

        print(f"\nSample {i+1}:")
        print(f"  USER text (first 120 chars):  {user_text[:120]}...")
        print(f"  Has image/video:              {has_image}")
        print(f"  ASSISTANT:                    {assistant_out}")

    print("\nData pipeline OK. Remove --dry_run to start training.")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    from unsloth import FastVisionModel, is_bfloat16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    # ── 1. Load model ──────────────────────────────────────────────────────
    logger.info(f"Loading base model: {args.model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit                = False,
        use_gradient_checkpointing  = "unsloth",
    )

    # Register <answer> / </answer> as special tokens so the model learns
    # to generate them as single, stable tokens (required for h_T extraction
    # in Stage 2 grpo_teacher.py).
    # NOTE: FastVisionModel returns a Processor, not a bare tokenizer.
    #       Vocab operations must go through tokenizer.tokenizer (the inner tokenizer).
    ANSWER_OPEN  = "<answer>"
    ANSWER_CLOSE = "</answer>"
    inner_tok = tokenizer.tokenizer   # Qwen3VLProcessor → underlying tokenizer
    tokens_to_add = [t for t in [ANSWER_OPEN, ANSWER_CLOSE]
                     if t not in inner_tok.get_vocab()]
    if tokens_to_add:
        n = inner_tok.add_special_tokens({"additional_special_tokens": tokens_to_add})
        logger.info(f"Registered {n} special token(s): {tokens_to_add}")
    else:
        logger.info(f"Tokens already in vocab: {[ANSWER_OPEN, ANSWER_CLOSE]}")

    # Grow the embedding table to cover the new tokens
    new_vocab_size = len(inner_tok)
    model.resize_token_embeddings(new_vocab_size)
    logger.info(f"Vocabulary size after registration: {new_vocab_size}")

    # Small LoRA — this is format/habit learning only, not knowledge
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False,   # keep vision encoder frozen
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r            = 16,
        lora_alpha   = 32,
        lora_dropout = 0.05,
        bias         = "none",
        use_rslora   = False,
        target_modules= "all-linear",
    )
    logger.info("LoRA adapters applied (r=16, language layers only).")

    # ── 2. Build dataset ───────────────────────────────────────────────────
    dataset = build_dataset(args.sample_frac, args.seed)
    logger.info(f"Training on {len(dataset):,} samples.")

    # ── 3. SFT config ──────────────────────────────────────────────────────
    sft_cfg = SFTConfig(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_ratio                = 0.05,
        lr_scheduler_type           = "cosine",
        bf16                        = is_bfloat16_supported(),
        fp16                        = not is_bfloat16_supported(),
        save_strategy               = "steps",
        save_steps                  = 100,
        save_total_limit            = 5,        # keep last 5 checkpoints to avoid filling disk
        logging_steps               = 20,
        report_to                   = "wandb" if args.wandb else "none",
        run_name                    = "stage1_5_answer_format",
        remove_unused_columns       = False,
        dataset_text_field          = "",          # required for vision SFT
        dataset_kwargs              = {"skip_prepare_dataset": True},
        max_seq_length              = args.max_seq_len,
        dataloader_num_workers      = 2,
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = dataset,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        args          = sft_cfg,
    )

    logger.info("=" * 60)
    logger.info("Starting Stage 1.5 — Answer Format SFT")
    logger.info(f"  Samples    : {len(dataset):,}")
    logger.info(f"  Epochs     : {args.epochs}")
    logger.info(f"  Eff. batch : {args.batch_size * args.grad_accum}")
    logger.info(f"  LR         : {args.lr}")
    logger.info(f"  Output dir : {args.output_dir}")
    logger.info("=" * 60)

    trainer.train()

    # ── 4. Save ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving checkpoint to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        logger.info(f"Pushing to Hub: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

    logger.info("Stage 1.5 complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stage 1.5 SFT: teach <answer>...</answer> format on 10%% of Stage 1 data"
    )
    p.add_argument("--model",        default="shreethar/stage1_unsloth",
                   help="HF repo or local path of the Stage 1 checkpoint to fine-tune")
    p.add_argument("--output_dir",   default="checkpoints/stage1_5",
                   help="Where to save the fine-tuned checkpoint")
    p.add_argument("--push_to_hub",  default=None,
                   help="Optional HF repo to push to, e.g. shreethar/stage1_5_unsloth")
    p.add_argument("--sample_frac",  type=float, default=0.10,
                   help="Fraction of Stage 1 train records to use (default 10%%)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--epochs",       type=int,   default=2)
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--grad_accum",   type=int,   default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--max_seq_len",  type=int,   default=512)
    p.add_argument("--wandb",        action="store_true",
                   help="Enable WandB logging")
    p.add_argument("--dry_run",      action="store_true",
                   help="Check data pipeline only — no model loading, no training")
    args = p.parse_args()

    if args.dry_run:
        dry_run(args)
    else:
        train(args)
