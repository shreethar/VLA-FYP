"""
tokenizer_setup.py
------------------
Registers the <ans> and </ans> special tokens with the Qwen tokenizer.

WHY this is needed
------------------
The model learns to GENERATE <think>...</think><ans>...</ans> through Stage 1
SFT training on CoT traces — no special token registration is needed for that.

The ONLY reason <ans> must be a special token is architectural:
    grpo_teacher.py:_find_answer_positions() searches for a single token ID
    to locate the exact hidden state position for h_T extraction (L_distill).

    If <ans> is a plain string it tokenises into multiple sub-tokens:
        "<ans>" → ["<", "ans", ">"]   (3 tokens, ambiguous position)
    As a special token it is guaranteed to be exactly ONE token at ONE position.

<think> and </think> are NOT registered here because:
    1. Qwen3 models already include them natively as thinking-mode tokens.
    2. The architecture never needs to index their exact position — only <ans>
       position matters for h_T extraction.

WHAT this script does
---------------------
1. Loads the base tokenizer
2. Adds <ans> and </ans> as special tokens (if not already present)
3. Returns the answer_token_id to pass to GRPOTeacher and train_stage2.py
4. Saves the extended tokenizer to disk so all training scripts load the
   same tokenizer without repeating this setup

Usage
-----
    # Once, before any training:
    python tokenizer_setup.py \
        --model_name Qwen/Qwen3.5-4B \
        --save_dir   tokenizer/

    # The script prints the answer_token_id — use it in train_stage2.py:
    python training/train_stage2.py \
        --answer_token_id <PRINTED_ID> \
        ...

    # Or import and call directly in your launch script:
    from tokenizer_setup import setup_tokenizer
    tokenizer, answer_token_id = setup_tokenizer(model_name, save_dir)
"""

import os
import argparse
from transformers import AutoTokenizer


# These are the ONLY tokens we register as special.
# <think> / </think> are already in Qwen3's vocabulary.
THINK_END_TOKEN = "</think>"   # native Qwen3 token, already in vocab


def setup_tokenizer(
    model_name: str = "shreethar/stage1_unsloth",
    save_dir: str   = "tokenizer/",
) -> tuple:
    """
    Load and save the tokenizer, then record the </think> token ID.

    WHY </think> instead of <ans>:
        <ans> was needed as a single-token position anchor for h_T extraction.
        </think> serves the exact same role — it is Qwen3's native end-of-reasoning
        token (always a single token), and it marks the exact transition from
        thinking to answer, which is the correct h_T alignment point.
        No vocabulary extension needed → model embedding table is unchanged.

    Returns
    -------
    tokenizer         : AutoTokenizer (unchanged from base model)
    think_end_token_id: int — the single token ID for </think>
                        Pass this to GRPOTeacher(think_end_token_id=...) and
                        train_stage2(think_end_token_id=...)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Verify </think> is already in the vocabulary as a single token
    test_ids = tokenizer.encode(THINK_END_TOKEN, add_special_tokens=False)
    assert len(test_ids) == 1, (
        f"</think> tokenised to {len(test_ids)} tokens instead of 1. "
        f"IDs: {test_ids}. This model may not natively support Qwen3 thinking mode."
    )

    think_end_token_id = tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
    print(f"think_end_token_id (</think>) = {think_end_token_id}")
    print(f"Vocabulary size (unchanged):   {len(tokenizer)}")

    # Save tokenizer (no structural changes, but save for reproducibility)
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to: {save_dir}")

    # Write token ID config so other scripts don't need to re-import
    config_path = os.path.join(save_dir, "thinkflow_token_ids.txt")
    with open(config_path, "w") as f:
        f.write(f"think_end_token_id={think_end_token_id}\n")
        f.write(f"think_end_token={THINK_END_TOKEN}\n")
    print(f"Token IDs written to: {config_path}")

    return tokenizer, think_end_token_id


def load_answer_token_id(tokenizer_dir: str) -> int:
    """
    Read the think_end_token_id from the saved config file.

    Named 'load_answer_token_id' for backward compatibility with train_stage2.py.
    Returns the </think> token ID which is used as the h_T extraction anchor.

    Example
    -------
    think_end_token_id = load_answer_token_id("tokenizer/")
    """
    config_path = os.path.join(tokenizer_dir, "thinkflow_token_ids.txt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Token ID config not found at {config_path}. "
            "Run tokenizer_setup.py first."
        )
    with open(config_path) as f:
        for line in f:
            if line.startswith("think_end_token_id="):
                return int(line.strip().split("=")[1])
            # Backward compat: old files used 'answer_token_id'
            if line.startswith("answer_token_id="):
                return int(line.strip().split("=")[1])
    raise ValueError("think_end_token_id not found in config file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen3.5-4B",
        help="HuggingFace model name to load tokenizer from",
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="tokenizer/",
        help="Directory to save the tokenizer and token ID config",
    )
    args = parser.parse_args()

    _, think_end_token_id = setup_tokenizer(args.model_name, args.save_dir)
    print(f"\nPass this to train_stage2.py: --answer_token_id {think_end_token_id}")
