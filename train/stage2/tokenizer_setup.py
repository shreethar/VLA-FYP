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


# <answer> / </answer> are registered as special tokens so each becomes a
# single, stable token ID — required for exact position-based h_T extraction.
# NOTE: after registration, call model.resize_token_embeddings(new_vocab_size)
#       in every script that loads a model (stage1_5 SFT, train_stage2, etc.)
ANSWER_OPEN_TOKEN  = "<answer>"
ANSWER_CLOSE_TOKEN = "</answer>"


def setup_tokenizer(
    model_name: str = "shreethar/stage1_unsloth",
    save_dir: str   = "tokenizer/",
) -> tuple:
    """
    Load the tokenizer, register <answer> / </answer> as special tokens,
    and save everything so other scripts can load without re-registering.

    WHY <answer> as the h_T anchor:
        <answer> marks the START of the answer section.  The hidden state at
        this position represents "I am about to give the answer", which is the
        most semantically meaningful point for Teacher→Student distillation.
        </think> marks end-of-reasoning, which is less precise.

        Because <answer> is NOT in Qwen3's native vocabulary, registering it
        adds a new row to the embedding table.  Every script that loads any of
        the three models MUST call:
            model.resize_token_embeddings(new_vocab_size)
        after loading.  The new_vocab_size is saved to thinkflow_token_ids.txt.

    Returns
    -------
    tokenizer        : AutoTokenizer with <answer> / </answer> registered
    answer_token_id  : int — single token ID for <answer>
                       Pass this to GRPOTeacher(answer_token_id=...) and
                       train_stage2(answer_token_id=...)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Register tokens only if not already in vocab
    tokens_to_add = [
        tok for tok in [ANSWER_OPEN_TOKEN, ANSWER_CLOSE_TOKEN]
        if tok not in tokenizer.get_vocab()
    ]
    if tokens_to_add:
        n = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_to_add}
        )
        print(f"Added {n} special token(s): {tokens_to_add}")
    else:
        print(f"Tokens already present: {[ANSWER_OPEN_TOKEN, ANSWER_CLOSE_TOKEN]}")

    # Verify <answer> is exactly one token
    test_ids = tokenizer.encode(ANSWER_OPEN_TOKEN, add_special_tokens=False)
    assert len(test_ids) == 1, (
        f"<answer> tokenised to {len(test_ids)} tokens instead of 1. "
        f"IDs: {test_ids}. Special token registration failed."
    )

    answer_token_id = tokenizer.convert_tokens_to_ids(ANSWER_OPEN_TOKEN)
    new_vocab_size  = len(tokenizer)
    print(f"answer_token_id (<answer>) = {answer_token_id}")
    print(f"New vocabulary size        = {new_vocab_size}")

    # Save extended tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to: {save_dir}")

    # Write config so other scripts can read IDs without re-importing
    config_path = os.path.join(save_dir, "thinkflow_token_ids.txt")
    with open(config_path, "w") as f:
        f.write(f"answer_token_id={answer_token_id}\n")
        f.write(f"answer_open_token={ANSWER_OPEN_TOKEN}\n")
        f.write(f"answer_close_token={ANSWER_CLOSE_TOKEN}\n")
        f.write(f"new_vocab_size={new_vocab_size}\n")
    print(f"Token IDs written to: {config_path}")

    return tokenizer, answer_token_id


def load_answer_token_id(tokenizer_dir: str) -> int:
    """Read the <answer> token ID from the saved config."""
    config_path = os.path.join(tokenizer_dir, "thinkflow_token_ids.txt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Token ID config not found at {config_path}. "
            "Run tokenizer_setup.py first."
        )
    with open(config_path) as f:
        for line in f:
            if line.startswith("answer_token_id="):
                return int(line.strip().split("=")[1])
            if line.startswith("think_end_token_id="):   # backward compat
                return int(line.strip().split("=")[1])
    raise ValueError("answer_token_id not found in config file.")


def load_new_vocab_size(tokenizer_dir: str):
    """Return the vocab size after <answer> registration, for resize_token_embeddings()."""
    config_path = os.path.join(tokenizer_dir, "thinkflow_token_ids.txt")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        for line in f:
            if line.startswith("new_vocab_size="):
                return int(line.strip().split("=")[1])
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="shreethar/stage1_unsloth")
    parser.add_argument("--save_dir",   type=str, default="tokenizer/")
    args = parser.parse_args()

    _, answer_token_id = setup_tokenizer(args.model_name, args.save_dir)
    print(f"\nPass to train_stage2.py: --answer_token_id {answer_token_id}")
    print("IMPORTANT: call model.resize_token_embeddings(new_vocab_size) after loading each model.")
