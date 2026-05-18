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
        --model_name Qwen/Qwen2.5-VL-4B-Instruct \
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
ANS_OPEN_TOKEN  = "<ans>"
ANS_CLOSE_TOKEN = "</ans>"


def setup_tokenizer(
    model_name: str = "Qwen/Qwen2.5-VL-4B-Instruct",
    save_dir: str   = "tokenizer/",
) -> tuple:
    """
    Load, extend, and save the tokenizer.

    Returns
    -------
    tokenizer       : AutoTokenizer with <ans> / </ans> registered
    answer_token_id : int — the single token ID for <ans>
                      Pass this to GRPOTeacher(answer_token_id=...) and
                      train_stage2(answer_token_id=...)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Check which tokens are already present
    tokens_to_add = []
    for tok in [ANS_OPEN_TOKEN, ANS_CLOSE_TOKEN]:
        if tok not in tokenizer.get_vocab():
            tokens_to_add.append(tok)

    if tokens_to_add:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        print(f"Added {num_added} special token(s): {tokens_to_add}")
    else:
        print(f"Tokens already present: {[ANS_OPEN_TOKEN, ANS_CLOSE_TOKEN]}")

    # Verify <ans> is a single token
    test_ids = tokenizer.encode(ANS_OPEN_TOKEN, add_special_tokens=False)
    assert len(test_ids) == 1, (
        f"<ans> tokenised to {len(test_ids)} tokens instead of 1. "
        f"IDs: {test_ids}. Something went wrong with special token registration."
    )

    answer_token_id = tokenizer.convert_tokens_to_ids(ANS_OPEN_TOKEN)
    print(f"answer_token_id (<ans>) = {answer_token_id}")
    print(f"Vocabulary size after extension: {len(tokenizer)}")

    # Save extended tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to: {save_dir}")

    # Also write the answer_token_id to a small config file so other
    # scripts can read it without re-importing this module
    config_path = os.path.join(save_dir, "thinkflow_token_ids.txt")
    with open(config_path, "w") as f:
        f.write(f"answer_token_id={answer_token_id}\n")
        f.write(f"ans_open_token={ANS_OPEN_TOKEN}\n")
        f.write(f"ans_close_token={ANS_CLOSE_TOKEN}\n")
    print(f"Token IDs written to: {config_path}")

    return tokenizer, answer_token_id


def load_answer_token_id(tokenizer_dir: str) -> int:
    """
    Read the answer_token_id from the saved config file.
    Use this in train_stage2.py instead of hardcoding.

    Example
    -------
    answer_token_id = load_answer_token_id("tokenizer/")
    """
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
    raise ValueError("answer_token_id not found in config file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen2.5-VL-4B-Instruct",
        help="HuggingFace model name to load tokenizer from",
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="tokenizer/",
        help="Directory to save the extended tokenizer",
    )
    args = parser.parse_args()

    _, answer_token_id = setup_tokenizer(args.model_name, args.save_dir)
    print(f"\nPass this to train_stage2.py: --answer_token_id {answer_token_id}")