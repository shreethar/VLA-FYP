"""
qa_reward.py
------------
Format correctness reward (r_format) for GRPO Teacher scoring.

r_format definition:
    - Exact format match (structural correctness) → 1.0
    - Free-form QA answer → average ROUGE-1/ROUGE-2/ROUGE-L score
    - Missing required tags or completely unparseable output → 0.0

What counts as "exact format":
    The Teacher must produce output matching EXACTLY:
        <think> ... </think><ans>x1,y1;x2,y2;x3,y3;x4,y4;x5,y5</ans>

    Structural check (order matters):
        1. <think>...</think> tag present and non-empty
        2. <ans>...</ans> tag present and contains exactly K=5 coordinate pairs
        3. No extraneous text outside the two tags
        4. All coordinates are valid floats in [0, 1]

For datasets that include free-form QA targets (RoboVQA, RoboFAC, EgoPlan):
    ground_truth dict contains "qa_answer" (str).
    In that case the format reward is the ROUGE score of the generated
    answer against the reference, since exact coordinate matching is
    not applicable.

ROUGE is computed using the `rouge_score` library (pure Python, no Java).
Install: pip install rouge-score
"""

import re
from typing import List, Optional

import torch

# Optional ROUGE — imported lazily so the module loads even without rouge_score
try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Structural format checker
# ---------------------------------------------------------------------------

_THINK_PATTERN = re.compile(
    r"<think>\s*(.*?)\s*</think>",
    re.DOTALL | re.IGNORECASE,
)
_ANS_PATTERN = re.compile(
    r"</think>\s*<ans>\s*([\d.]+,[\d.]+(?:;[\d.]+,[\d.]+)*)\s*</ans>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_EXTRA_TEXT_PATTERN = re.compile(
    r"^(?:(?!\s*<think>).)*<think>",
    re.DOTALL,
)


def check_structural_format(text: str, K: int = 5) -> bool:
    """
    Returns True iff the rollout text exactly matches the required structure:
        <think>NON-EMPTY</think><ans>K coordinate pairs</ans>

    Checks (in order):
        1. No leading text before <think>
        2. <think> block is non-empty
        3. <ans> block immediately follows </think>
        4. <ans> contains exactly K semicolon-separated x,y pairs
        5. All coordinates are floats in [0, 1]
        6. Nothing follows </ans>
    """
    text = text.strip()

    # 1. Must start with <think>
    if not text.lower().startswith("<think>"):
        return False

    # 2. <think> block must be non-empty
    think_match = _THINK_PATTERN.match(text)
    if not think_match or not think_match.group(1).strip():
        return False

    # 3 & 6. <ans> immediately after </think> with nothing following
    ans_match = _ANS_PATTERN.search(text)
    if not ans_match:
        return False

    # 4 & 5. Validate coordinates
    pairs = ans_match.group(1).strip().split(";")
    if len(pairs) != K:
        return False

    for pair in pairs:
        parts = pair.strip().split(",")
        if len(parts) != 2:
            return False
        try:
            x, y = float(parts[0]), float(parts[1])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return False
        except ValueError:
            return False

    return True


# ---------------------------------------------------------------------------
# ROUGE scorer (lazy init, shared instance)
# ---------------------------------------------------------------------------

_SCORER = None

def _get_scorer():
    global _SCORER
    if _SCORER is None:
        if not _ROUGE_AVAILABLE:
            raise ImportError(
                "rouge_score is required for QA reward. "
                "Install with: pip install rouge-score"
            )
        _SCORER = _rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    return _SCORER


def compute_rouge_score(hypothesis: str, reference: str) -> float:
    """
    Average of ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    Returns a float in [0, 1].
    """
    scorer = _get_scorer()
    scores = scorer.score(reference, hypothesis)
    return (
        scores["rouge1"].fmeasure
        + scores["rouge2"].fmeasure
        + scores["rougeL"].fmeasure
    ) / 3.0


def extract_think_content(text: str) -> str:
    """Extract content inside <think>...</think> for QA scoring."""
    match = _THINK_PATTERN.search(text)
    return match.group(1).strip() if match else text.strip()


# ---------------------------------------------------------------------------
# Format reward class
# ---------------------------------------------------------------------------

class FormatReward:
    """
    Computes r_format for a batch of rollout texts.

    Two modes, selected per sample based on ground_truth:
        1. Coordinate mode (ground_truth has "gt_waypoints" but no "qa_answer"):
               r_format = 1.0 if structure is perfect, else 0.0
        2. QA mode (ground_truth has "qa_answer"):
               r_format = average ROUGE(generated_answer, reference_answer)

    ground_truth dict keys:
        "gt_waypoints"  : [batch, K, 2]  — coordinate supervision (may be None)
        "qa_answer"     : List[str]       — free-form QA reference (may be None)
        "task_type"     : List[str]       — "waypoint" or "qa" per sample
    """

    def __init__(self, K: int = 5):
        self.K = K

    def __call__(
        self,
        rollout_ids,
        rollout_text: List[str],
        pixel_values=None,
        image_grid_thw=None,
        ground_truth: dict = None,
    ) -> torch.Tensor:
        """
        Returns
        -------
        rewards : [batch]  float32 tensor  (values in [0, 1])
        """
        batch   = len(rollout_text)
        rewards = torch.zeros(batch, dtype=torch.float32)

        # Determine mode per sample
        task_types  = None
        qa_answers  = None
        if ground_truth is not None:
            task_types = ground_truth.get("task_type", None)
            qa_answers = ground_truth.get("qa_answer",  None)

        for i, text in enumerate(rollout_text):
            is_qa = (
                task_types is not None
                and task_types[i] == "qa"
                and qa_answers is not None
            )

            if is_qa:
                # Free-form QA: score the <think> content against the reference
                hypothesis = extract_think_content(text)
                reference  = qa_answers[i] if isinstance(qa_answers, list) else qa_answers
                try:
                    rewards[i] = compute_rouge_score(hypothesis, reference)
                except Exception:
                    rewards[i] = 0.0
            else:
                # Coordinate mode: binary structural check
                rewards[i] = 1.0 if check_structural_format(text, K=self.K) else 0.0

        return rewards   # [batch]


# ---------------------------------------------------------------------------
# QAReward alias (used in train_stage2.py entry point)
# ---------------------------------------------------------------------------
# This is kept as a thin alias so the import in train_stage2.py is clean.
# The actual logic lives in FormatReward above.

QAReward = FormatReward