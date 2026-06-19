"""
qa_reward.py  (ThinkFlow-VLA — Stage 2 GRPO)
--------------------------------------------
Format correctness reward (r_format) for GRPO Teacher scoring.

r_format definition
-------------------
This file was completely rearchitected. The original check_structural_format
only required </think> to be present in < 2000 chars, meaning the model could
emit bare "</think>" and collect the full 0.1 format reward for free every step.

New design — graduated reward (0.0 / 0.5 / 1.0 × length_factor):

    0.0  ← missing <think> or </think>, OR think content < 20 chars
           Rationale: without a real reasoning block there is no format at all.

    0.5  ← valid <think>...</think> with ≥ 20 chars content,
           but answer block missing or unparseable.
           Rationale: partial credit keeps a gradient toward correct format
           even when coordinate prediction is wrong (prevents total reward
           sparsity early in training).

    1.0  ← full valid structure: <think>...</think> + exactly K valid
           coordinate pairs in either <ans>x,y;...</ans> or [x,y] format.
           Rationale: only reward 1.0 when the output is actually usable.

Length factor (multiplied into the score):
    len ≤ 2000 chars       →  ×1.00
    2000 < len ≤ 4000      →  linearly decays ×1.00 → ×0.50
    len > 4000             →  ×0.50
    Old: hard cutoff at 2000 chars returned 0.0 — abrupt discontinuity
    in reward that made the length boundary hard to learn.

QA mode (RoboVQA, RoboFAC, EgoPlan):
    Still uses ROUGE-1/2/L average, but now also requires valid think
    structure (both tags + ≥ 20 chars). Previously ROUGE was computed
    even on outputs with no tags at all.

ROUGE: pip install rouge-score
"""

import re
from typing import List, Optional

import torch

try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared patterns (consistent with action_reward.py)
# ---------------------------------------------------------------------------

_THINK_OPEN    = re.compile(r'<think>',   re.IGNORECASE)
_THINK_CLOSE   = re.compile(r'</think>',  re.IGNORECASE)
_THINK_CONTENT = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)

# Preferred: <ans>x,y;x,y;x,y;x,y;x,y</ans>
_ANS_TAG = re.compile(
    r'<ans>\s*([\d.]+,[\d.]+(?:;[\d.]+,[\d.]+)*)\s*</ans>',
    re.IGNORECASE,
)

# Fallback: [x, y] bracket pairs
_BRACKET_PAIR = re.compile(
    r'\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]',
)

# Bounding box pattern: [xmin, ymin, xmax, ymax]
_BBOX_RE = re.compile(
    r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
)

def parse_bbox(text) -> Optional[List[float]]:
    if isinstance(text, (list, tuple)):
        if len(text) == 4 and all(isinstance(x, (int, float)) for x in text):
            return [float(x) for x in text]
        return None
    if not text or not isinstance(text, str):
        return None
    m = _BBOX_RE.search(text)
    if not m:
        return None
    try:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]
    except Exception:
        return None

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two boxes."""
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    x_left = max(xmin1, xmin2)
    y_top = max(ymin1, ymin2)
    x_right = min(xmax1, xmax2)
    y_bottom = min(ymax1, ymax2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    union_area = box1_area + box2_area - intersection_area
    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


# ---------------------------------------------------------------------------
# Structural format checker
# ---------------------------------------------------------------------------

def check_structural_format(text: str, K: int = 5) -> float:
    """
    Graduated format reward: 0.0 / 0.5 / 1.0 × length_factor.

    Level 0 (return 0.0) — hard failures:
        - <think> tag missing
        - </think> tag missing
        - Think content < 20 chars (catches bare "</think>" with no reasoning)

    Level 1 (return 0.5 × length_factor) — partial credit:
        - Think structure valid but answer block absent or unparseable.
        - Keeps reward signal alive for format learning when the model is
          still developing coordinate prediction ability.

    Level 2 (return 1.0 × length_factor) — full credit:
        - Think structure valid AND exactly K coordinate pairs found in
          either <ans>x,y;x,y</ans> or [x,y] bracket format.

    Length factor:
        chars ≤ 2000         → 1.00
        2000 < chars ≤ 4000  → linear decay 1.00 → 0.50
        chars > 4000         → 0.50
    """
    # --- Level 0: think structure (hard requirement) ---
    if not _THINK_CLOSE.search(text):
        return 0.0

    parts = re.split(r'</think>', text, maxsplit=1, flags=re.IGNORECASE)
    think_content = _THINK_OPEN.sub('', parts[0]).strip()
    if len(think_content) < 20:
        return 0.0

    # --- Soft length penalty ---
    clean_text = text
    for stop_token in ["<|im_end|>", "<|vision_pad|>", "<|pad|>", "<|endoftext|>"]:
        clean_text = clean_text.split(stop_token)[0]
    char_len = len(clean_text.strip())

    if char_len > 4000:
        length_factor = 0.5
    elif char_len > 2000:
        # Linear decay: 1.0 at 2000 chars → 0.5 at 4000 chars
        length_factor = 1.0 - 0.5 * (char_len - 2000) / 2000.0
    else:
        length_factor = 1.0

    # --- Duplicate </think> penalty (strict 0.0) ---
    num_close_tags = len(_THINK_CLOSE.findall(text))
    if num_close_tags > 1:
        return 0.0

    # --- Level 1 / 2: check for answer block after </think> ---
    after_think = parts[1]

    # Try <ans> tag first
    ans_m = _ANS_TAG.search(after_think)
    if ans_m:
        raw_pairs = ans_m.group(1).split(";")
        try:
            split_pairs = [p.strip().split(",") for p in raw_pairs]
            coords = [[float(x), float(y)] for x, y in split_pairs]
        except (ValueError, TypeError, IndexError):
            # Tag present but content unparseable → partial credit
            return 0.5 * length_factor

        if len(coords) != K:
            return 0.5 * length_factor

        # All coordinates in a recognised range?
        flat = [v for pair in coords for v in pair]
        in_unit  = all(0.0 <= v <= 1.0    for v in flat)
        in_kilo  = all(0.0 <= v <= 1000.0 for v in flat)
        if not (in_unit or in_kilo):
            return 0.5 * length_factor

        return 1.0 * length_factor

    # Try bracket fallback [x, y]
    found = _BRACKET_PAIR.findall(after_think)
    if len(found) == K:
        return 1.0 * length_factor

    # Think structure valid, answer absent
    return 0.5 * length_factor


# ---------------------------------------------------------------------------
# ROUGE scorer (lazy singleton)
# ---------------------------------------------------------------------------

_SCORER = None

def _get_scorer():
    global _SCORER
    if _SCORER is None:
        if not _ROUGE_AVAILABLE:
            raise ImportError(
                "rouge_score required for QA reward — pip install rouge-score"
            )
        _SCORER = _rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
        )
    return _SCORER


def compute_rouge_score(hypothesis: str, reference: str) -> float:
    """Average of ROUGE-1, ROUGE-2, ROUGE-L F1.  Returns float in [0, 1]."""
    if not hypothesis or not reference or not isinstance(hypothesis, str) or not isinstance(reference, str):
        return 0.0
    scorer = _get_scorer()
    scores = scorer.score(reference, hypothesis)
    return (
        scores["rouge1"].fmeasure
        + scores["rouge2"].fmeasure
        + scores["rougeL"].fmeasure
    ) / 3.0


def extract_think_content(text: str) -> str:
    """Return stripped content of the first <think>...</think> block."""
    if not _THINK_CLOSE.search(text):
        return text.strip()
    parts = re.split(r'</think>', text, maxsplit=1, flags=re.IGNORECASE)
    return _THINK_OPEN.sub('', parts[0]).strip()


# ---------------------------------------------------------------------------
# Format reward class
# ---------------------------------------------------------------------------

class FormatReward:
    """
    Computes r_format for a batch of rollout texts.

    Two modes per sample:
        Coordinate mode  (ground_truth has "gt_waypoints" but no "qa_answer"):
            r_format = graduated structural check (0.0 / 0.5 / 1.0 × len_factor)

        QA mode  (ground_truth has "qa_answer"):
            r_format = average ROUGE-1/2/L against the reference answer.
            Requires valid think structure — returns 0.0 if tags missing or
            content < 20 chars (was: ROUGE was computed unconditionally).

    ground_truth dict keys:
        "gt_waypoints"  : [batch, K, 2]  — coordinate supervision
        "qa_answer"     : List[str]       — QA reference answers
        "task_type"     : List[str]       — "waypoint" or "qa" per sample
    """

    def __init__(self, K: int = 5):
        self.K = K

    def __call__(
        self,
        rollout_ids,
        rollout_text:  List[str],
        pixel_values   = None,
        image_grid_thw = None,
        ground_truth:  dict = None,
        pixel_values_videos = None,
        video_grid_thw = None,
    ) -> torch.Tensor:
        batch   = len(rollout_text)
        rewards = torch.zeros(batch, dtype=torch.float32)

        task_types = None
        qa_answers = None
        datasets = None
        if ground_truth is not None:
            task_types = ground_truth.get("task_type", None)
            qa_answers = ground_truth.get("qa_answer",  None)
            datasets   = ground_truth.get("dataset",    None)

        for i, text in enumerate(rollout_text):
            is_qa = (
                task_types is not None
                and task_types[i] == "qa"
                and qa_answers is not None
            )

            if is_qa:
                # Require EXACTLY ONE valid think structure before awarding ROUGE score
                close_tags = _THINK_CLOSE.findall(text)
                if len(close_tags) != 1:
                    rewards[i] = 0.0
                    continue
                parts = re.split(r'</think>', text, maxsplit=1, flags=re.IGNORECASE)
                think_content = _THINK_OPEN.sub('', parts[0]).strip()
                if len(think_content) < 20:
                    rewards[i] = 0.0
                    continue

                # Strip generation padding/end tokens from the hypothesis
                hypothesis = parts[1].split('<|im_end|>')[0].split('<|vision_pad|>')[0].strip()
                
                reference  = (
                    qa_answers[i] if isinstance(qa_answers, list) else qa_answers
                )
                
                is_affordance = (
                    datasets is not None
                    and i < len(datasets)
                    and datasets[i] == "sharerobot_affordance"
                )

                if is_affordance:
                    box_hyp = parse_bbox(hypothesis)
                    box_ref = parse_bbox(reference)
                    if box_hyp is not None and box_ref is not None:
                        qa_score = compute_iou(box_hyp, box_ref)
                    else:
                        qa_score = 0.0
                else:
                    try:
                        qa_score = compute_rouge_score(hypothesis, reference)
                    except Exception:
                        qa_score = 0.0

                clean_text = text
                for stop_token in ["<|im_end|>", "<|vision_pad|>", "<|pad|>", "<|endoftext|>"]:
                    clean_text = clean_text.split(stop_token)[0]
                char_len = len(clean_text.strip())

                if char_len > 4000:
                    length_factor = 0.5
                elif char_len > 2000:
                    length_factor = 1.0 - 0.5 * (char_len - 2000) / 2000.0
                else:
                    length_factor = 1.0

                r_format = length_factor
                rewards[i] = 0.8 * qa_score + 0.2 * r_format
            else:
                rewards[i] = check_structural_format(text, K=self.K)

        return rewards


# ---------------------------------------------------------------------------
# Alias kept for backward compatibility with train_stage2.py
# ---------------------------------------------------------------------------

QAReward = FormatReward