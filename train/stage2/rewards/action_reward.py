"""
action_reward.py  (ThinkFlow-VLA — Stage 2 GRPO)
-------------------------------------------------
Visual trajectory reward for GRPO Teacher scoring.

Total reward:
    r = 0.9 * r_visual + 0.1 * r_format

Visual reward:
    r_visual = 0.5 * r_goal + 0.5 * r_traj

Goal reward (endpoint precision):
    r_goal = 0.5 * (f(p1, p̂1) + f(pK, p̂K))
    f(p, p') = max(0, 1 - ||p - p'||_2^2)

Trajectory reward (path geometry via DTW):
    r_traj = max(0, 1 - DTW_norm(τ, τ̂))

    DTW is normalised by max(N, M) * sqrt(2).
    Previous normalisation was (N + M) which left a mean r_traj ≈ 0.76
    for completely random predictions — nearly half the useful reward range
    was wasted as a floor that provided no learning signal.

Parser (enforced strictly):
    The Teacher must produce:
        <think> ... non-trivial reasoning ... </think>
        followed by EITHER:
            <ans>x1,y1;x2,y2;x3,y3;x4,y4;x5,y5</ans>    ← preferred
            [x1,y1] [x2,y2] [x3,y3] [x4,y4] [x5,y5]    ← fallback

    Hard parse rules (any failure → None → 0 visual reward):
      1. Both <think> and </think> tags present
      2. Think content ≥ 20 chars (prevents trivially empty reasoning)
      3. Exactly K coordinate pairs (was: take-last-K if ≥ K — exploitable)
      4. Auto-normalise only when arr.max() > 2.0 (was: > 1.0, which
         incorrectly rescaled values like [1.05, 0.98] by 1/1000)
      5. Trajectory diversity: arr.std() ≥ 0.01
         (was: no check — model could output [0.5,0.5]×5 and collect
         mean r_traj ≈ 0.81 for the collapsed single-point prediction)

Coordinate space: normalised [0, 1] matching Stage 1 convention.
Parse failures → r_visual = 0.0.
"""

import re
import torch
import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------------
# DTW (pure NumPy — no external dependency)
# ---------------------------------------------------------------------------

def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    DTW distance normalised to ≈ [0, 1].

    Normaliser: max(N, M) * sqrt(2)
        — max(N,M) is the minimum-length warping path (diagonal case).
        — sqrt(2) is the maximum Euclidean distance between two points
          in the unit square [0,1]².
    Together they give the theoretical upper bound on the diagonal-path DTW,
    anchoring the scale so a perfect match returns 0 and a worst-case match
    returns ≥ 1 (capped at 1.0).

    OLD: normalised by (N+M) → mean r_traj ≈ 0.76 for random predictions.
    NEW: normalised by max(N,M)*√2 → mean r_traj ≈ 0.66 for random predictions.
    The remaining floor (~0.66) is acceptable: GRPO advantage normalisation
    (A = r − mean_group) removes it within each rollout group.

    Parameters
    ----------
    seq_a : [N, 2]  predicted waypoints  (float32, values in [0,1])
    seq_b : [M, 2]  ground-truth waypoints

    Returns
    -------
    Normalised DTW distance capped at 1.0.
    """
    N, M = len(seq_a), len(seq_b)

    # Per-pair Euclidean cost matrix
    cost = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            diff = seq_a[i] - seq_b[j]
            cost[i, j] = np.sqrt((diff * diff).sum())

    # Accumulated cost via DP
    acc = np.full((N, M), np.inf, dtype=np.float32)
    acc[0, 0] = cost[0, 0]
    for i in range(1, N):
        acc[i, 0] = cost[i, 0] + acc[i - 1, 0]
    for j in range(1, M):
        acc[0, j] = cost[0, j] + acc[0, j - 1]
    for i in range(1, N):
        for j in range(1, M):
            acc[i, j] = cost[i, j] + min(
                acc[i - 1, j],       # insertion
                acc[i, j - 1],       # deletion
                acc[i - 1, j - 1],   # match
            )

    normaliser = max(N, M) * float(np.sqrt(2))
    return min(1.0, float(acc[N - 1, M - 1]) / normaliser)


# ---------------------------------------------------------------------------
# Waypoint parser
# ---------------------------------------------------------------------------

_THINK_OPEN    = re.compile(r'<think>',   re.IGNORECASE)
_THINK_CLOSE   = re.compile(r'</think>',  re.IGNORECASE)
_THINK_CONTENT = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)

# Preferred answer format:  <ans>x,y;x,y;x,y;x,y;x,y</ans>
_ANS_TAG = re.compile(
    r'<ans>\s*([\d.]+,[\d.]+(?:;[\d.]+,[\d.]+)*)\s*</ans>',
    re.IGNORECASE,
)

# Fallback answer format:  [x, y]  ...  [x, y]
_BRACKET_PAIR = re.compile(
    r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',
)


def parse_waypoints(text: str, K: int = 5) -> Optional[np.ndarray]:
    """
    Parse exactly K waypoints from a Teacher rollout. Returns None on failure.

    Guards (in order; any failure → None → 0 visual reward):
      1. Both <think> and </think> tags present.
      2. Think content ≥ 20 chars — prevents empty/trivial reasoning.
      3. Exactly K coordinate pairs after </think> — strict count, no overflow.
      4. Auto-normalise threshold 2.0 (was 1.0) — prevents mishandling values
         like [1.05, 0.98] which are slightly out of [0,1] but not 1000-scale.
      5. arr.std() ≥ 0.01 — rejects degenerate single-point collapsed output.

    Supports both <ans>x,y;...</ans> (preferred) and [x,y] bracket format.
    """
    # 1. End think tag required
    if not _THINK_CLOSE.search(text):
        return None

    # 2. Non-trivial think content
    parts = re.split(r'</think>', text, maxsplit=1, flags=re.IGNORECASE)
    think_content = _THINK_OPEN.sub('', parts[0]).strip()
    if len(think_content) < 20:
        return None

    # Search only after </think>
    after_think = text.split("</think>", 1)[-1]

    # 3a. Try <ans>x,y;x,y</ans> format first (canonical)
    ans_m = _ANS_TAG.search(after_think)
    if ans_m:
        raw_pairs = ans_m.group(1).split(";")
        try:
            split_pairs = [p.strip().split(",") for p in raw_pairs]
            if len(split_pairs) != K:
                return None
            waypoints = [[float(x), float(y)] for x, y in split_pairs]
        except (ValueError, TypeError, IndexError):
            return None
    else:
        # 3b. Fall back to [x, y] bracket format
        found = _BRACKET_PAIR.findall(after_think)
        if len(found) != K:   # strict: exactly K — no take-last-K
            return None
        try:
            waypoints = [[float(x), float(y)] for x, y in found]
        except ValueError:
            return None

    arr = np.array(waypoints, dtype=np.float32)  # [K, 2]

    # 4. Normalise: only when clearly 0-1000 scale (threshold 2.0, not 1.0)
    #    Avoids mishandling coords like [1.05, 0.98] → [0.001, 0.001]
    if arr.max() > 2.0:
        arr = arr / 1000.0
    arr = np.clip(arr, 0.0, 1.0)

    # 5. Diversity check — reject collapsed trajectories
    if float(arr.std()) < 0.01:
        return None

    return arr   # [K, 2]


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

def compute_goal_reward(
    pred: np.ndarray,   # [K, 2]
    gt:   np.ndarray,   # [K, 2]
) -> float:
    """
    r_goal = 0.5 * (f(p1, p̂1) + f(pK, p̂K))
    f(p, p') = max(0, 1 - ||p - p'||_2^2)

    Squared L2 distance gives the correct scale for [0,1]² coordinates:
    — Same point              →  sq_dist = 0      → reward = 1.0
    — Full single-axis offset →  sq_dist = 1.0    → reward = 0.0
    — Diagonal (worst case)   →  sq_dist = 2.0    → reward = 0.0  (clipped)
    """
    def f(p: np.ndarray, p_hat: np.ndarray) -> float:
        sq_dist = float(((p - p_hat) ** 2).sum())
        return max(0.0, 1.0 - sq_dist)

    return 0.5 * (f(pred[0], gt[0]) + f(pred[-1], gt[-1]))


def compute_traj_reward(
    pred: np.ndarray,   # [K, 2]
    gt:   np.ndarray,   # [K, 2]
) -> float:
    """
    r_traj = max(0, 1 - DTW_norm(τ, τ̂))
    Uses improved DTW normalisation (see dtw_distance docstring).
    """
    return max(0.0, 1.0 - dtw_distance(pred, gt))


def compute_visual_reward(
    pred: np.ndarray,   # [K, 2]
    gt:   np.ndarray,   # [K, 2]
) -> float:
    """r_visual = 0.5 * r_goal + 0.5 * r_traj"""
    return 0.5 * compute_goal_reward(pred, gt) + 0.5 * compute_traj_reward(pred, gt)


# ---------------------------------------------------------------------------
# RewardFunction-compatible classes
# ---------------------------------------------------------------------------

class ActionAlignedReward:
    """
    Computes r_visual for a batch of rollouts.

    ground_truth dict must contain:
        "gt_waypoints": torch.Tensor [batch, K, 2]  normalised [0,1]
    """

    def __init__(self, K: int = 5):
        self.K = K

    def __call__(
        self,
        rollout_ids,
        rollout_text:   List[str],
        pixel_values    = None,
        image_grid_thw  = None,
        ground_truth:   dict = None,
        pixel_values_videos = None,
        video_grid_thw = None,
    ) -> torch.Tensor:
        assert ground_truth is not None and "gt_waypoints" in ground_truth, (
            "ActionAlignedReward requires ground_truth['gt_waypoints']"
        )
        gt_tensor = ground_truth["gt_waypoints"]   # [batch, K, 2]
        batch     = len(rollout_text)
        rewards   = torch.zeros(batch, dtype=torch.float32)

        for i, text in enumerate(rollout_text):
            pred = parse_waypoints(text, K=self.K)
            if pred is None:
                rewards[i] = 0.0
                continue
            gt = gt_tensor[i].cpu().numpy().astype(np.float32)
            rewards[i] = compute_visual_reward(pred, gt)

        return rewards


class CombinedActionReward:
    """
    Assembles r = 0.9 * r_visual + 0.1 * r_format.

    Pass as a single entry in reward_fns (weight=1.0) to enforce the
    0.9 / 0.1 split at the reward level rather than via reward_weights.
    """

    def __init__(self, visual_reward: ActionAlignedReward, format_reward):
        self.visual = visual_reward
        self.format = format_reward

    def __call__(
        self,
        rollout_ids,
        rollout_text:   List[str],
        pixel_values    = None,
        image_grid_thw  = None,
        ground_truth:   dict = None,
        pixel_values_videos = None,
        video_grid_thw = None,
    ) -> torch.Tensor:
        r_visual = self.visual(
            rollout_ids, rollout_text,
            pixel_values, image_grid_thw, ground_truth,
            pixel_values_videos, video_grid_thw,
        )
        r_format = self.format(
            rollout_ids, rollout_text,
            pixel_values, image_grid_thw, ground_truth,
            pixel_values_videos, video_grid_thw,
        )

        batch = len(rollout_text)
        device = r_visual.device if hasattr(r_visual, 'device') else torch.device("cpu")
        final_rewards = torch.zeros(batch, dtype=torch.float32, device=device)

        task_types = None
        if ground_truth is not None:
            task_types = ground_truth.get("task_type", None)

        for i in range(batch):
            tt = task_types[i] if task_types else "trajectory"
            if tt == "qa":
                final_rewards[i] = r_format[i]
            else:
                final_rewards[i] = 0.9 * r_visual[i] + 0.1 * r_format[i]

        return final_rewards