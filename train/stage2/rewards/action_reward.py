"""
action_reward.py
----------------
Visual trajectory reward for GRPO Teacher scoring.

Total reward:
    r = 0.9 * r_visual + 0.1 * r_format

Visual reward:
    r_visual = 0.5 * r_goal + 0.5 * r_traj

Goal reward (endpoint precision):
    r_goal = 0.5 * (f(p1, p̂1) + f(pK, p̂K))
    f(p, p') = max(0, 1 - ||p - p'||_2^2)

Trajectory reward (path geometry via DTW):
    r_traj = max(0, 1 - DTW(τ, τ̂))

Where:
    τ    — predicted waypoints parsed from the rollout text
    τ̂   — ground truth K=5 waypoints in normalised [0, 1] space
    p1   — start waypoint (index 0)
    pK   — end waypoint   (index K-1)

DTW is computed on CPU with a custom O(NK) implementation to avoid
dependencies on dtaidistance or similar. Euclidean distance between
2D points is used as the per-element cost.

Parsing:
    The Teacher generates waypoints in the <ans> tag as:
        <ans>x1,y1;x2,y2;x3,y3;x4,y4;x5,y5</ans>
    Coordinates are in [0,1] normalised space (matching Stage 1 convention).
    Parse failures (malformed output) receive r_visual = 0.0.
"""

import re
import torch
import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# DTW (pure Python/NumPy — no external dependency)
# ---------------------------------------------------------------------------

def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Standard DTW between two sequences of 2D points.

    Parameters
    ----------
    seq_a : [N, 2]  predicted waypoints
    seq_b : [M, 2]  ground truth waypoints

    Returns
    -------
    Normalised DTW distance in [0, ∞).
    Normalised by (N + M) so sequences of different lengths are comparable.
    """
    N, M = len(seq_a), len(seq_b)
    # Cost matrix: Euclidean distance between each pair of points
    cost = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            diff = seq_a[i] - seq_b[j]
            cost[i, j] = np.sqrt((diff * diff).sum())

    # Accumulated cost matrix
    acc = np.full((N, M), np.inf, dtype=np.float32)
    acc[0, 0] = cost[0, 0]

    for i in range(1, N):
        acc[i, 0] = cost[i, 0] + acc[i - 1, 0]
    for j in range(1, M):
        acc[0, j] = cost[0, j] + acc[0, j - 1]

    for i in range(1, N):
        for j in range(1, M):
            acc[i, j] = cost[i, j] + min(
                acc[i - 1, j],      # insertion
                acc[i, j - 1],      # deletion
                acc[i - 1, j - 1],  # match
            )

    # Normalise by path length
    return float(acc[N - 1, M - 1]) / (N + M)


# ---------------------------------------------------------------------------
# Waypoint parser
# ---------------------------------------------------------------------------

_ANS_PATTERN = re.compile(
    r"<ans>\s*([\d.]+,[\d.]+(?:;[\d.]+,[\d.]+)*)\s*</ans>",
    re.IGNORECASE,
)


def parse_waypoints(text: str, K: int = 5) -> Optional[np.ndarray]:
    """
    Parse K waypoints from a Teacher rollout string.

    Expected format inside <ans>...</ans>:
        x1,y1;x2,y2;...;xK,yK
    Coordinates assumed to be in [0, 1] normalised space.

    Returns
    -------
    waypoints : [K, 2] float32 ndarray, or None if parsing fails.
    """
    match = _ANS_PATTERN.search(text)
    if not match:
        return None

    try:
        pairs = match.group(1).strip().split(";")
        if len(pairs) != K:
            return None
        waypoints = []
        for pair in pairs:
            xy = pair.strip().split(",")
            if len(xy) != 2:
                return None
            x, y = float(xy[0]), float(xy[1])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return None         # out-of-range coordinate
            waypoints.append([x, y])
        return np.array(waypoints, dtype=np.float32)  # [K, 2]

    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

def compute_goal_reward(
    pred: np.ndarray,   # [K, 2]
    gt: np.ndarray,     # [K, 2]
) -> float:
    """
    r_goal = 0.5 * (f(p1, p̂1) + f(pK, p̂K))
    f(p, p') = max(0, 1 - ||p - p'||_2^2)

    Uses squared L2 distance — consistent with the paper's notation ||.||_2^2.
    """
    def f(p: np.ndarray, p_hat: np.ndarray) -> float:
        sq_dist = float(((p - p_hat) ** 2).sum())
        return max(0.0, 1.0 - sq_dist)

    r_start = f(pred[0],  gt[0])    # p1  vs p̂1
    r_end   = f(pred[-1], gt[-1])   # pK  vs p̂K
    return 0.5 * (r_start + r_end)


def compute_traj_reward(
    pred: np.ndarray,   # [K, 2]
    gt: np.ndarray,     # [K, 2]
) -> float:
    """
    r_traj = max(0, 1 - DTW(τ, τ̂))

    DTW distance is normalised by (N+M) so it's scale-independent.
    A perfect match (DTW=0) → r_traj=1.0.
    DTW ≥ 1 after normalisation → r_traj=0.0 (clipped).
    """
    d = dtw_distance(pred, gt)
    return max(0.0, 1.0 - d)


def compute_visual_reward(
    pred: np.ndarray,   # [K, 2]
    gt: np.ndarray,     # [K, 2]
) -> float:
    """
    r_visual = 0.5 * r_goal + 0.5 * r_traj
    """
    r_goal = compute_goal_reward(pred, gt)
    r_traj = compute_traj_reward(pred, gt)
    return 0.5 * r_goal + 0.5 * r_traj


# ---------------------------------------------------------------------------
# RewardFunction-compatible class
# ---------------------------------------------------------------------------

class ActionAlignedReward:
    """
    Computes r_visual = 0.5 * r_goal + 0.5 * r_traj for a batch of rollouts.

    Called by GRPOTeacher.score_rollouts() as one entry in reward_fns.
    The total reward r = 0.9 * r_visual + 0.1 * r_format is assembled in
    CombinedReward (see bottom of this file) which wraps both reward objects.

    ground_truth dict must contain:
        "gt_waypoints": torch.Tensor [batch, K, 2]  normalised [0,1]
    """

    def __init__(self, K: int = 5):
        self.K = K

    def __call__(
        self,
        rollout_ids,            # unused — we work from text
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
        assert ground_truth is not None and "gt_waypoints" in ground_truth, (
            "ActionAlignedReward requires ground_truth['gt_waypoints']"
        )

        gt_tensor = ground_truth["gt_waypoints"]   # [batch, K, 2]
        batch     = len(rollout_text)
        rewards   = torch.zeros(batch, dtype=torch.float32)

        for i, text in enumerate(rollout_text):
            pred = parse_waypoints(text, K=self.K)
            if pred is None:
                # Parse failure → zero visual reward
                rewards[i] = 0.0
                continue

            gt = gt_tensor[i].cpu().numpy().astype(np.float32)  # [K, 2]
            rewards[i] = compute_visual_reward(pred, gt)

        return rewards   # [batch]


# ---------------------------------------------------------------------------
# Combined reward  r = 0.9 * r_visual + 0.1 * r_format
# ---------------------------------------------------------------------------

class CombinedActionReward:
    """
    Assembles the total reward:
        r = 0.9 * r_visual + 0.1 * r_format

    Pass this as a SINGLE entry in reward_fns with weight=1.0 rather than
    passing ActionAlignedReward and QAReward separately, to ensure the
    0.9 / 0.1 split is always enforced at the reward level (not via
    reward_weights, which are applied per-function).

    Parameters
    ----------
    visual_reward : ActionAlignedReward instance
    format_reward : FormatReward instance (imported from qa_reward.py)
    """

    def __init__(self, visual_reward: ActionAlignedReward, format_reward):
        self.visual = visual_reward
        self.format = format_reward

    def __call__(
        self,
        rollout_ids,
        rollout_text: List[str],
        pixel_values=None,
        image_grid_thw=None,
        ground_truth: dict = None,
    ) -> torch.Tensor:
        r_visual = self.visual(
            rollout_ids, rollout_text,
            pixel_values, image_grid_thw, ground_truth,
        )
        r_format = self.format(
            rollout_ids, rollout_text,
            pixel_values, image_grid_thw, ground_truth,
        )
        return 0.9 * r_visual + 0.1 * r_format   # [batch]