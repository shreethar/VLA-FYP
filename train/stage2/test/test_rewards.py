"""
test_rewards.py
---------------
Tests for rewards/action_reward.py and rewards/qa_reward.py.

These are pure CPU tests — no model loading required.
Run:  python -m pytest test/test_rewards.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np

from rewards.action_reward import (
    dtw_distance,
    parse_waypoints,
    compute_goal_reward,
    compute_traj_reward,
    compute_visual_reward,
    ActionAlignedReward,
    CombinedActionReward,
)
from rewards.qa_reward import (
    check_structural_format,
    FormatReward,
    extract_think_content,
)


# ===========================================================================
# DTW
# ===========================================================================

class TestDTW:
    def test_identical_sequences(self):
        """DTW of identical sequences should be 0."""
        seq = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        assert dtw_distance(seq, seq) == pytest.approx(0.0, abs=1e-6)

    def test_different_sequences(self):
        """DTW of different sequences should be positive."""
        a = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        d = dtw_distance(a, b)
        assert d > 0

    def test_single_point(self):
        """DTW of two single points = euclidean / 2."""
        a = np.array([[0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.3, 0.4]], dtype=np.float32)
        # distance = 0.5, normalised by (1+1) = 2 → 0.25
        d = dtw_distance(a, b)
        assert d == pytest.approx(0.5 / 2, abs=1e-5)

    def test_symmetry(self):
        """DTW should be symmetric."""
        a = np.random.rand(4, 2).astype(np.float32)
        b = np.random.rand(3, 2).astype(np.float32)
        assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a), abs=1e-5)


# ===========================================================================
# Waypoint parsing
# ===========================================================================

class TestParseWaypoints:
    def test_valid_5_waypoints(self):
        text = "<think>reasoning</think><ans>0.1,0.2;0.3,0.4;0.5,0.6;0.7,0.8;0.9,0.1</ans>"
        wp = parse_waypoints(text, K=5)
        assert wp is not None
        assert wp.shape == (5, 2)
        np.testing.assert_allclose(wp[0], [0.1, 0.2], atol=1e-6)

    def test_wrong_count(self):
        text = "<ans>0.1,0.2;0.3,0.4</ans>"
        assert parse_waypoints(text, K=5) is None

    def test_out_of_range(self):
        text = "<ans>1.5,0.2;0.3,0.4;0.5,0.6;0.7,0.8;0.9,0.1</ans>"
        assert parse_waypoints(text, K=5) is None

    def test_malformed(self):
        text = "no tags here"
        assert parse_waypoints(text, K=5) is None

    def test_whitespace_tolerant(self):
        text = "<ans> 0.1,0.2 ; 0.3,0.4 ; 0.5,0.6 ; 0.7,0.8 ; 0.9,0.1 </ans>"
        wp = parse_waypoints(text, K=5)
        assert wp is not None
        assert wp.shape == (5, 2)


# ===========================================================================
# Goal / trajectory / visual reward
# ===========================================================================

class TestGoalReward:
    def test_perfect_match(self):
        wp = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]], dtype=np.float32)
        assert compute_goal_reward(wp, wp) == pytest.approx(1.0, abs=1e-6)

    def test_distant_endpoints(self):
        pred = np.array([[0.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
        gt   = np.array([[1.0, 1.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
        r = compute_goal_reward(pred, gt)
        # Both endpoints are 2.0 away (squared), so f = max(0, 1 - 2) = 0
        assert r == pytest.approx(0.0, abs=1e-6)


class TestTrajReward:
    def test_perfect_match(self):
        wp = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        assert compute_traj_reward(wp, wp) == pytest.approx(1.0, abs=1e-6)

    def test_bounded_0_1(self):
        pred = np.random.rand(5, 2).astype(np.float32)
        gt   = np.random.rand(5, 2).astype(np.float32)
        r = compute_traj_reward(pred, gt)
        assert 0.0 <= r <= 1.0


class TestVisualReward:
    def test_perfect_match(self):
        wp = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]], dtype=np.float32)
        r = compute_visual_reward(wp, wp)
        assert r == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# ActionAlignedReward class
# ===========================================================================

class TestActionAlignedReward:
    def test_batch_scoring(self):
        reward_fn = ActionAlignedReward(K=5)
        texts = [
            "<think>ok</think><ans>0.1,0.2;0.3,0.4;0.5,0.6;0.7,0.8;0.9,0.1</ans>",
            "malformed output",
        ]
        gt_wp = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]],
        ], dtype=torch.float32)

        rewards = reward_fn(
            rollout_ids=None,
            rollout_text=texts,
            ground_truth={"gt_waypoints": gt_wp},
        )
        assert rewards.shape == (2,)
        assert rewards[0] > 0.9    # near perfect
        assert rewards[1] == 0.0   # parse failure


# ===========================================================================
# Format checking (qa_reward.py)
# ===========================================================================

class TestStructuralFormat:
    def test_valid(self):
        text = "<think>some reasoning</think><ans>0.1,0.2;0.3,0.4;0.5,0.6;0.7,0.8;0.9,0.1</ans>"
        assert check_structural_format(text, K=5) is True

    def test_missing_think(self):
        text = "<ans>0.1,0.2;0.3,0.4;0.5,0.6;0.7,0.8;0.9,0.1</ans>"
        assert check_structural_format(text, K=5) is False

    def test_empty_think(self):
        text = "<think></think><ans>0.1,0.2;0.3,0.4;0.5,0.6;0.7,0.8;0.9,0.1</ans>"
        assert check_structural_format(text, K=5) is False

    def test_extra_text_after_ans(self):
        text = "<think>ok</think><ans>0.1,0.2;0.3,0.4;0.5,0.6;0.7,0.8;0.9,0.1</ans> extra"
        assert check_structural_format(text, K=5) is False

    def test_wrong_k(self):
        text = "<think>ok</think><ans>0.1,0.2;0.3,0.4</ans>"
        assert check_structural_format(text, K=5) is False


class TestExtractThinkContent:
    def test_extraction(self):
        text = "<think>the reasoning goes here</think><ans>stuff</ans>"
        assert extract_think_content(text) == "the reasoning goes here"

    def test_no_think_tag(self):
        text = "just plain text"
        assert extract_think_content(text) == "just plain text"


class TestFormatReward:
    def test_waypoint_mode(self):
        reward_fn = FormatReward(K=5)
        texts = [
            "<think>ok</think><ans>0.1,0.2;0.3,0.4;0.5,0.6;0.7,0.8;0.9,0.1</ans>",
            "malformed",
        ]
        gt = {"gt_waypoints": torch.rand(2, 5, 2), "task_type": ["waypoint", "waypoint"]}
        rewards = reward_fn(None, texts, ground_truth=gt)
        assert rewards[0] == 1.0
        assert rewards[1] == 0.0

    def test_qa_mode(self):
        """QA mode requires rouge_score. Skip if unavailable."""
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            pytest.skip("rouge_score not installed")

        reward_fn = FormatReward(K=5)
        texts = ["<think>the cat sat on the mat</think><ans>0.1,0.2</ans>"]
        gt = {
            "task_type": ["qa"],
            "qa_answer": ["the cat sat on the mat"],
        }
        rewards = reward_fn(None, texts, ground_truth=gt)
        assert rewards[0] > 0.5   # high ROUGE against identical text


# ===========================================================================
# CombinedActionReward
# ===========================================================================

class TestCombinedActionReward:
    def test_combined_weights(self):
        visual_fn = ActionAlignedReward(K=5)
        format_fn = FormatReward(K=5)
        combined  = CombinedActionReward(visual_fn, format_fn)

        texts = [
            "<think>reason</think><ans>0.1,0.2;0.3,0.4;0.5,0.6;0.7,0.8;0.9,0.1</ans>",
        ]
        gt_wp = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]],
        ], dtype=torch.float32)
        gt = {
            "gt_waypoints": gt_wp,
            "task_type": ["waypoint"],
        }

        r = combined(None, texts, ground_truth=gt)
        assert r.shape == (1,)
        # r_visual ≈ 1.0 (perfect), r_format = 1.0 (valid struct)
        # total = 0.9 * 1.0 + 0.1 * 1.0 = 1.0
        assert r[0] == pytest.approx(1.0, abs=0.05)
