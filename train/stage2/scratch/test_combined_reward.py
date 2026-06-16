import sys
sys.path.append('/home/ubuntu/VLA-FYP/train/stage2')

import torch
from rewards.action_reward import ActionAlignedReward, CombinedActionReward
from rewards.qa_reward import FormatReward

visual = ActionAlignedReward(K=5)
fmt = FormatReward(K=5)
comb = CombinedActionReward(visual, fmt)

# Mock ground truth coordinates (5 points)
ground_truth = {
    "task_type": ["trajectory", "trajectory", "trajectory"],
    "gt_waypoints": torch.tensor([
        [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]],
        [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]],
        [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]]
    ])
}

rollout_text = [
    # 0. Perfect match, valid reasoning
    "Reasoning about the action.\n</think>\n\n[0.1, 0.2] [0.2, 0.3] [0.3, 0.4] [0.4, 0.5] [0.5, 0.6]<|im_end|>",
    # 1. Missing close think tag
    "Reasoning about the action but no end tag. [0.1, 0.2] [0.2, 0.3] [0.3, 0.4] [0.4, 0.5] [0.5, 0.6]<|im_end|>",
    # 2. Duplicate close think tag
    "Reasoning about the action. </think> </think> [0.1, 0.2] [0.2, 0.3] [0.3, 0.4] [0.4, 0.5] [0.5, 0.6]<|im_end|>"
]

rewards = comb(
    rollout_ids=None,
    rollout_text=rollout_text,
    ground_truth=ground_truth
)

print("Rollout 0 (Perfect match, valid reasoning):", rewards[0].item())
print("Rollout 1 (Missing close think tag):        ", rewards[1].item())
print("Rollout 2 (Duplicate close think tag):      ", rewards[2].item())
