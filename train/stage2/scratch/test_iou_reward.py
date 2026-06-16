import sys
sys.path.append('/home/ubuntu/VLA-FYP/train/stage2')

from rewards.qa_reward import FormatReward, compute_iou

# Test individual IoU logic
box1 = [658, 460, 706, 596]
box2 = [616, 539, 683, 604]
print("Partial overlap IoU:", compute_iou(box1, box2))
print("Exact match IoU:    ", compute_iou(box1, box1))
print("No overlap IoU:     ", compute_iou(box1, [0, 0, 10, 10]))

# Test FormatReward interface with datasets="sharerobot_affordance"
ground_truth = {
    "task_type": ["qa", "qa"],
    "qa_answer": ["[658, 460, 706, 596]", "[658, 460, 706, 596]"],
    "dataset": ["sharerobot_affordance", "sharerobot_affordance"]
}

rollout_text = [
    # Rollout 0: matching bbox
    "The target object is located at this box.\n</think>\n\n[658, 460, 706, 596]<|im_end|>",
    # Rollout 1: partially overlapping bbox
    "The target object is located at this box.\n</think>\n\n[616, 539, 683, 604]<|im_end|>"
]

reward_fn = FormatReward(K=5)
rewards = reward_fn(
    rollout_ids=None,
    rollout_text=rollout_text,
    ground_truth=ground_truth
)

print("Rollout 0 Reward (Expected 1.0):", rewards[0].item())
print("Rollout 1 Reward (Expected ~0.32):", rewards[1].item())
