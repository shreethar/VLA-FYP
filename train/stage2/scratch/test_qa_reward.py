import sys
sys.path.append('/home/ubuntu/VLA-FYP/train/stage2')

from rewards.qa_reward import FormatReward, compute_rouge_score

# Mocking ground truth answers
ground_truth = {
    "task_type": ["qa"],
    "qa_answer": ["[631, 211, 702, 284]"]
}

# The rollout text matching what the user posted (with padding)
rollout_text = [
    "The task is to open the door on the left, which means identifying the door handle that corresponds to the left-hand door. In the image, there are two door handles, one on the left door and one on the right door. The left door handle is located at approximately [631, 211, 702, 284]. The purpose of the bounding box is to encapsulate the handle so that a grasping mechanism can easily identify it. The coordinates were determined based on the visual evidence in the image, focusing on the handle that would be used to open the left door.\n</think>\n\n[631, 211, 702, 284]<|im_end|><|vision_pad|>" + "<|vision_pad|>" * 3000
]

reward_fn = FormatReward(K=5)
rewards = reward_fn(
    rollout_ids=None,
    rollout_text=rollout_text,
    ground_truth=ground_truth
)

print("Hypothesis: [631, 211, 702, 284]")
print("Reference:  [631, 211, 702, 284]")
print("ROUGE score:", compute_rouge_score("[631, 211, 702, 284]", "[631, 211, 702, 284]"))
print("Final Reward (Expected 1.0):", rewards[0].item())
