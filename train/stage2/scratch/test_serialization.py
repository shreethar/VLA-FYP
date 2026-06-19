import torch
import os
import shutil

def run_test():
    # 1. Create a dummy output dir
    test_dir = "checkpoints/test_serialization"
    os.makedirs(test_dir, exist_ok=True)
    
    # 2. Mock identifiers
    global_step = 42
    micro_step = 1
    sample_ids = ["traj_dataset_123", "qa_dataset_456"]

    # Mock inputs
    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones_like(input_ids)
    image_grid_thw = torch.tensor([[1, 28, 28]])
    video_grid_thw = None
    
    pixel_values = torch.randn(1, 28 * 28, 64)
    pixel_values_videos = None

    # Mock supervision
    gt_waypoints = torch.randn(2, 6, 2)
    ground_truth = {
        "gt_waypoints": gt_waypoints,
        "task_type": ["trajectory", "qa"],
        "qa_answer": [None, "yes"],
        "dataset": ["traj_dataset", "qa_dataset"],
        "sample_ids": sample_ids,
    }

    # Mock RolloutBuffer
    class MockBuffer:
        def __init__(self):
            self.tau_pos_ids = torch.randint(0, 1000, (2, 32))
            self.tau_pos_mask = torch.ones_like(self.tau_pos_ids)
            self.tau_neg_ids = torch.randint(0, 1000, (2, 32))
            self.tau_neg_mask = torch.ones_like(self.tau_neg_ids)
            self.tau_pos_response_mask = torch.ones_like(self.tau_pos_ids)
            self.tau_neg_response_mask = torch.ones_like(self.tau_neg_ids)
            self.h_T = torch.randn(2, 3584)
            self.rewards = torch.tensor([[0.8], [0.2]])
            self.best_idx = torch.tensor([0, 1])
            self.rollout_texts = ["plan: do x", "answer: yes"]

    buffer = MockBuffer()

    # 3. Construct save dictionary exactly matching the updated schema
    data_to_save = {
        # identifiers
        "global_step":           global_step,
        "micro_step":            micro_step,
        "sample_ids":            sample_ids,

        # student prompt
        "input_ids":             input_ids.cpu(),
        "attention_mask":        attention_mask.cpu(),
        "image_grid_thw":        image_grid_thw.cpu() if image_grid_thw is not None else None,
        "video_grid_thw":        video_grid_thw.cpu() if video_grid_thw is not None else None,

        # prefer paths, not processed pixels
        "pixel_values":          pixel_values.cpu() if pixel_values is not None else None,
        "pixel_values_videos":   pixel_values_videos.cpu() if pixel_values_videos is not None else None,

        # supervision
        "gt_waypoints":          gt_waypoints.cpu(),
        "ground_truth":          ground_truth,

        # teacher preference targets
        "tau_pos_ids":           buffer.tau_pos_ids.cpu(),
        "tau_pos_mask":          buffer.tau_pos_mask.cpu(),
        "tau_neg_ids":           buffer.tau_neg_ids.cpu(),
        "tau_neg_mask":          buffer.tau_neg_mask.cpu(),
        "tau_pos_response_mask": buffer.tau_pos_response_mask.cpu(),
        "tau_neg_response_mask": buffer.tau_neg_response_mask.cpu(),

        # distillation target
        "h_T":                   buffer.h_T.cpu().to(torch.bfloat16) if buffer.h_T is not None else torch.zeros(input_ids.shape[0], 3584, dtype=torch.bfloat16),

        # metadata/debugging
        "rewards":               buffer.rewards.cpu(),
        "best_idx":              buffer.best_idx.cpu(),
        "rollout_texts":         buffer.rollout_texts,
    }

    # Save to disk
    file_path = os.path.join(test_dir, "step_000042_micro_01.pt")
    torch.save(data_to_save, file_path)
    print(f"Saved serialization test file to: {file_path}")

    # 4. Load from disk and verify
    data_loaded = torch.load(file_path, map_location="cpu")
    
    # Assertions
    assert data_loaded["global_step"] == 42
    assert data_loaded["micro_step"] == 1
    assert data_loaded["sample_ids"] == sample_ids
    assert data_loaded["input_ids"].shape == (2, 64)
    assert data_loaded["pixel_values"].shape == (1, 28 * 28, 64)
    assert data_loaded["pixel_values_videos"] is None
    assert data_loaded["h_T"].dtype == torch.bfloat16
    assert data_loaded["h_T"].shape == (2, 3584)
    
    # Cleanup
    shutil.rmtree(test_dir)
    print("Serialization & deserialization unit test passed successfully!")

if __name__ == "__main__":
    run_test()
