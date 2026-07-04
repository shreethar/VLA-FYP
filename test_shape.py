import torch
d = torch.load('/home/ubuntu/VLA-FYP/train/stage2/checkpoints/stage2_decoupled_mini/offline_data/step_000000_micro_00.pt', map_location='cpu')
print('rewards shape:', d['rewards'].shape)
print('rewards dtype:', d['rewards'].dtype)
print('input_ids shape:', d['input_ids'].shape)
