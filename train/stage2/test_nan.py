import torch
from safetensors.torch import load_file
import os

ckpt_dir = "/home/ubuntu/VLA-FYP/train/stage2/checkpoints/stage2_decoupled_mini"
state_file = os.path.join(ckpt_dir, "training_state.pt")
if os.path.exists(state_file):
    state = torch.load(state_file, map_location="cpu")
    print(f"Found training state at step {state['step']}")
else:
    print("No training state found.")

