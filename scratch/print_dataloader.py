from transformers import AutoProcessor
import sys
sys.path.insert(0, 'train/stage2')
from stage2_dataloader import build_stage2_dataloader

print("Loading processor...")
processor = AutoProcessor.from_pretrained("unsloth/Qwen3.5-4B", trust_remote_code=True)
print("Building dataloader...")
loader = build_stage2_dataloader(
    hf_repo="shreethar/FYP-Stage2-dataset",
    processor=processor,
    split="test",
    batch_size=1,
    num_workers=1,
    max_length=2048
)

print("Getting a batch...")
batch = next(iter(loader))

print("\n--- BATCH OUTPUT ---")
for key, value in batch.items():
    if hasattr(value, 'shape'):
        print(f"{key}: tensor of shape {value.shape} and dtype {value.dtype}")
    elif isinstance(value, dict):
        print(f"{key}: dict with keys {list(value.keys())}")
        for k, v in value.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: tensor of shape {v.shape}")
            else:
                print(f"  {k}: {type(v)} = {str(v)[:200]}")
    elif isinstance(value, list):
        print(f"{key}: list of length {len(value)}")
        print(f"  First element: {type(value[0])} = {str(value[0])[:200]}")
    else:
        print(f"{key}: {type(value)} = {value}")

print("\n--- DECODED PROMPT ---")
print(processor.decode(batch["input_ids"][0], skip_special_tokens=False))
