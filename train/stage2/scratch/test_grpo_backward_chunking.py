import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import sys
import os

# Add parent directory to sys.path so we can import training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.grpo_teacher import GRPOTeacher

class DummyGRPOTeacher(GRPOTeacher):
    def __init__(self, backward_batch_size: int = 1):
        # Override GRPOTeacher.__init__ to avoid loading real weights
        nn.Module.__init__(self)
        self.G = 2
        self.kl_coef = 0.05
        self.offload_ref_model = False
        self.backward_batch_size = backward_batch_size
        
        class TinyVLM(nn.Module):
            def __init__(self):
                super().__init__()
                class Config:
                    image_token_id = 100
                    video_token_id = 200
                self.config = Config()
                
                self.embed = nn.Embedding(1000, 16)
                
                class Visual(nn.Module):
                    def forward(self, pv, grid_thw=None):
                        return torch.zeros(pv.shape[0], 16, device=pv.device)
                
                class ActualModel(nn.Module):
                    def __init__(self, embed):
                        super().__init__()
                        self.language_model = nn.Module()
                        self.language_model.embed_tokens = embed
                        self.visual = Visual()
                
                class Wrapper(nn.Module):
                    def __init__(self, embed):
                        super().__init__()
                        self.model = ActualModel(embed)
                
                self.model = Wrapper(self.embed)
                self.lm_head = nn.Linear(16, 1000)
                
            def forward(self, inputs_embeds, attention_mask, position_ids, use_cache=False, return_dict=True):
                logits = self.lm_head(inputs_embeds)
                class Output:
                    def __init__(self, logits):
                        self.logits = logits
                return Output(logits)
                
        self.vlm = TinyVLM()
        self._ref_model = TinyVLM()

def run_test():
    print("Initializing Dummy Teacher models...")
    teacher_chunk1 = DummyGRPOTeacher(backward_batch_size=1)
    teacher_chunk4 = DummyGRPOTeacher(backward_batch_size=4)
    
    # Synchronize weights so they are identical
    teacher_chunk4.vlm.load_state_dict(teacher_chunk1.vlm.state_dict())
    teacher_chunk4._ref_model.load_state_dict(teacher_chunk1._ref_model.state_dict())
    
    # 1. Setup mock data
    # G = 2, B = 4, prompt_len = 10, response_len = 20, seq = 30
    G = 2
    B = 4
    prompt_len = 10
    seq_len = 30
    
    # inputs_ids: [B, seq_len]
    # Rollout 0 and 1
    torch.manual_seed(42)
    all_ids = [
        torch.randint(10, 90, (B, seq_len)),
        torch.randint(10, 90, (B, seq_len))
    ]
    # Inject correct number of image (3) and video (5) tokens into prompt section
    # The prompt section (first prompt_len tokens) must be identical across all G groups
    for g in range(G):
        all_ids[g][0, 3:6] = 100
        all_ids[g][2, 5:10] = 200
    
    all_masks = [
        torch.ones(B, seq_len, dtype=torch.long),
        torch.ones(B, seq_len, dtype=torch.long)
    ]
    advantages = torch.randn(G, B)
    
    # Image values (1 item in batch has image for each rollout, 3 patches total)
    pixel_values = torch.randn(3, 3, 28, 28)
    image_grid_thw = torch.tensor([[1, 1, 3]])
    
    # Video values (1 item in batch has video, 5 patches total)
    pixel_values_videos = torch.randn(5, 3, 28, 28)
    video_grid_thw = torch.tensor([[1, 1, 5]])
    
    # 2. Test forward + backward pass for chunk_size=1
    print("\n--- Running with backward_batch_size = 1 ---")
    loss1, kl1, raw_kl1 = teacher_chunk1.compute_grpo_loss(
        all_ids=all_ids,
        all_masks=all_masks,
        advantages=advantages,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        prompt_len=prompt_len,
        grad_accum_steps=1,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw
    )
    print(f"loss1: {loss1.item():.6f}, kl1: {kl1.item():.6f}, raw_kl1: {raw_kl1.item():.6f}")
    
    # Gradients are already computed during compute_grpo_loss internal loops
    grads1 = {name: param.grad.clone() for name, param in teacher_chunk1.vlm.named_parameters() if param.grad is not None}
    
    # 3. Test forward + backward pass for chunk_size=4
    print("\n--- Running with backward_batch_size = 4 (Full batch) ---")
    loss4, kl4, raw_kl4 = teacher_chunk4.compute_grpo_loss(
        all_ids=all_ids,
        all_masks=all_masks,
        advantages=advantages,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        prompt_len=prompt_len,
        grad_accum_steps=1,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw
    )
    print(f"loss4: {loss4.item():.6f}, kl4: {kl4.item():.6f}, raw_kl4: {raw_kl4.item():.6f}")
    
    grads4 = {name: param.grad.clone() for name, param in teacher_chunk4.vlm.named_parameters() if param.grad is not None}
    
    # 4. Compare outputs and gradients
    print("\n--- Verifying Equivalence ---")
    # Losses should be identical
    assert torch.allclose(loss1, loss4, atol=1e-5), f"Loss mismatch: {loss1} vs {loss4}"
    assert torch.allclose(kl1, kl4, atol=1e-5), f"KL loss mismatch: {kl1} vs {kl4}"
    assert abs(raw_kl1 - raw_kl4) < 1e-5, f"Raw KL mismatch: {raw_kl1} vs {raw_kl4}"
    print("✓ Losses and KL values match perfectly!")
    
    # Gradients should be identical
    for name in grads1:
        assert torch.allclose(grads1[name], grads4[name], atol=1e-5), f"Gradient mismatch for {name}"
    print("✓ Gradients match perfectly!")
    print("\nAll GRPO backward chunking tests passed successfully!")

if __name__ == "__main__":
    run_test()
