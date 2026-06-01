"""
spatial_forcing.py
[VGGT update]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ProjectionMLP(nn.Module):
    def __init__(self, llm_dim: int, vggt_dim: int, align_loss_type: str = "cosine", use_vlm_norm: bool = False) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim
        self.align_loss_type = align_loss_type
        
        hidden_dim = (vggt_dim + llm_dim) // 2

        self.fc1 = nn.Linear(self.llm_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, self.vggt_dim, bias=True)
        self.act_fn1 = nn.GELU()

        self.vlm_norm = nn.LayerNorm(self.llm_dim, eps=1e-6) if use_vlm_norm else None
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def align_dimension(self, LLM_embedding: torch.Tensor) -> torch.Tensor:
        if self.vlm_norm is not None:
            LLM_embedding = self.vlm_norm(LLM_embedding)
        projected_features = self.fc1(LLM_embedding)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features

    def compute_align_loss_cosine(self, vision_hidden: torch.Tensor, VGGT_hidden: torch.Tensor):
        align_loss = 0.0
        bsz = vision_hidden.shape[0]

        for _vision, _VGGT in zip(vision_hidden, VGGT_hidden):
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _VGGT = torch.nn.functional.normalize(_VGGT, dim=-1)
            
            cosine_sim = (_vision * _VGGT).sum(dim=-1)
            align_loss += 1.0 - cosine_sim.mean()

        align_loss /= bsz
        return align_loss

    def forward(self, LLM_emb, target_emb):
        if self.align_loss_type == "cosine":
            # Project in bf16 to save VRAM
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                LLM_emb = self.align_dimension(LLM_emb)
                
            # CRITICAL: Cast to fp32 for cosine similarity to avoid underflow/NaNs in bf16
            align_loss = self.compute_align_loss_cosine(LLM_emb.float(), target_emb.float())
            return align_loss
        else:
            raise NotImplementedError(f"Align loss type {self.align_loss_type} is not implemented.")

class VGGTExtractor(nn.Module):

    def __init__(
        self,
        checkpoint: str = "model.pt",
        output_dim: int = 1024,
        device: torch.device = torch.device("cuda"),
        layer_idx: int = -1,
    ):
        super().__init__()
        self.device = device
        self.layer_idx = layer_idx
        try:
            from vggt.models.vggt import VGGT
        except ImportError:
            raise ImportError(
                "VGGT is not installed"
                "Then verify: python -c 'from vggt.models.vggt import VGGT'"
            )
            
        self.model = VGGT(
            enable_camera = False,
            enable_point = False,
            enable_depth = False,
            enable_track = False,
            feature_only = True,
        ).to(self.device)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def load_checkpoint(self, checkpoint: str):
        self.model.load_state_dict(torch.load(checkpoint), strict = False)
        print("VGGT checkpoint loaded successfully")
        

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vggt_output = self.model(images)
        aggt_vggt_hidden = vggt_output["features"][self.layer_idx]
        patch_start_idx = vggt_output["patch_start_idx"]
        original_image = vggt_output["images"]
        vggt_hidden = agg_vggt_hidden[:, :, patch_start_idx:, :]
                
        return vggt_hidden, original_image

    def pool(self, vggt_hidden: torch.Tensor, original_image: torch.Tensor, vision_hidden: torch.Tensor, interpolate_method: str = "bilinear", use_vggt_pe: bool = False) -> torch.Tensor:
        H, W = original_image.shape[-2:]
        patch_h, patch_w = H // self.model.patch_size, W // self.model.patch_size
        aligned_vggt_hidden = custom_pooling(
            vggt_hidden,
            (patch_h, patch_w),
            (H, W),
            vision_hidden, 
            interpolate_method, 
            use_vggt_pe
        )
        return aligned_vggt_hidden