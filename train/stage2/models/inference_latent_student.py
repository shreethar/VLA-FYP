"""
latent_student.py  [Qwen3.5-4B update]
---------------------------------------
Latent Student (Fθ) — ThinkFlow-VLA Stage 2.
Backbone: Qwen/Qwen3.5-4B

Key changes vs Qwen2.5-VL version:
  hidden_dim : 2048 → 2560
  num_layers : 28   → 32      (mid_layer 14 → 16)
  model class: Qwen2_5_VLForConditionalGeneration → AutoModelForCausalLM
  architecture: hybrid (GatedDeltaNet + GatedAttention, 3:1 per block)
    - KV cache still works via HF unified cache interface
    - No manual assumption about cache structure
  vision encoder: accessed via base.visual if present (early-fusion variant)
  <think> tokens: native in Qwen3.5, no special handling needed

Core mechanics (unchanged):
  Latent loop: z_m = last_hidden_state[:, 0, :] → fed as inputs_embeds (no vocab)
  Spatial tokens: K=5 learnable params → parallel forward → SpatialMLP → waypoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

# Qwen3.5-4B constants
QWEN35_4B_HIDDEN_DIM = 2560
QWEN35_4B_NUM_LAYERS = 32
QWEN35_0_8B_HIDDEN_DIM = 1024
QWEN35_0_8B_NUM_LAYERS = 24


class SpatialMLP(nn.Module):
    """[batch, K, d] → [batch, K, 2] in [0,1]"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


class LatentStudent(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-4B",
        M: int = 6,
        K: int = 5,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.M = M
        self.K = K

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        if hasattr(base, "visual"):
            for p in base.visual.parameters():
                p.requires_grad = False

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            bias="none",
        )
        self.vlm = get_peft_model(base, lora_cfg)

        self.hidden_dim: int   = getattr(self.vlm.config, "hidden_size",       QWEN35_4B_HIDDEN_DIM)
        self.num_layers: int   = getattr(self.vlm.config, "num_hidden_layers",  QWEN35_4B_NUM_LAYERS)
        self.mid_layer_idx: int = self.num_layers // 2

        self.spatial_tokens = nn.Parameter(torch.randn(K, self.hidden_dim) * 0.02)
        self.spatial_mlp    = SpatialMLP(self.hidden_dim)
        self._image_token_id = getattr(self.vlm.config, "image_token_id", None)

    @property
    def _base_transformer(self):
        return self.vlm.model

    @property
    def _embed_tokens(self):
        return self._base_transformer.embed_tokens

    def _build_input_embeds(self, input_ids, pixel_values, image_grid_thw):
        embeds = self._embed_tokens(input_ids)
        if pixel_values is not None and hasattr(self.vlm, "visual"):
            with torch.no_grad():
                img_feats = (self.vlm.visual(pixel_values, grid_thw=image_grid_thw)
                             if image_grid_thw is not None
                             else self.vlm.visual(pixel_values))
            if self._image_token_id is not None:
                mask = (input_ids == self._image_token_id)
                if mask.any():
                    embeds = embeds.clone()
                    embeds[mask] = img_feats.to(embeds.dtype)
        return embeds

    def encode_prefix(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        """Run prompt through transformer, return (last_hidden, past_key_values)."""
        embeds = self._build_input_embeds(input_ids, pixel_values, image_grid_thw)
        out = self._base_transformer(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        return out.last_hidden_state[:, -1, :], out.past_key_values

    def generate_latents(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        """
        M=6 autoregressive latent steps + K=5 parallel spatial tokens.
        Returns (latents: List[M×[batch,d]], spatial_hidden: [batch,K,d], waypoints: [batch,K,2])
        """
        B      = input_ids.shape[0]
        device = input_ids.device

        seed, past_kv = self.encode_prefix(input_ids, pixel_values, image_grid_thw, attention_mask)
        cur_attn  = attention_mask
        cur_embed = seed.unsqueeze(1)   # [B,1,d]
        latents   = []

        for _ in range(self.M):
            cur_attn = torch.cat([cur_attn,
                                  torch.ones(B,1,device=device,dtype=cur_attn.dtype)], dim=1)
            out = self._base_transformer(
                inputs_embeds=cur_embed,
                attention_mask=cur_attn,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            z_m     = out.last_hidden_state[:, 0, :]
            past_kv = out.past_key_values
            latents.append(z_m)
            cur_embed = z_m.unsqueeze(1)

        # Spatial tokens
        sp_embeds = self.spatial_tokens.unsqueeze(0).expand(B,-1,-1).to(cur_embed.dtype)
        cur_attn  = torch.cat([cur_attn,
                                torch.ones(B,self.K,device=device,dtype=cur_attn.dtype)], dim=1)
        sp_out = self._base_transformer(
            inputs_embeds=sp_embeds,
            attention_mask=cur_attn,
            past_key_values=past_kv,
            use_cache=False,
            return_dict=True,
        )
        spatial_hidden = sp_out.last_hidden_state       # [B,K,d]
        waypoints      = self.spatial_mlp(spatial_hidden) # [B,K,2]
        return latents, spatial_hidden, waypoints

    def get_answer_hidden_state(self, input_ids, pixel_values, image_grid_thw,
                                attention_mask, answer_token_positions):
        """h_S at <ans> position for L_distill. Returns [batch, d]."""
        embeds = self._build_input_embeds(input_ids, pixel_values, image_grid_thw)
        out = self._base_transformer(
            inputs_embeds=embeds, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        )
        h = out.last_hidden_state
        return h[torch.arange(h.shape[0], device=h.device), answer_token_positions]

    def get_mid_layer_visual_features(self, input_ids, pixel_values,
                                      image_grid_thw, attention_mask):
        """
        Hidden states at layer L/2=16 at visual token positions.
        Returns [batch, num_visual_tokens, d] or [batch, 1, d] if no visual tokens.
        """
        embeds = self._build_input_embeds(input_ids, pixel_values, image_grid_thw)
        out = self._base_transformer(
            inputs_embeds=embeds, attention_mask=attention_mask,
            use_cache=False, output_hidden_states=True, return_dict=True,
        )
        mid = out.hidden_states[self.mid_layer_idx + 1]   # [B, seq, d]

        if self._image_token_id is None:
            return mid

        mask = (input_ids == self._image_token_id)
        n    = mask[0].sum().item()
        if n == 0:
            return torch.zeros(input_ids.shape[0], 1, self.hidden_dim,
                               device=mid.device, dtype=mid.dtype)
        return mid[mask].view(input_ids.shape[0], n, self.hidden_dim)

    def forward(self, input_ids=None, pixel_values=None, image_grid_thw=None,
                attention_mask=None, labels=None):
        return self.vlm(input_ids=input_ids, pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw, attention_mask=attention_mask,
                        labels=labels)

    def print_trainable_parameters(self):
        self.vlm.print_trainable_parameters()
        print(f"  hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, mid_layer={self.mid_layer_idx}")