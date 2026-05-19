"""
verbalizer.py  [Qwen3.5-0.8B update]
--------------------------------------
Verbalizer (Vψ) — ThinkFlow-VLA Stage 2.
Backbone: Qwen/Qwen3.5-0.8B

Key changes vs Qwen3-0.6B version:
  model     : Qwen/Qwen3-0.6B → Qwen/Qwen3.5-0.8B
  hidden_dim: ~1024 (0.6B) → 1024 (0.8B — same, no change to CA block dims)
  num_layers: ~28 → 24
  architecture: hybrid (GatedDeltaNet + GatedAttention, 3:1 per block, 6 blocks)

CRITICAL CHANGE — forward pass approach:
  The old version used manual layer-by-layer iteration, which breaks on
  GatedDeltaNet layers (different interface from standard attention layers).
  
  NEW APPROACH: PyTorch forward hooks.
  A hook is registered on each transformer layer AFTER its forward call.
  The hook injects CrossAttentionBlock output into the hidden states.
  This is architecture-agnostic — works on any layer type.

Hook pattern:
    def hook(module, input, output):
        h = output[0]              # hidden states (always first element)
        h = ca_block(h, latents)   # inject CA
        return (h,) + output[1:]   # return modified output

The hooks are registered before each forward call and removed after,
so they carry no persistent state and don't affect inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

QWEN35_0_8B_HIDDEN_DIM = 1024
QWEN35_0_8B_NUM_LAYERS = 24


# ---------------------------------------------------------------------------
# CrossAttentionBlock — one per Verbalizer layer
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Q = Verbalizer hidden states  [batch, seq, d_verb]
    K,V = Student latents projected to d_verb  [batch, M, d_verb]
    Output: h + LayerNorm(MHA(Q,K,V))
    """
    def __init__(self, query_dim: int, kv_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert query_dim % num_heads == 0
        self.k_proj   = nn.Linear(kv_dim, query_dim, bias=False)
        self.v_proj   = nn.Linear(kv_dim, query_dim, bias=False)
        self.attn     = nn.MultiheadAttention(query_dim, num_heads, dropout=dropout, batch_first=True)
        self.q_norm   = nn.LayerNorm(query_dim, eps=1e-6)
        self.out_norm = nn.LayerNorm(query_dim, eps=1e-6)

    def forward(self, hidden: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        k   = self.k_proj(latents)
        v   = self.v_proj(latents)
        q   = self.q_norm(hidden)
        out, _ = self.attn(q, k, v)
        return self.out_norm(hidden + out)


# ---------------------------------------------------------------------------
# Verbalizer
# ---------------------------------------------------------------------------

class Verbalizer(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        student_hidden: int = 2560,      # Qwen3.5-4B hidden size
        lora_rank: int = 32,
        lora_alpha: int = 64,
        ca_num_heads: int = 8,
        ca_dropout: float = 0.0,
        dpo_beta: float = 0.1,
    ):
        super().__init__()
        self.dpo_beta = dpo_beta

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            bias="none",
        )
        self.lm = get_peft_model(base, lora_cfg)

        self.hidden_dim: int = getattr(self.lm.config, "hidden_size",      QWEN35_0_8B_HIDDEN_DIM)
        self.num_layers: int = getattr(self.lm.config, "num_hidden_layers", QWEN35_0_8B_NUM_LAYERS)

        self.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(self.hidden_dim, student_hidden, ca_num_heads, ca_dropout)
            for _ in range(self.num_layers)
        ])
        self._frozen = False

    # -----------------------------------------------------------------------
    # Forward hook injection
    # -----------------------------------------------------------------------

    def _register_ca_hooks(self, latents: torch.Tensor) -> List:
        """
        Register one forward hook per transformer layer.
        Each hook injects CA output after the layer's own computation.
        Hooks are stored and returned so the caller can remove them after use.
        """
        hooks = []
        layers = self.lm.model.layers   # works for all Qwen variants

        for layer_idx, layer in enumerate(layers):
            ca_block = self.ca_blocks[layer_idx]

            def make_hook(block, z):
                def hook(module, inp, output):
                    # output is a tuple; first element is always hidden states
                    h = output[0]
                    h = block(h, z)
                    return (h,) + output[1:]
                return hook

            h = layer.register_forward_hook(make_hook(ca_block, latents))
            hooks.append(h)

        return hooks

    @staticmethod
    def _remove_hooks(hooks: List):
        for h in hooks:
            h.remove()

    # -----------------------------------------------------------------------
    # Sequence log-probs helper
    # -----------------------------------------------------------------------

    def _compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        log π_ψ(response | z) via hook-injected forward pass.
        Returns [batch] summed log-probs over response tokens.
        """
        hooks = self._register_ca_hooks(latents)
        try:
            out = self.lm(input_ids=input_ids, attention_mask=attention_mask,
                          return_dict=True)
        finally:
            self._remove_hooks(hooks)

        log_probs = F.log_softmax(out.logits, dim=-1)          # [B, seq, vocab]
        shift_lp  = log_probs[:, :-1, :]
        shift_ids = input_ids[:, 1:]
        shift_mask = response_mask[:, 1:]
        token_lp  = shift_lp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
        return (token_lp * shift_mask).sum(dim=-1)             # [batch]

    # -----------------------------------------------------------------------
    # Public loss functions
    # -----------------------------------------------------------------------

    def compute_lm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,      # pass z.detach() during warm-up
        labels: torch.Tensor,
    ) -> torch.Tensor:
        hooks = self._register_ca_hooks(latents)
        try:
            out = self.lm(input_ids=input_ids, attention_mask=attention_mask,
                          labels=labels, return_dict=True)
        finally:
            self._remove_hooks(hooks)
        return out.loss

    def compute_dpo_loss(
        self,
        pos_input_ids: torch.Tensor,
        neg_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        neg_attention_mask: torch.Tensor,
        latents: torch.Tensor,             # NOT detached in frozen phase
        pos_response_mask: torch.Tensor,
        neg_response_mask: torch.Tensor,
        ref_pos_log_probs: Optional[torch.Tensor] = None,
        ref_neg_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:

        log_pi_pos = self._compute_sequence_log_probs(
            pos_input_ids, pos_attention_mask, latents, pos_response_mask
        )
        log_pi_neg = self._compute_sequence_log_probs(
            neg_input_ids, neg_attention_mask, latents, neg_response_mask
        )

        if ref_pos_log_probs is not None and ref_neg_log_probs is not None:
            margin = self.dpo_beta * ((log_pi_pos - ref_pos_log_probs)
                                      - (log_pi_neg - ref_neg_log_probs))
        else:
            margin = self.dpo_beta * (log_pi_pos - log_pi_neg)

        loss = -F.logsigmoid(margin).mean()

        with torch.no_grad():
            metrics = {
                "dpo_loss":      loss.item(),
                "reward_margin": margin.mean().item(),
                "dpo_accuracy":  (margin > 0).float().mean().item(),
                "log_pi_pos":    log_pi_pos.mean().item(),
                "log_pi_neg":    log_pi_neg.mean().item(),
            }

        return loss, metrics

    # -----------------------------------------------------------------------
    # Freeze / unfreeze
    # -----------------------------------------------------------------------

    def freeze_for_student_training(self):
        if self._frozen:
            return
        for p in self.parameters():
            p.requires_grad = False
        self._frozen = True
        print(f"[Verbalizer] Frozen. Trainable params remaining: "
              f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def unfreeze_ca_and_lora(self):
        for p in self.ca_blocks.parameters():
            p.requires_grad = True
        for n, p in self.lm.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        self._frozen = False

    def is_frozen(self) -> bool:
        return self._frozen

    def print_trainable_parameters(self):
        self.lm.print_trainable_parameters()
        ca = sum(p.numel() for p in self.ca_blocks.parameters())
        print(f"  ca_blocks: {ca:,}  hidden_dim={self.hidden_dim}  num_layers={self.num_layers}")

    @staticmethod
    def stack_latents(latents: List[torch.Tensor]) -> torch.Tensor:
        """List[M × [batch,d]] → [batch, M, d]"""
        return torch.stack(latents, dim=1)