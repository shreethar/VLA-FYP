"""
verbalizer.py
-------------
Verbalizer (Vψ) for ThinkFlow-VLA Stage 2.

Base model: Qwen3.5-0.8B (hybrid Gated DeltaNet + Attention architecture)

Architecture
------------
At EVERY transformer layer, a new CrossAttentionBlock is inserted:
    h_l  = OriginalTransformerLayer_l(h_{l-1})          ← untouched SA/DeltaNet + FFN
    h_l  = CrossAttentionBlock_l(Q=h_l, K=z, V=z)      ← new CA reads Student latents

z = stack of Student's M=6 latent vectors, shape [batch, M, d_student].
Since d_student (2560) ≠ d_verbalizer (1024), each CA block has its own K/V projection.

Qwen3.5-0.8B architecture:
  - 24 transformer layers (3:1 Gated DeltaNet : standard Attention)
  - Hidden dim: 1024
  - Vocab: 248,320

Model hierarchy (after PEFT wrapping):
  self.lm                              → PeftModel
  self.lm.model                        → Qwen3_5ForCausalLM
  self.lm.model.model                  → Qwen3_5TextModel (.embed_tokens, .layers, .norm)
  self.lm.model.lm_head                → Linear

CA injection uses forward hooks on each decoder layer rather than manual
layer-by-layer forward.  This avoids replicating the complex mask/position
logic (4D position_ids, separate masks for linear_attention vs full_attention
layers, rotary embeddings) that Qwen3.5's TextModel.forward() handles internally.

Training schedule (controlled externally by train_stage2.py):
  Steps 0 – 3000  : warm-up  — CA blocks + LoRA trainable, LM loss on τ+
  Steps 3000 – 4500: frozen  — all Vψ params frozen, DPO gradient flows into Student

Loss functions implemented here:
  compute_lm_loss    : cross-entropy on τ+ tokens     (warm-up phase)
  compute_dpo_loss   : DPO preference loss on τ+/τ−   (both phases, but only updates
                       Vψ params during warm-up; Student params always)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# Cross-Attention Block (one per Verbalizer transformer layer)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    A single cross-attention block that reads M Student latent vectors.

    Q  = hidden states from the current Verbalizer layer  [batch, seq, d_verb]
    K,V = Student latents z projected to d_verb            [batch, M,   d_verb]

    Output replaces the query sequence via residual + pre-norm:
        h = LayerNorm(h + MultiheadAttn(Q=h, K=k, V=v))
    """

    def __init__(
        self,
        query_dim: int,   # Verbalizer hidden size  (d_verb = 1024)
        kv_dim: int,      # Student hidden size     (d_student = 2560)
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert query_dim % num_heads == 0, (
            f"query_dim {query_dim} must be divisible by num_heads {num_heads}"
        )

        # Project Student latents into Verbalizer's space for K and V
        self.k_proj = nn.Linear(kv_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, query_dim, bias=False)

        # Standard multi-head cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pre-norm on queries (applied before attention, following modern practice)
        self.q_norm = nn.LayerNorm(query_dim, eps=1e-6)

        # Post-norm on residual output
        self.out_norm = nn.LayerNorm(query_dim, eps=1e-6)

    def forward(
        self,
        hidden: torch.Tensor,    # [batch, seq, d_verb]
        latents: torch.Tensor,   # [batch, M, d_student]
    ) -> torch.Tensor:
        # Project latents → K and V in verbalizer space
        k = self.k_proj(latents)  # [batch, M, d_verb]
        v = self.v_proj(latents)  # [batch, M, d_verb]

        # Normalize queries before attention
        q = self.q_norm(hidden)   # [batch, seq, d_verb]

        # Cross-attention: every verbalizer token attends to all M latents
        ca_out, _ = self.attn(q, k, v)  # [batch, seq, d_verb]

        # Residual connection + post-norm
        return self.out_norm(hidden + ca_out)  # [batch, seq, d_verb]


# ---------------------------------------------------------------------------
# Verbalizer
# ---------------------------------------------------------------------------

class Verbalizer(nn.Module):
    """
    Qwen3.5-0.8B with per-layer cross-attention blocks conditioned on Student latents.

    CA injection uses forward hooks on each decoder layer.  This avoids
    replicating the complex internal logic of Qwen3.5's TextModel.forward()
    (4D position_ids, create_causal_mask, separate masks for linear_attention
    vs full_attention layers, rotary embeddings, etc.).

    Hook mechanism:
        After each decoder_layer returns hidden_states, a registered
        post-forward hook applies the corresponding CrossAttentionBlock.
        The hooks read latents from self._current_latents (set before each
        forward call).

    Parameters
    ----------
    model_name     : HuggingFace repo ID for the 0.8B base model
    student_hidden : hidden size of the Student VLM (d_student = 2560)
    lora_rank      : LoRA rank for the base attention layers
    lora_alpha     : LoRA scaling
    ca_num_heads   : attention heads inside each CrossAttentionBlock
    ca_dropout     : dropout in cross-attention (0 during distillation)
    dpo_beta       : β temperature for DPO loss
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        student_hidden: int = 2560,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        ca_num_heads: int = 8,
        ca_dropout: float = 0.0,
        dpo_beta: float = 0.1,
    ):
        super().__init__()
        self.dpo_beta = dpo_beta

        # ------------------------------------------------------------------
        # 1. Load Qwen3.5-0.8B
        # ------------------------------------------------------------------
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        # ------------------------------------------------------------------
        # 2. Wrap with LoRA
        # ------------------------------------------------------------------
        # Target modules span both layer types in the hybrid architecture:
        #   - Standard Attention: q_proj, k_proj, v_proj, o_proj
        #   - Gated DeltaNet:     out_proj, in_proj_qkv, in_proj_z
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                # Standard attention projections
                "q_proj", "k_proj", "v_proj", "o_proj",
                # Gated DeltaNet main projections
                "out_proj", "in_proj_qkv", "in_proj_z",
            ],
            bias="none",
        )
        self.lm = get_peft_model(base, lora_cfg)

        # Explicitly enable gradient checkpointing for the Verbalizer
        # The enable_input_require_grads() is MANDATORY because PEFT freezes the embedding
        # layer, which otherwise causes PyTorch to silently skip checkpointing entirely!
        self.lm.enable_input_require_grads()
        self.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.lm.config.use_cache = False

        # Infer verbalizer hidden dim and layer count from config
        self.hidden_dim: int = self.lm.config.hidden_size        # 1024 for 0.8B
        self.num_layers: int = self.lm.config.num_hidden_layers  # 24 for 0.8B

        # ------------------------------------------------------------------
        # 3. Insert one CrossAttentionBlock per transformer layer
        # ------------------------------------------------------------------
        self.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(
                query_dim=self.hidden_dim,
                kv_dim=student_hidden,
                num_heads=ca_num_heads,
                dropout=ca_dropout,
            )
            for _ in range(self.num_layers)
        ]).to(base.dtype)

        # ------------------------------------------------------------------
        # 4. Register forward hooks on each decoder layer
        # ------------------------------------------------------------------
        # _current_latents is set before each forward call and read by hooks
        self._current_latents: Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_ca_hooks()

        # Freeze tracking
        self._frozen: bool = False

    def _register_ca_hooks(self):
        """
        Register a post-forward hook on each decoder layer that applies
        the corresponding CrossAttentionBlock.

        The hook reads latents from self._current_latents, which must be
        set before calling the model's forward.
        """
        transformer = self._transformer

        for layer_idx, layer in enumerate(transformer.layers):
            ca_block = self.ca_blocks[layer_idx]

            def make_hook(ca_blk, idx):
                def hook_fn(module, args, output):
                    # Decoder layers return hidden_states directly (not a tuple)
                    # in newer transformers versions
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    if self._current_latents is not None:
                        hidden = ca_blk(hidden, self._current_latents)

                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    else:
                        return hidden
                return hook_fn

            handle = layer.register_forward_hook(make_hook(ca_block, layer_idx))
            self._hooks.append(handle)

    # -----------------------------------------------------------------------
    # Internal: access helpers for the Qwen3.5 model hierarchy
    # -----------------------------------------------------------------------

    @property
    def _transformer(self):
        """
        Returns the Qwen3_5TextModel (the actual transformer stack).

        Hierarchy after PEFT wrapping:
          self.lm (PeftModel) → .model (Qwen3_5ForCausalLM)
          → .model (Qwen3_5TextModel: .embed_tokens, .layers, .norm)
        """
        return self.lm.model.model

    @property
    def _lm_head(self):
        """Returns the LM head (Linear layer for logit projection)."""
        return self.lm.model.lm_head

    # -----------------------------------------------------------------------
    # Forward with latent injection
    # -----------------------------------------------------------------------

    def _forward_with_latents(
        self,
        input_ids: torch.Tensor,        # [batch, seq]
        attention_mask: torch.Tensor,    # [batch, seq]
        latents: torch.Tensor,           # [batch, M, d_student]  — stacked z_1…z_M
        labels: Optional[torch.Tensor] = None,  # [batch, seq] for LM loss
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with CA injection via hooks.

        The hooks read from self._current_latents which is set here before
        calling the model and cleared after.  This lets the model's own
        forward() handle all the complex mask/position/rotary logic while
        we inject CA at each layer boundary.

        Returns
        -------
        logits : [batch, seq, vocab_size]
        loss   : scalar CE loss if labels provided, else None
        """
        # Set latents for hooks to read
        self._current_latents = latents

        try:
            # Use the model's native forward — this handles:
            #   - 4D position_ids computation
            #   - create_causal_mask for full_attention layers
            #   - _update_linear_attn_mask for linear_attention layers
            #   - rotary embeddings
            #   - layer iteration with correct mask selection
            # The registered hooks inject CA after each layer
            outputs = self._transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
        finally:
            # Clear latents reference to avoid holding memory
            self._current_latents = None

        hidden = outputs.last_hidden_state  # [batch, seq, d_verb]
        logits = self._lm_head(hidden)      # [batch, seq, vocab_size]

        # --- Optional LM loss ---
        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def _compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,       # [batch, seq]
        attention_mask: torch.Tensor,   # [batch, seq]
        latents: torch.Tensor,          # [batch, M, d_student]
        response_mask: torch.Tensor,    # [batch, seq] — 1 on response tokens only
    ) -> torch.Tensor:
        """
        Compute per-sequence sum of log-probabilities over response tokens.

        Used by compute_dpo_loss to get log π_ψ(τ | z).

        Returns
        -------
        log_probs : [batch]  — sum of token log-probs over response positions
        """
        logits, _ = self._forward_with_latents(input_ids, attention_mask, latents)

        # Memory-efficient manual log-probs (avoids allocating [batch, seq, vocab] tensor)
        shift_logits = logits[:, :-1, :]                    # [batch, seq-1, vocab]
        shift_labels    = input_ids[:, 1:]                  # [batch, seq-1]
        shift_mask      = response_mask[:, 1:]              # [batch, seq-1]

        # Gather the logit of each ground-truth token
        target_logits = shift_logits.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)  # [batch, seq-1]
        
        token_log_probs = target_logits - torch.logsumexp(shift_logits, dim=-1) # [batch, seq-1]

        # Sum over response positions only
        seq_log_probs = (token_log_probs * shift_mask).sum(dim=-1)  # [batch]

        return seq_log_probs

    # -----------------------------------------------------------------------
    # Public loss functions
    # -----------------------------------------------------------------------

    def compute_lm_loss(
        self,
        input_ids: torch.Tensor,       # [batch, seq]  — τ+ sequence
        attention_mask: torch.Tensor,
        latents: torch.Tensor,         # [batch, M, d_student]  — Student z (DETACHED during warm-up)
        labels: torch.Tensor,          # [batch, seq]  — τ+ with -100 on prefix positions
    ) -> torch.Tensor:
        """
        Warm-up loss: standard cross-entropy on τ+ tokens.
        Trains the CA blocks and LoRA to learn to read Student latents.

        NOTE: during warm-up, pass latents.detach() so gradients do NOT
        flow back into the Student yet. The Student is updated by L_distill
        and L_ans only during warm-up.

        Returns
        -------
        lm_loss : scalar
        """
        _, loss = self._forward_with_latents(
            input_ids, attention_mask, latents, labels=labels
        )
        return loss

    def compute_dpo_loss(
        self,
        pos_input_ids: torch.Tensor,   # [batch, seq] — τ+ tokenized
        neg_input_ids: torch.Tensor,    # [batch, seq] — τ− tokenized
        pos_attention_mask: torch.Tensor,
        neg_attention_mask: torch.Tensor,
        latents: torch.Tensor,          # [batch, M, d_student] — Student z (NO detach here)
        pos_response_mask: torch.Tensor,   # [batch, seq] — 1 on τ+ response tokens
        neg_response_mask: torch.Tensor,   # [batch, seq] — 1 on τ− response tokens
        ref_pos_log_probs: Optional[torch.Tensor] = None,  # [batch] reference model log-probs
        ref_neg_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        DPO preference loss.

        After Verbalizer is frozen (step > 3000), all gradients from this loss
        flow through the latents tensor back into the Student's parameters.

        DPO objective:
            L_DPO = -E[ log σ( β * (log π(τ+|z) − log π_ref(τ+|z))
                                 − β * (log π(τ−|z) − log π_ref(τ−|z)) ) ]

        If reference log-probs not provided (common simplification), reduces to:
            L_DPO = -E[ log σ( β * (log π(τ+|z) − log π(τ−|z)) ) ]

        Parameters
        ----------
        ref_pos_log_probs / ref_neg_log_probs:
            Pass pre-computed reference model log-probs if using a reference
            policy (e.g., initial Verbalizer checkpoint). Pass None to use
            the simplified reference-free variant.

        Returns
        -------
        dpo_loss : scalar
        metrics  : dict with reward margin and accuracy for logging
        """
        # Log-probs under current Verbalizer policy
        # latents NOT detached → gradient flows into Student when Vψ is frozen
        log_pi_pos = self._compute_sequence_log_probs(
            pos_input_ids, pos_attention_mask, latents, pos_response_mask
        )  # [batch]
        log_pi_neg = self._compute_sequence_log_probs(
            neg_input_ids, neg_attention_mask, latents, neg_response_mask
        )  # [batch]

        # DPO reward margins
        if ref_pos_log_probs is not None and ref_neg_log_probs is not None:
            # Full DPO: subtract reference log-probs
            pi_log_ratios_pos = log_pi_pos - ref_pos_log_probs
            pi_log_ratios_neg = log_pi_neg - ref_neg_log_probs
        else:
            # Simplified (reference-free) DPO
            pi_log_ratios_pos = log_pi_pos
            pi_log_ratios_neg = log_pi_neg

        reward_margin = self.dpo_beta * (pi_log_ratios_pos - pi_log_ratios_neg)

        # DPO loss: -log sigmoid(reward_margin)
        dpo_loss = -F.logsigmoid(reward_margin).mean()

        # Logging metrics
        with torch.no_grad():
            metrics = {
                "dpo_loss":       dpo_loss.item(),
                "reward_margin":  reward_margin.mean().item(),
                "dpo_accuracy":   (reward_margin > 0).float().mean().item(),
                "log_pi_pos":     log_pi_pos.mean().item(),
                "log_pi_neg":     log_pi_neg.mean().item(),
            }

        return dpo_loss, metrics

    # -----------------------------------------------------------------------
    # Warm-up → freeze transition
    # -----------------------------------------------------------------------

    def freeze_for_student_training(self):
        """
        Called at step 3000.

        Freezes ALL Verbalizer parameters (base model + LoRA + CA blocks).
        After this, DPO gradients flow through the latents tensor only,
        updating the Student's weights — not the Verbalizer's.
        """
        if self._frozen:
            return

        for param in self.parameters():
            param.requires_grad = False

        self._frozen = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Verbalizer] Frozen. Trainable params remaining: {trainable:,}")

    def unfreeze_ca_and_lora(self):
        """
        Re-enables gradient flow through CA blocks and LoRA layers.
        Mainly useful for warm-up resumption after a checkpoint reload.
        """
        # CA blocks are always fully trainable
        for param in self.ca_blocks.parameters():
            param.requires_grad = True

        # LoRA layers: only the lora_A/lora_B matrices, not the frozen base weights
        for name, param in self.lm.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        self._frozen = False

    def is_frozen(self) -> bool:
        return self._frozen

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def print_trainable_parameters(self):
        self.lm.print_trainable_parameters()
        ca_params = sum(p.numel() for p in self.ca_blocks.parameters())
        print(f"  ca_blocks (all layers): {ca_params:,} params")

    @staticmethod
    def stack_latents(latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Converts the Student's output (List of M tensors [batch, d]) into
        the [batch, M, d] tensor the Verbalizer expects.

        Call this before passing latents into any Verbalizer method.
        """
        return torch.stack(latents, dim=1)  # [batch, M, d]