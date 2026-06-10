"""
verbalizer.py
-------------
Verbalizer (Vψ) — ThinkFlow-VLA Stage 2.
Backbone: Qwen/Qwen3.5-0.8B  (AutoModelForImageTextToText)

Architecture
------------
Base: Qwen3.5-0.8B with LoRA rank=32 on all projection types.

At EVERY transformer layer, one CrossAttentionBlock is registered as a
persistent forward hook (registered ONCE in __init__, lives for the model
lifetime). The hook reads Student latents from self._current_latents, which
is set immediately before each forward call and cleared in a try/finally.

Hook pattern (persistent, not per-call):
    hook(module, input, output):
        h   = output[0]                     # hidden states — always first element
        h   = ca_block(h, self._current_latents)
        return (h,) + output[1:]            # return modified output tuple

Model hierarchy after get_peft_model wrapping:
    self.lm                                   # PeftModel
    self.lm.model                             # AutoModelForImageTextToText (base)
    self.lm.model.model                       # Qwen3_5Model (inner)
    self.lm.model.model.visual                # vision encoder — FROZEN (Verbalizer is text-only)
    self.lm.model.model.language_model        # transformer stack
    self.lm.model.model.language_model.embed_tokens
    self.lm.model.model.language_model.layers # 24 hybrid layers
    self.lm.model.lm_head                     # output projection

Forward calls bypass the vision encoder entirely — Verbalizer only processes
text sequences (τ+/τ−). All forward calls go through language_model directly.

Training schedule (controlled externally by train_stage2.py):
    Steps 0–3000   : warm-up — LM loss on τ+ (latents DETACHED → Vψ trains, Student does not)
    Steps 3000–4500: frozen  — DPO loss, Vψ frozen → gradient flows through CA into Student

LoRA: rank=32, targets all attention + DeltaNet-specific + FFN projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from transformers import AutoModelForImageTextToText
from peft import LoraConfig, TaskType, get_peft_model

# Architecture constants for Qwen3.5-0.8B
QWEN35_0_8B_HIDDEN_DIM = 1024
QWEN35_0_8B_NUM_LAYERS = 24

# Full LoRA target set — mirrors Student (covers all layer types in hybrid arch)
QWEN35_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "out_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# CrossAttentionBlock — one instance per transformer layer
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Cross-attends Verbalizer hidden states to Student latents z.

    Q  = Verbalizer hidden states  [batch, seq, d_verb]
    K,V = Student latents z, projected from d_student → d_verb

    Residual + post-norm applied to the CA output.

    Parameters
    ----------
    query_dim : Verbalizer hidden size   (d_verb = 1024)
    kv_dim    : Student hidden size      (d_student = 2560)
    num_heads : attention heads inside CA
    dropout   : dropout on CA weights
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert query_dim % num_heads == 0, (
            f"query_dim {query_dim} must be divisible by num_heads {num_heads}"
        )

        # Project Student latents (d_student) → Verbalizer space (d_verb)
        self.k_proj = nn.Linear(kv_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, query_dim, bias=False)

        self.attn     = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.q_norm   = nn.LayerNorm(query_dim, eps=1e-6)
        self.out_norm = nn.LayerNorm(query_dim, eps=1e-6)

        self.gate = nn.Parameter(torch.tensor(-3.5))

    def forward(
        self,
        hidden: torch.Tensor,    # [batch, seq, d_verb]
        latents: torch.Tensor,   # [batch, M, d_student]
    ) -> torch.Tensor:
        k = self.k_proj(latents)                  # [batch, M, d_verb]
        v = self.v_proj(latents)                  # [batch, M, d_verb]
        q = self.q_norm(hidden)                   # [batch, seq, d_verb]
        ca_out, _ = self.attn(q, k, v)            # [batch, seq, d_verb]
        # return self.out_norm(hidden + ca_out)      # residual + post-norm
        return hidden  + torch.sigmoid(self.gate) * self.out_norm(ca_out)   # residual + learned gate * CA + post-norm
        # --> if gate = 0, output becomes hidden (base model's raw state). if i keep our_norm outside, i'd get out_norm(hidden)
        # which instroduces a subtl normalization shift at step 0. the gated form preserves the base distribution perfectly
        


# ---------------------------------------------------------------------------
# Verbalizer
# ---------------------------------------------------------------------------

class Verbalizer(nn.Module):
    """
    Qwen3.5-0.8B with persistent per-layer CA hooks conditioned on Student latents.

    Parameters
    ----------
    model_name     : HuggingFace repo ID   (default: unsloth/Qwen3.5-0.8B)
    student_hidden : Student hidden size   (2560 for Qwen3.5-4B)
    lora_rank      : LoRA rank             (default: 32)
    lora_alpha     : LoRA scaling          (default: 64)
    ca_num_heads   : heads per CA block    (default: 8)
    ca_dropout     : CA dropout            (default: 0.0)
    dpo_beta       : β for DPO loss        (default: 0.1)
    """

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3.5-0.8B",
        student_hidden: int = 2560,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        ca_num_heads: int = 8,
        ca_dropout: float = 0.0,
        dpo_beta: float = 0.1,
        new_vocab_size: int = -1,           # vocab size after <answer> registration
    ):
        super().__init__()
        self.dpo_beta = dpo_beta

        # ------------------------------------------------------------------
        # 1. Load Qwen3.5-0.8B
        # ------------------------------------------------------------------
        base = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )

        # ------------------------------------------------------------------
        # 2. Freeze vision encoder — Verbalizer is text-only.
        #    τ+/τ− are pure text sequences; the visual head is never used.
        # ------------------------------------------------------------------
        if hasattr(base.model, "visual"):
            for p in base.model.visual.parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------
        # 3. Wrap with LoRA
        # ------------------------------------------------------------------
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=QWEN35_LORA_TARGETS,
            bias="none",
        )
        self.lm = get_peft_model(base, lora_cfg)

        # Resize embedding table to match extended tokenizer (<answer> token)
        if new_vocab_size > 0:
            self.lm.resize_token_embeddings(new_vocab_size)

        # ------------------------------------------------------------------
        # 4. Read architecture constants
        #    Path: self.lm.model.model.config.text_config.hidden_size
        # ------------------------------------------------------------------
        inner_cfg = getattr(
            self.lm.model.model.config, "text_config", self.lm.model.model.config
        )
        self.hidden_dim: int = getattr(inner_cfg, "hidden_size",      QWEN35_0_8B_HIDDEN_DIM)
        self.num_layers: int = getattr(inner_cfg, "num_hidden_layers", QWEN35_0_8B_NUM_LAYERS)

        # ------------------------------------------------------------------
        # 5. CrossAttentionBlocks — one per layer
        # ------------------------------------------------------------------
        self.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(self.hidden_dim, student_hidden, ca_num_heads, ca_dropout)
            for _ in range(self.num_layers)
        ])
        self.ca_blocks.to(self.lm.device)
        self.ca_blocks.to(torch.bfloat16)

        # ------------------------------------------------------------------
        # 6. Instance variable for passing latents into persistent hooks.
        #    Set immediately before each forward call; cleared in try/finally.
        #    None when no forward is running.
        # ------------------------------------------------------------------
        self._current_latents: Optional[torch.Tensor] = None

        # ------------------------------------------------------------------
        # 7. Register persistent hooks on each transformer layer.
        #    Hooks fire after the layer's own computation and inject CA output.
        #    Registered ONCE here; never re-registered.
        # ------------------------------------------------------------------
        self._hooks: List = []
        self._register_persistent_hooks()

        # Freeze state flag
        self._frozen: bool = False

    # -----------------------------------------------------------------------
    # Persistent hook registration (called once in __init__)
    # -----------------------------------------------------------------------

    def _register_persistent_hooks(self):
        """
        Register one post-forward hook per transformer layer.

        The hook reads self._current_latents at call time (not at registration
        time), so the same hook correctly uses different latents each forward
        call. This is more efficient than per-call register/remove and avoids
        hook leaks if an exception is raised mid-forward.

        Hook logic:
            output[0] is always the hidden state tensor for both
            GatedAttention and GatedDeltaNet layers in Qwen3.5.
            We inject CA, return the modified tuple.
        """
        layers = self.lm.model.model.language_model.layers

        for layer_idx, layer in enumerate(layers):
            ca_block = self.ca_blocks[layer_idx]
            verbalizer_ref = self   # capture self for latent access

            def make_hook(block, vref):
                def hook(module, inp, output):
                    # Guard: if latents not set (e.g. during parameter init),
                    # pass through unchanged
                    if vref._current_latents is None:
                        return output

                    h = output[0] if isinstance(output, tuple) else output # [B, seq, d_verb]
                    latents = vref._current_latents.to(h.dtype)
                    h = block(h, latents)  # CA injection
                    return (h,) + output[1:] if isinstance(output, tuple) else h  # return modified tuple
                return hook

            handle = layer.register_forward_hook(make_hook(ca_block, verbalizer_ref))
            self._hooks.append(handle)

    # -----------------------------------------------------------------------
    # Context manager for safe latent injection
    # -----------------------------------------------------------------------

    def _with_latents(self, latents: torch.Tensor):
        """
        Returns a context manager that sets self._current_latents before
        entering the forward pass and clears it after (even on exception).

        Usage:
            with self._with_latents(z):
                out = self.lm.model.model.language_model(...)
        """
        verbalizer_ref = self

        class _LatentContext:
            def __enter__(self_):
                verbalizer_ref._current_latents = latents
                return self_

            def __exit__(self_, exc_type, exc_val, exc_tb):
                # DO NOT clear latents here. Gradient checkpointing requires latents 
                # to persist until the backward pass is complete.
                # Latents will be cleared manually via self.clear_latents() after backprop.
                return False   # do not suppress exceptions

        return _LatentContext()

    # -----------------------------------------------------------------------
    # Internal: language-model-only forward (text path)
    # -----------------------------------------------------------------------

    @property
    def _language_model(self) -> nn.Module:
        """Transformer stack — embed_tokens + layers + norm."""
        return self.lm.model.model.language_model

    @property
    def _embed_tokens(self) -> nn.Embedding:
        return self._language_model.embed_tokens

    @property
    def _lm_head(self) -> nn.Linear:
        return self.lm.model.lm_head

    def _lm_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run a text-only forward pass through the language model with CA hooks active.
        Vision encoder is bypassed entirely — τ+/τ− are pure text.

        Returns
        -------
        logits : [batch, seq, vocab_size]
        loss   : scalar CE loss if labels provided, else None
        """
        embeds = self._embed_tokens(input_ids)    # [B, seq, d_verb]

        with self._with_latents(latents):
            out = self._language_model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )

        logits = self._lm_head(out.last_hidden_state)   # [B, seq, vocab]

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    # -----------------------------------------------------------------------
    # Sequence log-probs (used by DPO)
    # -----------------------------------------------------------------------

    def _compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,
        response_mask: torch.Tensor,   # [B, seq] — 1 on response tokens only
    ) -> torch.Tensor:
        """
        Compute sum of per-token log-probs over response positions.

        log π_ψ(response | z) = Σ_{t∈response} log p(token_t | token_{<t}, z)

        Returns
        -------
        seq_log_probs : [batch]
        """
        logits, _ = self._lm_forward(input_ids, attention_mask, latents)

        log_probs = F.log_softmax(logits, dim=-1)           # [B, seq, vocab]

        # Shift: logits[i] predicts token[i+1]
        shift_lp   = log_probs[:, :-1, :]                  # [B, seq-1, vocab]
        shift_ids  = input_ids[:, 1:]                       # [B, seq-1]
        shift_mask = response_mask[:, 1:]                   # [B, seq-1]

        token_lp   = shift_lp.gather(
            -1, shift_ids.unsqueeze(-1)
        ).squeeze(-1)                                        # [B, seq-1]

        return (token_lp * shift_mask).sum(dim=-1)          # [B]

    # -----------------------------------------------------------------------
    # Public loss: LM warm-up
    # -----------------------------------------------------------------------

    def compute_lm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,         # pass z.detach() during warm-up
        labels: torch.Tensor,          # -100 on prompt tokens, real ids on response
    ) -> torch.Tensor:
        """
        Warm-up loss: cross-entropy on τ+ tokens.

        The Verbalizer's CA blocks and LoRA learn to translate Student latents
        into high-quality reasoning text. Passing latents.detach() ensures
        this loss does NOT update the Student's weights during warm-up —
        the Student is only updated by L_distill + L_ans + L_spatial at this stage.

        Returns
        -------
        lm_loss : scalar
        """
        _, loss = self._lm_forward(input_ids, attention_mask, latents, labels=labels)
        return loss

    # -----------------------------------------------------------------------
    # Public loss: DPO
    # -----------------------------------------------------------------------

    def compute_dpo_loss(
        self,
        pos_input_ids: torch.Tensor,
        neg_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        neg_attention_mask: torch.Tensor,
        latents: torch.Tensor,                  # NOT detached in frozen phase
        pos_response_mask: torch.Tensor,
        neg_response_mask: torch.Tensor,
        ref_pos_log_probs: Optional[torch.Tensor] = None,
        ref_neg_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        DPO preference loss.

        When Verbalizer is frozen (step > 3000), all Vψ parameters have
        requires_grad=False. Gradients from this loss flow through the
        latents tensor → CA k_proj/v_proj computation → back to the Student.

        L_DPO = -E[ log σ( β * (log π(τ+|z) − log π(τ−|z)) ) ]
        (reference-free variant when ref log-probs not provided)

        Returns
        -------
        dpo_loss : scalar
        metrics  : dict for logging
        """
        log_pi_pos = self._compute_sequence_log_probs(
            pos_input_ids, pos_attention_mask, latents, pos_response_mask
        )   # [B]
        log_pi_neg = self._compute_sequence_log_probs(
            neg_input_ids, neg_attention_mask, latents, neg_response_mask
        )   # [B]

        if ref_pos_log_probs is not None and ref_neg_log_probs is not None:
            margin = self.dpo_beta * (
                (log_pi_pos - ref_pos_log_probs) - (log_pi_neg - ref_neg_log_probs)
            )
        else:
            margin = self.dpo_beta * (log_pi_pos - log_pi_neg)

        dpo_loss = -F.logsigmoid(margin).mean()

        with torch.no_grad():
            metrics = {
                "dpo_loss":      dpo_loss.item(),
                "reward_margin": margin.mean().item(),
                "dpo_accuracy":  (margin > 0).float().mean().item(),
                "log_pi_pos":    log_pi_pos.mean().item(),
                "log_pi_neg":    log_pi_neg.mean().item(),
            }

        return dpo_loss, metrics

    # -----------------------------------------------------------------------
    # Freeze / unfreeze schedule
    # -----------------------------------------------------------------------

    def freeze_for_student_training(self):
        """
        Called at step 3000. Freezes all Vψ parameters permanently.

        After this, DPO gradients flow through the frozen CA computation
        graph → latents → Student LoRA weights. The Verbalizer's own
        parameters no longer receive any gradient updates.

        The persistent hooks remain registered and functional — they still
        execute and still inject CA output. The only change is that
        requires_grad=False on CA parameters means their weights don't update.
        """
        if self._frozen:
            return
        for p in self.parameters():
            p.requires_grad = False
        self._frozen = True
        remaining = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Verbalizer] Frozen. Trainable params remaining: {remaining:,}")

    def unfreeze_ca_and_lora(self):
        """
        Re-enables CA blocks and LoRA weights for gradient flow.
        Used when resuming from a warm-up checkpoint.
        """
        if not self._frozen:
            return
        for p in self.ca_blocks.parameters():
            p.requires_grad = True
        for n, p in self.lm.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        self._frozen = False

    def is_frozen(self) -> bool:
        return self._frozen

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def print_trainable_parameters(self):
        self.lm.print_trainable_parameters()
        ca = sum(p.numel() for p in self.ca_blocks.parameters())
        print(f"  ca_blocks  : {ca:,} params  [TRAINABLE until step 3000]")
        print(f"  hidden_dim : {self.hidden_dim}")
        print(f"  num_layers : {self.num_layers}")

    @staticmethod
    def stack_latents(latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Converts Student's List[M × [batch, d]] output to [batch, M, d]
        as expected by CA blocks. Call this before passing latents to any
        Verbalizer method.
        """
        return torch.stack(latents, dim=1)   # [batch, M, d]

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def clear_latents(self):
        """
        Clears the stored latents tensor to prevent memory leaks across
        training steps after gradient checkpointing is finished.
        """
        self._current_latents = None

    @torch.no_grad()
    def generate_from_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,
        generation_config=None,
    ) -> torch.Tensor:
        """
        Generate text conditioned on the given latents.
        This sets self._current_latents and calls self.lm.generate.
        The persistent hooks will automatically inject the latents into the layers.
        """
        self._current_latents = latents
        
        # Turn off caching override if gradient checkpointing is somehow active
        was_training = self.lm.training
        self.lm.eval()
        
        try:
            outputs = self.lm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                use_cache=True,
            )
        finally:
            if was_training:
                self.lm.train()
            # We don't automatically clear latents here in case the caller wants to 
            # inspect them, but usually they are cleared later.
            self.clear_latents()
            
        return outputs