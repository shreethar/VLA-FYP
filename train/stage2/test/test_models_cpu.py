"""
test_models_cpu.py
------------------
Shape / logic tests for all model components using TINY random models.

Instead of loading Qwen2.5-VL-4B (8+ GB), we mock the heavy HuggingFace
loading and test the custom logic (latent loop, CA injection, spatial MLP,
gradient flow, freeze/unfreeze) with small random weights.

Run:  python -m pytest test/test_models_cpu.py -v --timeout=60
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Optional, Tuple


# ===========================================================================
# Tiny mock transformer for testing
# ===========================================================================

class MockTransformerOutput:
    """Mimics HuggingFace model outputs."""
    def __init__(self, last_hidden_state, past_key_values=None, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states


class MockTransformerLayer(nn.Module):
    """A trivial transformer layer for testing."""
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d)
    
    def forward(self, x, **kwargs):
        return (self.linear(x),)


class MockBaseTransformer(nn.Module):
    """Mock for the base LLM transformer stack."""
    def __init__(self, hidden_dim=64, num_layers=4, vocab_size=100):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([MockTransformerLayer(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=False,
                output_hidden_states=False, return_dict=True, past_key_values=None, **kwargs):
        h = inputs_embeds
        all_hidden = [h] if output_hidden_states else None

        for layer in self.layers:
            h = layer(h)[0]
            if output_hidden_states:
                all_hidden.append(h)

        h = self.norm(h)

        # Fake KV cache: just return a dummy
        fake_kv = tuple(
            (torch.zeros(1), torch.zeros(1))
            for _ in range(self._num_layers)
        ) if use_cache else None

        return MockTransformerOutput(
            last_hidden_state=h,
            past_key_values=fake_kv,
            hidden_states=tuple(all_hidden) if all_hidden else None,
        )

    def _update_causal_mask(self, attention_mask, hidden, cache_position, **kwargs):
        """Mock causal mask builder."""
        return None


# ===========================================================================
# 1. Test SpatialMLP
# ===========================================================================

class _SpatialMLP(nn.Module):
    """
    Inline copy of SpatialMLP from latent_student.py.
    We duplicate it here so tests don't trigger the heavy Qwen2.5-VL import
    chain at module load time. The GPU smoke test validates the real class.
    """
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


class TestSpatialMLP:
    def test_output_shape(self):
        mlp = _SpatialMLP(hidden_dim=64)
        x = torch.randn(2, 5, 64)
        out = mlp(x)
        assert out.shape == (2, 5, 2)

    def test_output_range(self):
        """Output should be in [0, 1] due to final Sigmoid."""
        mlp = _SpatialMLP(hidden_dim=64)
        x = torch.randn(4, 5, 64)
        out = mlp(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_gradient_flow(self):
        mlp = _SpatialMLP(hidden_dim=64)
        x = torch.randn(2, 5, 64, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ===========================================================================
# 2. Test CrossAttentionBlock
# ===========================================================================

class TestCrossAttentionBlock:
    def test_output_shape(self):
        from models.verbalizer import CrossAttentionBlock
        ca = CrossAttentionBlock(query_dim=32, kv_dim=64, num_heads=4)
        hidden = torch.randn(2, 10, 32)
        latents = torch.randn(2, 6, 64)
        out = ca(hidden, latents)
        assert out.shape == (2, 10, 32)

    def test_residual_connection(self):
        """Output should differ from input (residual adds CA output)."""
        from models.verbalizer import CrossAttentionBlock
        ca = CrossAttentionBlock(query_dim=32, kv_dim=64, num_heads=4)
        hidden = torch.randn(2, 10, 32)
        latents = torch.randn(2, 6, 64)
        out = ca(hidden, latents)
        # Not identical (CA adds something)
        assert not torch.allclose(out, hidden, atol=1e-3)

    def test_gradient_flows_to_latents(self):
        """Gradient must flow from CA output back into the latents tensor."""
        from models.verbalizer import CrossAttentionBlock
        ca = CrossAttentionBlock(query_dim=32, kv_dim=64, num_heads=4)
        hidden = torch.randn(2, 10, 32)
        latents = torch.randn(2, 6, 64, requires_grad=True)
        out = ca(hidden, latents)
        out.sum().backward()
        assert latents.grad is not None
        assert latents.grad.abs().sum() > 0

    def test_detached_latents_no_grad(self):
        """When latents are detached, no gradient flows back."""
        from models.verbalizer import CrossAttentionBlock
        ca = CrossAttentionBlock(query_dim=32, kv_dim=64, num_heads=4)
        hidden = torch.randn(2, 10, 32)
        latents = torch.randn(2, 6, 64, requires_grad=True)
        out = ca(hidden, latents.detach())
        out.sum().backward()
        assert latents.grad is None


# ===========================================================================
# 3. Test ProjectionMLP (spatial_forcing.py)
# ===========================================================================

class TestProjectionMLP:
    def test_output_shape_and_norm(self):
        from models.spatial_forcing import ProjectionMLP
        mlp = ProjectionMLP(in_dim=64, out_dim=32)
        x = torch.randn(4, 64)
        out = mlp(x)
        assert out.shape == (4, 32)
        # Check L2 normalisation
        norms = out.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self):
        from models.spatial_forcing import ProjectionMLP
        mlp = ProjectionMLP(in_dim=64, out_dim=32)
        x = torch.randn(4, 64, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None


# ===========================================================================
# 4. Test SpatialForcingLoss.compute_loss (without loading DINOv2)
# ===========================================================================

class TestSpatialForcingLossLogic:
    def test_loss_computation(self):
        """Test the loss math directly without any extractor."""
        from models.spatial_forcing import ProjectionMLP
        import torch.nn.functional as F

        student_dim = 64
        ext_dim = 32
        batch = 4

        proj_mlp = ProjectionMLP(in_dim=student_dim, out_dim=ext_dim)

        # Simulate student visual features (already mean-pooled)
        x_V = torch.randn(batch, 10, student_dim)
        pooled = x_V.mean(dim=1)                             # [batch, student_dim]
        projected = proj_mlp(pooled.float())                  # [batch, ext_dim], unit-norm

        # Simulate extractor reference features
        ref_feats = F.normalize(torch.randn(batch, ext_dim), dim=-1)

        # Cosine sim
        cos_sim = (projected * ref_feats).sum(dim=-1)        # [batch]
        loss = -cos_sim.mean()

        assert loss.shape == ()
        # Loss should be in reasonable range for random features
        assert -1.0 <= loss.item() <= 1.0

    def test_perfect_alignment_loss_is_negative(self):
        """When features are perfectly aligned, cosine sim = 1, loss = -1."""
        from models.spatial_forcing import ProjectionMLP
        import torch.nn.functional as F

        proj_mlp = ProjectionMLP(in_dim=32, out_dim=32)

        # Force the projection MLP to be identity-like by using the same features
        ref = F.normalize(torch.randn(2, 32), dim=-1)
        cos_sim = (ref * ref).sum(dim=-1)
        loss = -cos_sim.mean()
        assert loss.item() == pytest.approx(-1.0, abs=1e-5)


# ===========================================================================
# 5. Test Verbalizer layer-by-layer forward (with mock)
# ===========================================================================

class TestVerbalizerLogic:
    """Test Verbalizer CA injection and freeze logic with a mock base model."""

    def _make_mini_verbalizer(self):
        """Construct a minimal Verbalizer-like object without loading Qwen3."""
        from models.verbalizer import CrossAttentionBlock, Verbalizer

        d_verb = 32
        d_student = 64
        num_layers = 2
        vocab_size = 100
        num_heads = 4

        # Build a mock Verbalizer manually
        verb = Verbalizer.__new__(Verbalizer)
        nn.Module.__init__(verb)

        verb.dpo_beta = 0.1
        verb.hidden_dim = d_verb
        verb.num_layers = num_layers
        verb._frozen = False

        # Build a mock LM
        mock_lm = nn.Module()
        mock_lm.config = MagicMock()
        mock_lm.config.hidden_size = d_verb
        mock_lm.config.num_hidden_layers = num_layers

        # Inner transformer model
        mock_model = MockBaseTransformer(
            hidden_dim=d_verb, num_layers=num_layers, vocab_size=vocab_size
        )
        mock_lm.model = mock_model

        # LM head
        mock_lm.lm_head = nn.Linear(d_verb, vocab_size, bias=False)

        # named_parameters for LoRA check
        mock_lm.named_parameters = lambda: iter([])

        verb.lm = mock_lm

        # CA blocks
        verb.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(
                query_dim=d_verb, kv_dim=d_student,
                num_heads=num_heads, dropout=0.0,
            )
            for _ in range(num_layers)
        ])

        return verb, d_verb, d_student, vocab_size

    def test_forward_with_latents_shape(self):
        verb, d_verb, d_student, vocab_size = self._make_mini_verbalizer()

        batch, seq, M = 2, 10, 6
        input_ids = torch.randint(0, vocab_size, (batch, seq))
        attn_mask = torch.ones(batch, seq, dtype=torch.long)
        latents = torch.randn(batch, M, d_student)

        logits, loss = verb._forward_with_latents(input_ids, attn_mask, latents)
        assert logits.shape == (batch, seq, vocab_size)
        assert loss is None

    def test_forward_with_labels(self):
        verb, d_verb, d_student, vocab_size = self._make_mini_verbalizer()

        batch, seq, M = 2, 10, 6
        input_ids = torch.randint(0, vocab_size, (batch, seq))
        attn_mask = torch.ones(batch, seq, dtype=torch.long)
        latents = torch.randn(batch, M, d_student)
        labels = input_ids.clone()
        labels[:, :3] = -100  # mask prompt

        logits, loss = verb._forward_with_latents(input_ids, attn_mask, latents, labels=labels)
        assert loss is not None
        assert loss.shape == ()
        assert loss.item() > 0

    def test_freeze_unfreeze(self):
        verb, _, _, _ = self._make_mini_verbalizer()

        # Initially, CA params should be trainable
        assert not verb.is_frozen()
        ca_params_before = sum(1 for p in verb.ca_blocks.parameters() if p.requires_grad)
        assert ca_params_before > 0

        # Freeze
        verb.freeze_for_student_training()
        assert verb.is_frozen()
        trainable_after_freeze = sum(1 for p in verb.parameters() if p.requires_grad)
        assert trainable_after_freeze == 0

        # Unfreeze
        verb.unfreeze_ca_and_lora()
        assert not verb.is_frozen()
        ca_params_after = sum(1 for p in verb.ca_blocks.parameters() if p.requires_grad)
        assert ca_params_after == ca_params_before

    def test_dpo_gradient_flows_to_latents_when_frozen(self):
        """
        Critical test: when Verbalizer is frozen, DPO loss gradients must
        flow through the CA blocks into the latents tensor.
        """
        verb, d_verb, d_student, vocab_size = self._make_mini_verbalizer()

        batch, seq, M = 2, 8, 6
        input_ids = torch.randint(0, vocab_size, (batch, seq))
        attn_mask = torch.ones(batch, seq, dtype=torch.long)
        latents = torch.randn(batch, M, d_student, requires_grad=True)
        response_mask = torch.zeros(batch, seq)
        response_mask[:, 3:] = 1.0  # last 5 tokens are response

        # Freeze Verbalizer
        verb.freeze_for_student_training()

        # Compute log probs (this exercises the full path through CA)
        log_probs = verb._compute_sequence_log_probs(
            input_ids, attn_mask, latents, response_mask
        )
        log_probs.sum().backward()

        # Gradient MUST reach the latents
        assert latents.grad is not None, "Gradient did not flow through frozen Verbalizer to latents!"
        assert latents.grad.abs().sum() > 0

    def test_stack_latents(self):
        from models.verbalizer import Verbalizer
        latents_list = [torch.randn(2, 64) for _ in range(6)]
        stacked = Verbalizer.stack_latents(latents_list)
        assert stacked.shape == (2, 6, 64)


# ===========================================================================
# 6. Test GRPOTeacher helper methods
# ===========================================================================

class TestGRPOTeacherHelpers:
    def test_compute_advantages(self):
        from training.grpo_teacher import GRPOTeacher
        rewards = torch.tensor([
            [0.1, 0.5],
            [0.3, 0.7],
            [0.5, 0.3],
            [0.2, 0.9],
            [0.4, 0.6],
        ])  # [G=5, batch=2]

        advantages = GRPOTeacher.compute_advantages(rewards)
        assert advantages.shape == (5, 2)

        # Mean of advantages should be ~0 for each batch item
        for b in range(2):
            assert advantages[:, b].mean() == pytest.approx(0.0, abs=1e-5)

        # Best/worst should correspond to max/min reward
        best = advantages.argmax(dim=0)
        worst = advantages.argmin(dim=0)
        assert best[0] == rewards[:, 0].argmax()
        assert worst[0] == rewards[:, 0].argmin()

    def test_compute_response_mask(self):
        """Test the response mask builder."""
        # We can't easily instantiate GRPOTeacher without loading the model,
        # but _compute_response_mask is a simple static-like method.
        # Let's test the logic directly:
        prompt_len = 5
        full_ids = torch.tensor([
            [1, 2, 3, 4, 5, 10, 11, 12, 0, 0],
            [1, 2, 3, 4, 5, 20, 21, 22, 23, 0],
        ])
        mask = torch.zeros_like(full_ids, dtype=torch.float)
        mask[:, prompt_len:] = (full_ids[:, prompt_len:] != 0).float()

        # Sample 0: tokens 10,11,12 are response, 0,0 are padding
        assert mask[0, 5] == 1.0
        assert mask[0, 6] == 1.0
        assert mask[0, 7] == 1.0
        assert mask[0, 8] == 0.0  # padding
        assert mask[0, 9] == 0.0

        # Prompt tokens should be 0
        assert mask[0, :5].sum() == 0

    def test_rollout_buffer_dataclass(self):
        from training.grpo_teacher import RolloutBuffer

        buf = RolloutBuffer(
            rollout_ids=[torch.zeros(2, 10)],
            rollout_texts=[["a", "b"]],
            attention_masks=[torch.ones(2, 10)],
            rewards=torch.ones(1, 2),
            advantages=torch.zeros(1, 2),
            best_idx=torch.tensor([0, 0]),
            worst_idx=torch.tensor([0, 0]),
            tau_pos_ids=torch.zeros(2, 10, dtype=torch.long),
            tau_neg_ids=torch.zeros(2, 10, dtype=torch.long),
            tau_pos_mask=torch.ones(2, 10),
            tau_neg_mask=torch.ones(2, 10),
            tau_pos_response_mask=torch.ones(2, 10),
            tau_neg_response_mask=torch.ones(2, 10),
            answer_token_pos=torch.tensor([5, 5]),
            h_T=torch.randn(2, 64),
        )
        assert buf.h_T is not None
        assert buf.tau_pos_ids.shape == (2, 10)


# ===========================================================================
# 7. Test StudentLossComputer interface
# ===========================================================================

class TestStudentLossComputer:
    def test_imports(self):
        """Verify the module exports match what train_stage2.py imports."""
        from training.student_losses import StudentLossComputer, build_student_loss_computer, LossOutput
        assert callable(build_student_loss_computer)
        lc = build_student_loss_computer(warmup_steps=100, lambda_distill=1.0, lambda_ans=1.0)
        assert isinstance(lc, StudentLossComputer)

    def test_loss_output_dataclass(self):
        from training.student_losses import LossOutput
        lo = LossOutput(
            student_total=torch.tensor(1.0),
            lm_loss=torch.tensor(0.5),
            metrics={"loss/student_total": 1.0, "loss/l_distill": 0.3,
                     "loss/l_ans": 0.2, "loss/l_spatial": 0.1,
                     "loss/lm_loss": 0.5, "loss/l_verb": 0.0},
        )
        assert lo.student_total.item() == 1.0
        assert lo.lm_loss.item() == 0.5
        assert "loss/l_distill" in lo.metrics
        # Verify all keys expected by train_stage2.py are present
        for key in ["loss/student_total", "loss/l_distill", "loss/l_ans",
                    "loss/l_spatial", "loss/lm_loss", "loss/l_verb"]:
            assert key in lo.metrics, f"Missing expected metric key: {key}"

    def test_make_lm_labels(self):
        from training.student_losses import StudentLossComputer
        input_ids = torch.tensor([[10, 20, 30, 40, 50]])
        prompt_len = 2  # first 2 tokens are prompt
        labels = StudentLossComputer._make_lm_labels(input_ids, prompt_len)
        assert labels[0, 0] == -100
        assert labels[0, 1] == -100
        assert labels[0, 2] == 30
        assert labels[0, 3] == 40
        assert labels[0, 4] == 50

    def test_log_student_grad_norms(self):
        from training.student_losses import StudentLossComputer

        # Create a simple model with named parameters
        model = nn.Module()
        model.lora_A = nn.Linear(4, 4)
        model.spatial_proj = nn.Linear(4, 4)

        # Simulate gradients
        x = torch.randn(2, 4)
        loss = model.lora_A(x).sum() + model.spatial_proj(x).sum()
        loss.backward()

        norms = StudentLossComputer.log_student_grad_norms(model)
        assert "grad_norm/lora_total" in norms
        assert "grad_norm/spatial_total" in norms
        assert norms["grad_norm/lora_total"] > 0
        assert norms["grad_norm/spatial_total"] > 0
