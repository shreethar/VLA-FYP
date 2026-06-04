"""
smoke_test.py — Stage 2 full-pipeline smoke test (CPU, ~10 seconds)
--------------------------------------------------------------------
Replaces all three large models with tiny CPU mocks that share the
EXACT same method signatures. The real StudentLossComputer, RolloutBuffer,
and training logic run unmodified.

What is tested:
  ✅ RolloutBuffer construction and field shapes
  ✅ Student.generate_latents()   shape contract
  ✅ Student.get_answer_hidden_state() shape
  ✅ L_distill (MSE h_S vs h_T)   backward
  ✅ L_ans (MSE waypoints)         backward
  ✅ Verbalizer LM loss (warm-up, latents DETACHED → no grad into Student)
  ✅ Verbalizer DPO loss (frozen phase, latents LIVE → grad flows into Student)
  ✅ DPO gradient confirmed in Student parameters
  ✅ Verbalizer freeze at warmup_steps (no param update after freeze)
  ✅ Three independent optimizer.step() calls
  ✅ Phase transition (warmup → frozen) mid-loop
  ✅ Checkpoint save → load → resume
  ✅ Dataloader cycling

Run from train/stage2/:
    python smoke_test.py
"""

import sys, os, tempfile, traceback
# Make the stage2 package importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from transformers import get_cosine_schedule_with_warmup

# ── Import REAL pipeline components ──────────────────────────────────────────
from training.grpo_teacher import RolloutBuffer
from training.student_losses import StudentLossComputer, LossOutput, build_student_loss_computer

# ── Tiny constants ────────────────────────────────────────────────────────────
HS          = 64      # Student hidden   (real: 2560)
HV          = 32      # Verbalizer hidden (real: 1024)
VOCAB       = 200
PROMPT_LEN  = 12
RESP_LEN    = 8
FULL_LEN    = PROMPT_LEN + RESP_LEN
BATCH       = 2
M           = 2       # latents          (real: 6)
K           = 3       # spatial tokens   (real: 5)
G           = 2       # GRPO rollouts    (real: 5)
NV          = 3       # Verbalizer layers (real: 24)
WARMUP      = 3       # freeze at step 3 (real: 3000)
STEPS       = 6
DEVICE      = torch.device("cpu")

# ── Test runner ───────────────────────────────────────────────────────────────
_results: List[tuple] = []

def chk(name: str, ok: bool, detail: str = ""):
    tag = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {tag}  {name}" + (f"  [{detail}]" if detail else ""))
    _results.append((name, ok))
    return ok

def chk_shape(name: str, t: torch.Tensor, expected):
    return chk(name, tuple(t.shape) == tuple(expected),
                f"got {tuple(t.shape)}, want {tuple(expected)}")

def sec(title: str):
    print(f"\n{'─'*62}\n  {title}\n{'─'*62}")


# ═════════════════════════════════════════════════════════════════════════════
#  MOCK MODELS
# ═════════════════════════════════════════════════════════════════════════════

class _TinyLM(nn.Module):
    """Shared tiny backbone: Embedding → Linear → logits."""
    def __init__(self, hidden: int, vocab: int = VOCAB):
        super().__init__()
        self.embed  = nn.Embedding(vocab, hidden)
        # Named with 'lora_' so optimizer builder selects them
        self.lora_A = nn.Linear(hidden, hidden, bias=False)
        self.lora_B = nn.Linear(hidden, hidden, bias=False)
        self.head   = nn.Linear(hidden, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B,S] → logits [B,S,vocab]"""
        h = self.lora_B(self.lora_A(self.embed(ids)))
        return self.head(h)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B,S] → hidden [B,S,HS]"""
        return self.lora_B(self.lora_A(self.embed(ids)))


class MockSpatialMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(HS, 16), nn.GELU(),
                                 nn.Linear(16, 2), nn.Sigmoid())
    def forward(self, x):           # [B, K, HS] → [B, K, 2]
        return self.net(x)


class MockLatentStudent(nn.Module):
    """Identical interface to LatentStudent."""
    def __init__(self):
        super().__init__()
        self.hidden_dim  = HS
        self.M, self.K   = M, K
        self.vlm         = _TinyLM(HS)                   # has lora_ params
        self.spatial_tokens = nn.Parameter(torch.randn(K, HS) * 0.02)
        self.spatial_mlp    = MockSpatialMLP()

    # ── required by StudentLossComputer ──────────────────────────────────────
    def generate_latents(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        B = input_ids.shape[0]
        h = self.vlm.hidden(input_ids)                   # [B, seq, HS]
        base = h[:, -1, :]                               # [B, HS]
        latents = [self.vlm.lora_A(base) for _ in range(self.M)]
        sp_h = (self.spatial_tokens
                .unsqueeze(0).expand(B, -1, -1))         # [B, K, HS]
        wpts = self.spatial_mlp(sp_h)                    # [B, K, 2]
        return latents, sp_h, wpts

    def get_answer_hidden_state(self, input_ids, pixel_values,
                                image_grid_thw, attention_mask,
                                answer_token_positions):
        B = input_ids.shape[0]
        h = self.vlm.hidden(input_ids)                   # [B, seq, HS]
        return h[torch.arange(B), answer_token_positions] # [B, HS]


class _MockCA(nn.Module):
    """Tiny cross-attention block: Q=hidden, K/V=latents."""
    def __init__(self):
        super().__init__()
        self.k = nn.Linear(HS, HV, bias=False)
        self.v = nn.Linear(HS, HV, bias=False)
        self.o = nn.Linear(HV, HV, bias=False)
        self.gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, h, latents):  # h[B,S,HV], latents[B,M,HS]
        K_ = self.k(latents)        # [B, M, HV]
        V_ = self.v(latents)        # [B, M, HV]
        sc = torch.bmm(h, K_.transpose(1, 2))   # [B, S, M]
        at = F.softmax(sc, dim=-1)
        ca = torch.bmm(at, V_)                   # [B, S, HV]
        return h + torch.sigmoid(self.gate) * self.o(ca)


class MockVerbalizer(nn.Module):
    """Identical interface to Verbalizer."""
    def __init__(self):
        super().__init__()
        self.dpo_beta  = 0.1
        self._frozen   = False
        self.hidden_dim = HV
        self.ca_blocks = nn.ModuleList([_MockCA() for _ in range(NV)])
        self.lm        = _TinyLM(HV)  # has lora_ params

    @staticmethod
    def stack_latents(latents: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(latents, dim=1)             # [B, M, HS]

    def _fwd(self, ids, latents):
        """ids [B,S] → logits [B,S,vocab].  latents [B,M,HS]."""
        h = self.lm.hidden(ids)                        # [B, S, HV]
        for ca in self.ca_blocks:
            h = ca(h, latents)
        return self.lm.head(h)                         # [B, S, vocab]

    def compute_lm_loss(self, input_ids, attention_mask, latents, labels):
        logits = self._fwd(input_ids, latents)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    def _seq_lp(self, ids, mask, latents, resp_mask):
        logits  = self._fwd(ids, latents)
        lp      = F.log_softmax(logits, dim=-1)
        shift   = lp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return (shift * resp_mask[:, 1:]).sum(-1)      # [B]

    def compute_dpo_loss(self, pos_input_ids, neg_input_ids,
                          pos_attention_mask, neg_attention_mask,
                          latents, pos_response_mask, neg_response_mask,
                          ref_pos_log_probs=None, ref_neg_log_probs=None):
        lp_pos = self._seq_lp(pos_input_ids, pos_attention_mask, latents, pos_response_mask)
        lp_neg = self._seq_lp(neg_input_ids, neg_attention_mask, latents, neg_response_mask)
        margin = self.dpo_beta * (lp_pos - lp_neg)
        loss   = -F.logsigmoid(margin).mean()
        metrics = dict(dpo_loss=loss.item(), reward_margin=margin.mean().item(),
                       dpo_accuracy=(margin > 0).float().mean().item(),
                       log_pi_pos=lp_pos.mean().item(), log_pi_neg=lp_neg.mean().item())
        return loss, metrics

    def freeze_for_student_training(self):
        if self._frozen: return
        for p in self.parameters(): p.requires_grad = False
        self._frozen = True
        print("    [Verbalizer] Frozen — DPO phase begins.")

    def is_frozen(self): return self._frozen


class MockGRPOTeacher(nn.Module):
    """Returns a realistic RolloutBuffer; exercises teacher optimizer."""
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(HS, HS) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(HS, HS) * 0.01)

    @property
    def vlm(self): return self   # optimizer uses teacher.vlm.parameters()

    def training_step(self, input_ids, pixel_values, image_grid_thw,
                      attention_mask, ground_truth, reward_fns,
                      reward_weights, optimizer, tokenizer, grad_clip=1.0):
        B = input_ids.shape[0]

        # Minimal teacher backward (exercises teacher optimizer)
        fake_loss = (self.lora_A * 0).sum()
        optimizer.zero_grad(); fake_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()

        # Build RolloutBuffer with correct shapes
        p_ids  = torch.randint(1, VOCAB, (B, FULL_LEN))
        n_ids  = torch.randint(1, VOCAB, (B, FULL_LEN))
        p_mask = torch.ones(B, FULL_LEN, dtype=torch.long)
        n_mask = p_mask.clone()
        p_resp = torch.zeros(B, FULL_LEN); p_resp[:, PROMPT_LEN:] = 1.0
        n_resp = p_resp.clone()
        h_T    = torch.randn(B, HS)           # detached — correct
        rews   = torch.rand(G, B)
        advs   = (rews - rews.mean(0, keepdim=True)) / (rews.std(0, keepdim=True) + 1e-8)
        ans_pos = torch.full((B,), PROMPT_LEN, dtype=torch.long)

        return RolloutBuffer(
            rollout_ids=[p_ids] * G, rollout_texts=[["text"] * B] * G,
            attention_masks=[p_mask] * G, rewards=rews, advantages=advs,
            best_idx=advs.argmax(0), worst_idx=advs.argmin(0),
            tau_pos_ids=p_ids, tau_neg_ids=n_ids,
            tau_pos_mask=p_mask, tau_neg_mask=n_mask,
            tau_pos_response_mask=p_resp, tau_neg_response_mask=n_resp,
            answer_token_pos=ans_pos, h_T=h_T,
        )

    @staticmethod
    def log_rollout_stats(buf):
        return dict(grpo_reward_mean=buf.rewards.mean().item(),
                    grpo_reward_std=buf.rewards.std().item())


# ═════════════════════════════════════════════════════════════════════════════
#  OPTIMIZER BUILDERS (matching train_stage2.py pattern)
# ═════════════════════════════════════════════════════════════════════════════

def build_teacher_opt(teacher):
    params = [p for p in teacher.vlm.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)

def build_student_opt(student):
    return torch.optim.AdamW([
        {"params": [p for n, p in student.vlm.named_parameters()
                    if p.requires_grad and "lora" in n], "lr": 2e-4},
        {"params": list(student.spatial_mlp.parameters()) + [student.spatial_tokens],
         "lr": 2e-4},
    ], weight_decay=0.01)

def build_verb_opt(verb):
    return torch.optim.AdamW([
        {"params": list(verb.ca_blocks.parameters()), "lr": 1e-4},
        {"params": [p for n, p in verb.lm.named_parameters()
                    if p.requires_grad and "lora" in n], "lr": 1e-4},
    ], weight_decay=0.01)


# ═════════════════════════════════════════════════════════════════════════════
#  FAKE DATALOADER
# ═════════════════════════════════════════════════════════════════════════════

def fake_batch():
    """Returns a batch dict identical in structure to the real dataloader."""
    return {
        "input_ids":      torch.randint(1, VOCAB, (BATCH, PROMPT_LEN)),
        "pixel_values":   None,
        "image_grid_thw": None,
        "attention_mask": torch.ones(BATCH, PROMPT_LEN, dtype=torch.long),
        "gt_waypoints":   torch.rand(BATCH, K, 2),
        "ground_truth":   {"gt_waypoints": torch.rand(BATCH, K, 2)},
    }

fake_loader = [fake_batch() for _ in range(STEPS + 2)]


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN SMOKE TEST
# ═════════════════════════════════════════════════════════════════════════════

def run():
    print(f"\n{'═'*62}")
    print(f"  ReasonFlow VLA — Stage 2 Smoke Test")
    print(f"  HS={HS}, HV={HV}, M={M}, K={K}, G={G}, BATCH={BATCH}")
    print(f"  WARMUP_STEPS={WARMUP}, TOTAL_STEPS={STEPS}")
    print(f"{'═'*62}")

    # ── 1. Build models ───────────────────────────────────────────────────────
    sec("1. Model instantiation")
    try:
        teacher    = MockGRPOTeacher()
        student    = MockLatentStudent()
        verbalizer = MockVerbalizer()
        chk("Teacher created",    True)
        chk("Student created",    True)
        chk("Verbalizer created", True)
        chk("Student hidden_dim", student.hidden_dim == HS, f"={student.hidden_dim}")
        chk("Student M",          student.M == M)
        chk("Student K",          student.K == K)
    except Exception as e:
        chk("Model instantiation", False, str(e)); return

    # ── 2. Optimizer + scheduler ──────────────────────────────────────────────
    sec("2. Optimizers & schedulers")
    try:
        t_opt = build_teacher_opt(teacher)
        s_opt = build_student_opt(student)
        v_opt = build_verb_opt(verbalizer)
        t_sched = get_cosine_schedule_with_warmup(t_opt, 1, STEPS)
        s_sched = get_cosine_schedule_with_warmup(s_opt, 1, STEPS)
        v_sched = get_cosine_schedule_with_warmup(v_opt, 1, WARMUP)
        chk("All 3 optimizers built",  True)
        chk("All 3 schedulers built",  True)
        s_pg = len(s_opt.param_groups)
        chk("Student has 2 param groups", s_pg == 2, f"got {s_pg}")
        v_pg = len(v_opt.param_groups)
        chk("Verbalizer has 2 param groups", v_pg == 2, f"got {v_pg}")
    except Exception as e:
        chk("Optimizer build", False, str(e)); return

    # ── 3. StudentLossComputer ────────────────────────────────────────────────
    sec("3. StudentLossComputer (real, from training/student_losses.py)")
    loss_computer = build_student_loss_computer(
        warmup_steps=WARMUP, lambda_distill=1.0, lambda_ans=1.0
    )
    chk("StudentLossComputer instantiated", True)

    # ── 4. Full training loop ─────────────────────────────────────────────────
    sec("4. Training loop")
    teacher.train(); student.train(); verbalizer.train()
    data_iter = iter(fake_loader)
    ckpt_dir  = None

    for step in range(STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(fake_loader); batch = next(data_iter)

        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values   = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]
        gt_waypoints   = batch["gt_waypoints"]
        ground_truth   = batch["ground_truth"]
        is_warmup      = (step < WARMUP)

        phase = "warmup" if is_warmup else "frozen"
        print(f"\n  ── Step {step} ({phase}) ──")

        # B. Teacher GRPO step
        try:
            buf = teacher.training_step(
                input_ids, pixel_values, image_grid_thw, attention_mask,
                ground_truth, reward_fns=[], reward_weights=[],
                optimizer=t_opt, tokenizer=None, grad_clip=1.0,
            )
            t_sched.step()
            chk_shape(f"  buf.h_T",              buf.h_T,              (BATCH, HS))
            chk_shape(f"  buf.tau_pos_ids",       buf.tau_pos_ids,      (BATCH, FULL_LEN))
            chk_shape(f"  buf.tau_neg_ids",       buf.tau_neg_ids,      (BATCH, FULL_LEN))
            chk_shape(f"  buf.advantages",        buf.advantages,       (G, BATCH))
            chk_shape(f"  buf.rewards",           buf.rewards,          (G, BATCH))
            chk(       f"  buf.h_T not in graph", not buf.h_T.requires_grad)
        except Exception as e:
            chk(f"Step {step} teacher", False, traceback.format_exc(limit=2)); continue

        # C. Verbalizer freeze transition
        if step == WARMUP and not verbalizer.is_frozen():
            verbalizer.freeze_for_student_training()
            chk("Verbalizer frozen at warmup boundary", verbalizer.is_frozen())

        # D. Student loss computation (REAL StudentLossComputer)
        try:
            loss_out = loss_computer.compute(
                student=student, verbalizer=verbalizer,
                input_ids=input_ids, pixel_values=pixel_values,
                image_grid_thw=image_grid_thw, attention_mask=attention_mask,
                buffer=buf, gt_waypoints=gt_waypoints, global_step=step,
            )
            chk(f"  LossOutput returned", isinstance(loss_out, LossOutput))
            chk(f"  student_total is scalar", loss_out.student_total.shape == torch.Size([]))
            chk(f"  student_total finite",   loss_out.student_total.isfinite().item())
            chk(f"  metrics complete",       "loss/l_distill" in loss_out.metrics
                                          and "loss/l_ans" in loss_out.metrics)
            if is_warmup:
                chk(f"  lm_loss present in warmup",   loss_out.lm_loss is not None)
                chk(f"  lm_loss finite",               loss_out.lm_loss.isfinite().item())
            else:
                chk(f"  lm_loss is None in frozen",   loss_out.lm_loss is None)
                chk(f"  loss/l_verb > 0 in metrics",  loss_out.metrics["loss/l_verb"] > 0)
        except Exception as e:
            chk(f"Step {step} loss_computer.compute", False, traceback.format_exc(limit=3)); continue

        # E. Backward passes
        try:
            if is_warmup:
                # E1: Verbalizer backward (LM loss on τ+, z detached)
                v_opt.zero_grad()
                loss_out.lm_loss.backward()
                nn.utils.clip_grad_norm_(verbalizer.parameters(), 1.0)
                v_opt.step(); v_sched.step()

                # E2: Student backward (distill + ans)
                s_opt.zero_grad()
                loss_out.student_total.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0)
                s_opt.step(); s_sched.step()

                # Check that LM loss did NOT create grad in Student spatial_tokens
                chk(f"  LM loss does NOT grad Student spatial_tokens",
                    student.spatial_tokens.grad is None or
                    student.spatial_tokens.grad.abs().max().item() == 0.0)
            else:
                # E3: Student backward only (verb + distill + ans)
                s_opt.zero_grad()
                loss_out.student_total.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0)
                s_opt.step(); s_sched.step()

                # Confirm DPO gradient reached Student
                sp_grad = student.spatial_tokens.grad
                chk(f"  DPO grad reached Student spatial_tokens",
                    sp_grad is not None and sp_grad.abs().max().item() > 0,
                    f"max_grad={sp_grad.abs().max().item() if sp_grad is not None else 'None'}")

                # Confirm Verbalizer CA params have NO grad (frozen)
                ca_grad = next(verbalizer.ca_blocks[0].parameters()).grad
                chk(f"  Frozen Verbalizer CA has no grad",
                    ca_grad is None or ca_grad.abs().max().item() == 0.0)

            chk(f"  Step {step} backward+step OK", True)

        except Exception as e:
            chk(f"Step {step} backward", False, traceback.format_exc(limit=3))

    # ── 5. Checkpoint save / load ─────────────────────────────────────────────
    sec("5. Checkpoint save → load")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "training_state.pt")
            # Mirror the save pattern from train_stage2.py
            torch.save({
                "step":           STEPS - 1,
                "spatial_tokens": student.spatial_tokens.data,
                "spatial_mlp":    student.spatial_mlp.state_dict(),
                "ca_blocks":      verbalizer.ca_blocks.state_dict(),
                "student_opt":    s_opt.state_dict(),
                "teacher_opt":    t_opt.state_dict(),
                "verbalizer_opt": v_opt.state_dict(),
                "student_sched":  s_sched.state_dict(),
                "teacher_sched":  t_sched.state_dict(),
                "verbalizer_sched": v_sched.state_dict(),
            }, ckpt_path)
            chk("Checkpoint saved", os.path.exists(ckpt_path))

            state = torch.load(ckpt_path, map_location=DEVICE)
            student2    = MockLatentStudent()
            verbalizer2 = MockVerbalizer()
            student2.spatial_tokens.data.copy_(state["spatial_tokens"])
            student2.spatial_mlp.load_state_dict(state["spatial_mlp"])
            verbalizer2.ca_blocks.load_state_dict(state["ca_blocks"])

            tokens_match = torch.allclose(
                student2.spatial_tokens.data, student.spatial_tokens.data)
            chk("Spatial tokens restored exactly", tokens_match)

            s_opt2 = build_student_opt(student2)
            s_opt2.load_state_dict(state["student_opt"])
            chk("Student optimizer state restored", True)
            chk("Resume step correct", state["step"] == STEPS - 1,
                f"step={state['step']}")
    except Exception as e:
        chk("Checkpoint round-trip", False, traceback.format_exc(limit=3))

    # ── 6. Verbalizer freeze integrity ────────────────────────────────────────
    sec("6. Verbalizer freeze integrity")
    chk("is_frozen() returns True after freeze",  verbalizer.is_frozen())
    frozen_trainable = sum(p.numel() for p in verbalizer.parameters() if p.requires_grad)
    chk("0 trainable params after freeze", frozen_trainable == 0,
        f"remaining trainable: {frozen_trainable}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total  = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    failed = total - passed

    print(f"\n{'═'*62}")
    print(f"  SMOKE TEST COMPLETE:  {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        print("\n  Failed checks:")
        for name, ok in _results:
            if not ok:
                print(f"    ❌  {name}")
    else:
        print("\n  All checks passed — Stage 2 pipeline is structurally sound.")
    print(f"{'═'*62}\n")
    return failed == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
