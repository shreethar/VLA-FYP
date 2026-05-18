"""
test_gpu_smoke.py
-----------------
GPU smoke tests — run ONLY on the rented GPU machine.
These actually load the real models and verify end-to-end shapes.

Run:  python test/test_gpu_smoke.py

This script:
  1. Loads Qwen3.5-4B with LoRA (LatentStudent)
  2. Runs the latent loop → checks output shapes
  3. Loads the Verbalizer (Qwen3.5-0.8B + CA blocks)
  4. Tests forward pass with latents
  5. Tests freeze + gradient flow through CA
  6. Loads VGGT extractor → tests spatial forcing
  7. Estimates peak VRAM usage

Expected VRAM: ~18-20 GB for all three models simultaneously.
Safe to run on a 24 GB card (RTX 5090 / A100 / etc.)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import time
import gc


def fmt_mem(bytes_val):
    return f"{bytes_val / 1024**3:.2f} GB"


def check_cuda():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires a GPU.")
        print("   Run the CPU tests instead: python -m pytest test/ -v -k 'not gpu'")
        sys.exit(1)
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   Total VRAM: {fmt_mem(torch.cuda.get_device_properties(0).total_mem)}")
    print()


def test_latent_student():
    print("=" * 60)
    print("TEST 1: LatentStudent — Latent Loop + Spatial Waypoints")
    print("=" * 60)

    from models.latent_student import LatentStudent

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    student = LatentStudent(
        model_name="Qwen/Qwen3.5-4B",
        M=6, K=5, lora_rank=64, lora_alpha=128,
    ).to(device)
    print(f"  ✅ Loaded in {time.time()-t0:.1f}s, VRAM: {fmt_mem(torch.cuda.memory_allocated())}")
    student.print_trainable_parameters()

    # Dummy inputs (no real images — just text tokens)
    batch = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)

    # Test generate_latents
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        latents, spatial_hidden, waypoints = student.generate_latents(
            input_ids=input_ids,
            pixel_values=None,
            image_grid_thw=None,
        )

    print(f"  ✅ generate_latents succeeded")
    print(f"     latents: {len(latents)} × [{latents[0].shape}]")
    print(f"     spatial_hidden: {spatial_hidden.shape}")
    print(f"     waypoints: {waypoints.shape}")
    assert len(latents) == 6, f"Expected M=6 latents, got {len(latents)}"
    assert latents[0].shape == (batch, student.hidden_dim)
    assert spatial_hidden.shape == (batch, 5, student.hidden_dim)
    assert waypoints.shape == (batch, 5, 2)
    assert waypoints.min() >= 0.0 and waypoints.max() <= 1.0

    # Test get_answer_hidden_state
    answer_positions = torch.tensor([10, 15], device=device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        h_ans = student.get_answer_hidden_state(
            input_ids=input_ids,
            pixel_values=None,
            image_grid_thw=None,
            answer_token_positions=answer_positions,
        )
    print(f"  ✅ get_answer_hidden_state: {h_ans.shape}")
    assert h_ans.shape == (batch, student.hidden_dim)

    peak = torch.cuda.max_memory_allocated()
    print(f"  📊 Peak VRAM after Student: {fmt_mem(peak)}")

    # Clean up
    del student
    gc.collect()
    torch.cuda.empty_cache()
    print()
    return True


def test_verbalizer():
    print("=" * 60)
    print("TEST 2: Verbalizer — CA Injection + Freeze + Gradient Flow")
    print("=" * 60)

    from models.verbalizer import Verbalizer

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    verbalizer = Verbalizer(
        model_name="Qwen/Qwen3.5-0.8B",
        student_hidden=2560,  # must match Student's hidden_dim (Qwen3.5-4B)
        lora_rank=32,
        lora_alpha=64,
    ).to(device)
    print(f"  ✅ Loaded in {time.time()-t0:.1f}s, VRAM: {fmt_mem(torch.cuda.memory_allocated())}")
    verbalizer.print_trainable_parameters()

    batch, seq, M = 2, 20, 6
    d_student = 2560
    vocab_size = verbalizer.lm.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch, seq), device=device)
    attn_mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    latents = torch.randn(batch, M, d_student, device=device, dtype=torch.bfloat16,
                           requires_grad=True)
    labels = input_ids.clone()
    labels[:, :5] = -100

    # Test LM loss (warm-up mode)
    lm_loss = verbalizer.compute_lm_loss(input_ids, attn_mask, latents.detach(), labels)
    print(f"  ✅ LM loss (warm-up): {lm_loss.item():.4f}")
    assert lm_loss.item() > 0

    # Test DPO loss
    response_mask = torch.zeros(batch, seq, device=device)
    response_mask[:, 5:] = 1.0
    dpo_loss, metrics = verbalizer.compute_dpo_loss(
        pos_input_ids=input_ids,
        neg_input_ids=input_ids,
        pos_attention_mask=attn_mask,
        neg_attention_mask=attn_mask,
        latents=latents,
        pos_response_mask=response_mask,
        neg_response_mask=response_mask,
    )
    print(f"  ✅ DPO loss: {dpo_loss.item():.4f}, metrics: {metrics}")

    # Test freeze + gradient flow
    verbalizer.freeze_for_student_training()
    print(f"  ✅ Frozen: {verbalizer.is_frozen()}")

    latents2 = torch.randn(batch, M, d_student, device=device, dtype=torch.bfloat16,
                            requires_grad=True)
    dpo_loss2, _ = verbalizer.compute_dpo_loss(
        pos_input_ids=input_ids,
        neg_input_ids=input_ids,
        pos_attention_mask=attn_mask,
        neg_attention_mask=attn_mask,
        latents=latents2,
        pos_response_mask=response_mask,
        neg_response_mask=response_mask,
    )
    dpo_loss2.backward()
    assert latents2.grad is not None, "CRITICAL: Gradient did not flow through frozen Verbalizer!"
    assert latents2.grad.abs().sum() > 0, "CRITICAL: Gradient is all zeros!"
    print(f"  ✅ Gradient flows through frozen Verbalizer → latents ✓")
    print(f"     latents.grad norm: {latents2.grad.norm().item():.6f}")

    peak = torch.cuda.max_memory_allocated()
    print(f"  📊 Peak VRAM after Verbalizer: {fmt_mem(peak)}")

    del verbalizer
    gc.collect()
    torch.cuda.empty_cache()
    print()
    return True


def test_spatial_forcing():
    print("=" * 60)
    print("TEST 3: Spatial Forcing — DINOv2 Extractor + ProjectionMLP")
    print("=" * 60)

    from models.spatial_forcing import SpatialForcingLoss

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    sf = SpatialForcingLoss(
        extractor_type="vggt",
        extractor_ckpt="facebook/VGGT-1B",
        student_dim=2560,
        extractor_dim=1024,
        lambda_sf=0.1,
    ).to(device)
    print(f"  ✅ Loaded in {time.time()-t0:.1f}s, VRAM: {fmt_mem(torch.cuda.memory_allocated())}")
    sf.print_trainable_parameters()

    batch = 2
    # DINOv2 expects 224x224 images with 3 channels
    pixel_values = torch.randn(batch, 3, 224, 224, device=device, dtype=torch.bfloat16)

    # Extract reference features
    ref_feats = sf.extract_reference_features(pixel_values)
    print(f"  ✅ Reference features: {ref_feats.shape}, norm: {ref_feats.norm(dim=-1)}")
    assert ref_feats.shape == (batch, 1024)

    # Compute loss
    x_V = torch.randn(batch, 50, 2560, device=device)  # student visual features
    loss = sf.compute_loss(x_V, ref_feats)
    print(f"  ✅ Spatial forcing loss: {loss.item():.4f}")
    assert loss.shape == ()

    # Verify only MLP trains
    for p in sf.extractor.parameters():
        assert not p.requires_grad, "Extractor params should be frozen!"
    mlp_trainable = sum(p.requires_grad for p in sf.proj_mlp.parameters())
    assert mlp_trainable > 0, "ProjectionMLP should be trainable!"
    print(f"  ✅ Extractor frozen, ProjectionMLP trainable")

    peak = torch.cuda.max_memory_allocated()
    print(f"  📊 Peak VRAM after SpatialForcing: {fmt_mem(peak)}")

    del sf
    gc.collect()
    torch.cuda.empty_cache()
    print()
    return True


def test_grpo_teacher():
    print("=" * 60)
    print("TEST 4: GRPOTeacher — Initialization + Advantage Computation")
    print("=" * 60)

    from training.grpo_teacher import GRPOTeacher

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    teacher = GRPOTeacher(
        model_name="Qwen/Qwen3.5-4B",
        G=5,
        answer_token_id=99999,  # dummy
        lora_rank=64,
        lora_alpha=128,
    ).to(device)
    print(f"  ✅ Loaded in {time.time()-t0:.1f}s, VRAM: {fmt_mem(torch.cuda.memory_allocated())}")
    teacher.print_trainable_parameters()

    # Test advantage computation (CPU-based, but verify in GPU context)
    rewards = torch.rand(5, 4, device=device)
    advantages = GRPOTeacher.compute_advantages(rewards)
    assert advantages.shape == (5, 4)
    for b in range(4):
        assert abs(advantages[:, b].mean().item()) < 1e-4

    # Test _find_answer_positions with dummy token
    teacher.answer_token_id = 42
    token_ids = torch.tensor([
        [1, 2, 3, 42, 5, 6],
        [1, 42, 3, 4, 5, 6],
    ], device=device)
    positions = teacher._find_answer_positions(token_ids)
    assert positions[0] == 3
    assert positions[1] == 1
    print(f"  ✅ _find_answer_positions correct: {positions.tolist()}")

    # Test h_T extraction shape (just forward pass, no rollouts needed)
    batch, seq = 2, 20
    tau_pos_ids = torch.randint(0, 1000, (batch, seq), device=device)
    tau_pos_mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    answer_pos = torch.tensor([5, 10], device=device)

    h_T = teacher.extract_answer_hidden_state(
        tau_pos_ids, tau_pos_mask,
        pixel_values=None, image_grid_thw=None,
        answer_token_pos=answer_pos,
    )
    print(f"  ✅ h_T shape: {h_T.shape}")
    assert h_T.shape == (batch, teacher.hidden_dim)

    peak = torch.cuda.max_memory_allocated()
    print(f"  📊 Peak VRAM after Teacher: {fmt_mem(peak)}")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    print()
    return True


def test_tokenizer_setup():
    print("=" * 60)
    print("TEST 5: Tokenizer Setup — <ans>/<ans> Registration")
    print("=" * 60)

    from tokenizer_setup import setup_tokenizer

    save_dir = os.path.join(os.path.dirname(__file__), "..", "_test_tokenizer_tmp")
    try:
        tokenizer, answer_token_id = setup_tokenizer(
            model_name="Qwen/Qwen3.5-4B",
            save_dir=save_dir,
        )
        print(f"  ✅ answer_token_id = {answer_token_id}")
        print(f"  ✅ Vocab size = {len(tokenizer)}")

        # Verify <ans> is a single token
        test_ids = tokenizer.encode("<ans>", add_special_tokens=False)
        assert len(test_ids) == 1, f"<ans> tokenised to {len(test_ids)} tokens!"
        assert test_ids[0] == answer_token_id
        print(f"  ✅ <ans> tokenises to single ID: {test_ids}")

        # Verify load_answer_token_id works
        from tokenizer_setup import load_answer_token_id
        loaded_id = load_answer_token_id(save_dir)
        assert loaded_id == answer_token_id
        print(f"  ✅ load_answer_token_id roundtrip OK")

    finally:
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)
    print()
    return True


def main():
    check_cuda()

    results = {}
    tests = [
        ("Tokenizer Setup", test_tokenizer_setup),
        ("Latent Student", test_latent_student),
        ("Verbalizer", test_verbalizer),
        ("Spatial Forcing", test_spatial_forcing),
        ("GRPO Teacher", test_grpo_teacher),
    ]

    for name, fn in tests:
        try:
            passed = fn()
            results[name] = "✅ PASSED" if passed else "❌ FAILED"
        except Exception as e:
            results[name] = f"❌ ERROR: {e}"
            import traceback
            traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:25s} {status}")

    total_failed = sum(1 for v in results.values() if "❌" in v)
    if total_failed == 0:
        print(f"\n🎉 All {len(tests)} GPU smoke tests passed!")
    else:
        print(f"\n⚠️  {total_failed}/{len(tests)} tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
