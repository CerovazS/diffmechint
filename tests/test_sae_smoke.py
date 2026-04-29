"""Phase 4 smoke + Phase 3↔Phase 4 integration tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from diffmechint.hooks import (
    ActivationBuffer,
    ResidualStreamTap,
    default_tap_layers,
    timestep_context,
)
from diffmechint.sae import (
    build_sae,
    evaluate_sae,
    hdf5_provider,
    synthetic_provider,
    train_sae,
)
from diffmechint.sit import SiT_models


# ---------------------------------------------------------------------------
# Data provider unit tests
# ---------------------------------------------------------------------------
def test_synthetic_provider_shape_and_finiteness() -> None:
    p = synthetic_provider(
        d_in=32, batch_size=64, n_features_dict=128, k_sparse=4, device="cpu", seed=1
    )
    batch = next(p)
    assert batch.shape == (64, 32)
    assert torch.isfinite(batch).all()


def test_hdf5_provider_round_trip(tmp_path: Path) -> None:
    """Write a 2-shard HDF5 buffer, then read it back via the provider.

    drop_last is per-shard so picks counts that divide cleanly:
    shard A: 4*16=64 → 1 batch; shard B: 8*16=128 → 2 batches; total 3.
    """
    a1 = np.random.randn(4, 16, 32).astype(np.float16)
    a2 = np.random.randn(8, 16, 32).astype(np.float16)
    for name, arr in (("0_0.h5", a1), ("0_1.h5", a2)):
        with h5py.File(tmp_path / name, "w") as f:
            f.create_dataset("activations", data=arr)
    p = hdf5_provider(tmp_path, batch_size=64, device="cpu", loop_forever=False)
    batches = list(p)
    assert len(batches) == 3
    for b in batches:
        assert b.shape == (64, 32)


def test_hdf5_provider_skips_partial_batch(tmp_path: Path) -> None:
    arr = np.random.randn(3, 4, 8).astype(np.float16)  # 12 tokens
    with h5py.File(tmp_path / "x.h5", "w") as f:
        f.create_dataset("activations", data=arr)
    p = hdf5_provider(tmp_path, batch_size=10, device="cpu", loop_forever=False)
    batches = list(p)
    # 12 tokens, batch 10 → only 1 full batch (12//10 = 1, the leftover 2 dropped)
    assert len(batches) == 1
    assert batches[0].shape == (10, 8)


# ---------------------------------------------------------------------------
# build_sae factory
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["topk", "batch_topk"])
def test_build_sae_variants_instantiate_on_cpu(variant: str) -> None:
    sae = build_sae(d_in=64, d_sae=128, k=4, variant=variant, device="cpu")
    assert sae.cfg.d_in == 64
    assert sae.cfg.d_sae == 128
    assert sae.cfg.k == 4
    n_params = sum(p.numel() for p in sae.parameters())
    # encoder + decoder + biases ~= 2 * d_in * d_sae + 2 * (d_sae or d_in)
    assert n_params > 64 * 128


def test_build_sae_matryoshka_requires_groups() -> None:
    with pytest.raises(ValueError, match="matryoshka_group_sizes"):
        build_sae(d_in=64, d_sae=128, k=4, variant="matryoshka", device="cpu")


def test_build_sae_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="Unknown SAE variant"):
        build_sae(d_in=64, d_sae=128, k=4, variant="badname", device="cpu")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Training smoke — loss must decrease
# ---------------------------------------------------------------------------
def _train_capturing_loss(sae, provider, total_samples: int, batch: int, lr: float) -> list[float]:
    """Hijack sae.training_forward_pass to record losses across the run."""
    losses: list[float] = []
    orig = sae.training_forward_pass
    def spy(*args, **kw):
        out = orig(*args, **kw)
        # SAELens TrainingSAEOutput has `.loss` (scalar tensor).
        losses.append(float(out.loss.detach().mean().item()))
        return out
    sae.training_forward_pass = spy  # type: ignore[assignment]
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            train_sae(
                sae, provider,
                out_dir=Path(tmp),
                total_training_samples=total_samples,
                train_batch_size_samples=batch,
                lr=lr,
                lr_warm_up_steps=10,
                n_checkpoints=0,
                save_final_checkpoint=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
                log_to_wandb=False,
            )
    finally:
        sae.training_forward_pass = orig  # restore
    return losses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only smoke")
def test_phase4_smoke_loss_decreases_on_synthetic() -> None:
    """Train tiny SAE on synthetic K-sparse data; final 10% mean loss < first 10% mean loss."""
    torch.manual_seed(0)
    sae = build_sae(
        d_in=64, d_sae=256, k=4, variant="topk", device="cuda", normalize_activations="none"
    )
    provider = synthetic_provider(
        d_in=64, batch_size=256, n_features_dict=256, k_sparse=4, device="cuda", seed=1
    )
    losses = _train_capturing_loss(sae, provider, total_samples=256_000, batch=256, lr=1e-3)
    assert len(losses) > 50
    n = len(losses)
    initial = sum(losses[: max(1, n // 10)]) / max(1, n // 10)
    final = sum(losses[-max(1, n // 10) :]) / max(1, n // 10)
    assert final < initial, f"loss did not decrease: initial={initial:.4f}, final={final:.4f}"
    # Sanity: dropped at least 25%.
    assert final < 0.75 * initial, f"loss dropped <25%: initial={initial:.4f}, final={final:.4f}"


# ---------------------------------------------------------------------------
# Phase 3 ↔ Phase 4 end-to-end integration
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU integration smoke")
def test_phase3_phase4_e2e(tmp_path: Path) -> None:
    """Pipe SiT-B/2 hook activations → HDF5 → SAE training → eval."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    # Phase 3 — extract activations from a SiT-B/2 forward.
    model = SiT_models["SiT-B/2"](input_size=32, in_channels=4, num_classes=1000).to(device)
    model.eval()
    shard_dir = tmp_path / "acts"
    buf = ActivationBuffer(shard_dir=shard_dir)
    with ResidualStreamTap(model, default_tap_layers(len(model.blocks)), buf):
        for t in (0.025, 0.20, 0.50):
            x = torch.randn(8, 4, 32, 32, device=device)
            y = torch.randint(0, 1000, (8,), device=device)
            with timestep_context(t), torch.no_grad():
                model(x, torch.full((8,), t, device=device), y)
    buf.flush_all()
    cells = sorted(shard_dir.glob("*.h5"))
    assert len(cells) == 9  # 3 layers × 3 t-bins

    # Pick the (layer-50%, t=0.20) shard — bin index 1.
    canonical = next(p for p in cells if p.name.endswith("_1.h5") and "6_" in p.name)
    assert canonical.exists()

    # Phase 4 — train a tiny SAE on the canonical cell.
    sae = build_sae(
        d_in=768, d_sae=2048, k=16, variant="topk", device="cuda", normalize_activations="none"
    )
    provider = hdf5_provider(canonical, batch_size=128, device="cuda", seed=0)
    train_sae(
        sae,
        provider,
        out_dir=tmp_path / "sae",
        total_training_samples=64_000,
        train_batch_size_samples=128,
        lr=1e-3,
        lr_warm_up_steps=10,
        n_checkpoints=0,
        save_final_checkpoint=False,
        device="cuda",
        log_to_wandb=False,
    )

    # Eval on a fresh draw from the same shard.
    eval_provider = hdf5_provider(canonical, batch_size=128, device="cuda", seed=99)
    metrics = evaluate_sae(sae, eval_provider, n_batches=4)
    assert metrics["recon_cosine"] > 0.0  # at least learned something
    assert 0.0 <= metrics["density"] <= 1.0
    assert metrics["live_features"] > 0
