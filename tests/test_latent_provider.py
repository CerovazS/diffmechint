"""Unit tests for the Phase 4.19 latent → SAE-token adapter."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from diffmechint.sae import (
    first_latent_d_in,
    latent_hdf5_provider,
    load_latent_val_tokens,
    patchify_latents,
)


def _write_latent_shard(path, n=8, c=4, h=8, w=8, seed=0):
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((n, c, h, w)).astype(np.float16)
    labels = rng.integers(0, 1000, size=(n,)).astype(np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("latents", data=latents)
        f.create_dataset("labels", data=labels)
    return latents


def test_patchify_matches_conv2d_token_order():
    # PatchEmbed = Conv2d(stride=p, kernel=p); with identity-like weights each
    # output channel reads one (c, ph, pw) slot, in that flattening order.
    n, c, h, w, p = 2, 3, 4, 4, 2
    x = torch.randn(n, c, h, w)
    d = c * p * p
    weight = torch.zeros(d, c, p, p)
    for i in range(d):
        ci, rem = divmod(i, p * p)
        ph, pw = divmod(rem, p)
        weight[i, ci, ph, pw] = 1.0
    conv = torch.nn.functional.conv2d(x, weight, stride=p)  # (N, D, H/p, W/p)
    ref = conv.flatten(2).transpose(1, 2)  # (N, T, D) row-major tokens
    out = patchify_latents(x, p)
    assert out.shape == (n, (h // p) * (w // p), d)
    assert torch.allclose(out, ref)


def test_patchify_rejects_indivisible_dims():
    with pytest.raises(ValueError):
        patchify_latents(torch.randn(1, 4, 9, 8), 2)


def test_provider_yields_flat_batches_and_terminates(tmp_path):
    shard = tmp_path / "00000.h5"
    _write_latent_shard(shard, n=8, c=4, h=8, w=8)
    # 8 images * 16 tokens = 128 tokens; batch 32 → 4 batches, drop-last ok.
    batches = list(
        latent_hdf5_provider(
            shard, batch_size=32, patch_size=2, device="cpu", loop_forever=False, seed=0
        )
    )
    assert len(batches) == 4
    assert all(b.shape == (32, 16) for b in batches)


def test_provider_is_seed_deterministic(tmp_path):
    shard = tmp_path / "00000.h5"
    _write_latent_shard(shard, n=8, c=4, h=8, w=8)

    def first_batch(seed):
        it = latent_hdf5_provider(
            shard, batch_size=16, patch_size=2, device="cpu", loop_forever=False, seed=seed
        )
        return next(iter(it))

    assert torch.equal(first_batch(0), first_batch(0))
    assert not torch.equal(first_batch(0), first_batch(1))


def test_load_latent_val_tokens_deterministic_prefix(tmp_path):
    shard = tmp_path / "val.h5"
    latents = _write_latent_shard(shard, n=4, c=4, h=8, w=8)
    t_full = load_latent_val_tokens(shard, patch_size=2)
    assert t_full.shape == (4 * 16, 16)
    t_cut = load_latent_val_tokens(shard, patch_size=2, max_tokens=10)
    assert t_cut.shape == (10, 16)
    assert torch.equal(t_full[:10], t_cut)
    # Spot-check against direct patchify of the stored array.
    ref = patchify_latents(torch.from_numpy(latents).float(), 2).reshape(-1, 16)
    assert torch.allclose(t_full, ref)


def test_first_latent_d_in(tmp_path):
    shard = tmp_path / "00000.h5"
    _write_latent_shard(shard, n=2, c=4, h=8, w=8)
    assert first_latent_d_in(shard, patch_size=2) == 16
    assert first_latent_d_in(shard, patch_size=1) == 4
