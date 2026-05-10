"""Verify per-feature z-score in CachedLatentDataset for spatial + sequence layouts."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from diffmechint.training.data import CachedLatentDataset
from diffmechint.training.precompute_latents import _RunningStats


def _write_shard(out_dir: Path, latents: np.ndarray, labels: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_dir / "00000.h5", "w") as f:
        f.create_dataset("latents", data=latents.astype(np.float16))
        f.create_dataset("labels", data=labels.astype(np.int32))


def _write_stats(out_dir: Path, latents: np.ndarray, feature_axis: int, kind: str) -> None:
    rs = _RunningStats(latents.shape[feature_axis], feature_axis=feature_axis)
    rs.update(latents.astype(np.float32))
    stats = {
        "format_version": "1",
        "adapter": "test",
        "hf_repo_id": None,
        "latent_shape": list(latents.shape[1:]),
        "kind": kind,
        "feature_axis": feature_axis,
        "feature_dim": latents.shape[feature_axis],
        "input_size": latents.shape[2] if kind == "spatial" else latents.shape[1],
        "scaling_factor": 1.0,
        "suggested_patch_size": 2 if kind == "spatial" else None,
        **rs.to_dict(),
        "images_written": int(latents.shape[0]),
        "wall_seconds": 0.0,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats))


def test_zscore_spatial_layout(tmp_path: Path) -> None:
    """Latents (N, C, H, W) — z-score by channel, broadcast over (H, W)."""
    rng = np.random.default_rng(0)
    N, C, H, W = 64, 4, 8, 8
    # Channel-specific shift+scale: μ_c = c, σ_c = c+1.
    base = rng.standard_normal((N, C, H, W)).astype(np.float32)
    mu = np.arange(C, dtype=np.float32).reshape(1, C, 1, 1)
    sigma = (np.arange(C, dtype=np.float32) + 1.0).reshape(1, C, 1, 1)
    latents = base * sigma + mu
    labels = rng.integers(0, 1000, size=N)

    _write_shard(tmp_path, latents, labels)
    _write_stats(tmp_path, latents, feature_axis=1, kind="spatial")

    ds = CachedLatentDataset(tmp_path, normalize=True)
    samples = torch.stack([ds[i]["latent"] for i in range(N)])  # (N, C, H, W)
    # After z-score, per-channel mean ≈ 0 and std ≈ 1 within fp16 precision.
    per_ch_mean = samples.mean(dim=(0, 2, 3))
    per_ch_std = samples.std(dim=(0, 2, 3))
    assert torch.allclose(per_ch_mean, torch.zeros(C), atol=1e-2), per_ch_mean
    assert torch.allclose(per_ch_std, torch.ones(C), atol=5e-2), per_ch_std


def test_zscore_sequence_layout(tmp_path: Path) -> None:
    """Latents (N, T, D) — z-score by feature D, broadcast over T (token axis)."""
    rng = np.random.default_rng(0)
    N, T, D = 64, 16, 8
    base = rng.standard_normal((N, T, D)).astype(np.float32)
    mu = np.arange(D, dtype=np.float32).reshape(1, 1, D)
    sigma = (np.arange(D, dtype=np.float32) + 1.0).reshape(1, 1, D)
    latents = base * sigma + mu
    labels = rng.integers(0, 1000, size=N)

    _write_shard(tmp_path, latents, labels)
    _write_stats(tmp_path, latents, feature_axis=-1, kind="sequence")

    ds = CachedLatentDataset(tmp_path, normalize=True)
    samples = torch.stack([ds[i]["latent"] for i in range(N)])  # (N, T, D)
    per_feat_mean = samples.mean(dim=(0, 1))
    per_feat_std = samples.std(dim=(0, 1))
    assert torch.allclose(per_feat_mean, torch.zeros(D), atol=1e-2), per_feat_mean
    assert torch.allclose(per_feat_std, torch.ones(D), atol=5e-2), per_feat_std


def test_normalize_off_returns_raw(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    latents = (rng.standard_normal((4, 4, 8, 8)) * 5.0 + 2.0).astype(np.float32)
    labels = rng.integers(0, 10, size=4)
    _write_shard(tmp_path, latents, labels)
    # No stats.json needed when normalize=False.
    ds = CachedLatentDataset(tmp_path, normalize=False)
    z = ds[0]["latent"]
    # Within fp16 quantization, raw read should match original.
    assert torch.allclose(z, torch.from_numpy(latents[0]).float(), atol=5e-3)


def test_normalize_missing_stats_raises(tmp_path: Path) -> None:
    latents = np.zeros((2, 4, 8, 8), dtype=np.float32)
    labels = np.zeros(2, dtype=np.int32)
    _write_shard(tmp_path, latents, labels)
    # No stats.json on disk.
    with pytest.raises(FileNotFoundError, match="stats.json"):
        CachedLatentDataset(tmp_path, normalize=True)
