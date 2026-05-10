"""Verify deterministic holdout split for CachedLatentDataset / DataModule."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from diffmechint.training.data import CachedLatentDataModule, CachedLatentDataset
from diffmechint.training.precompute_latents import _RunningStats


def _seed_shard(out_dir: Path, n: int = 200) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((n, 4, 8, 8)).astype(np.float32)
    labels = rng.integers(0, 1000, size=n).astype(np.int32)
    with h5py.File(out_dir / "00000.h5", "w") as f:
        f.create_dataset("latents", data=latents.astype(np.float16))
        f.create_dataset("labels", data=labels)
    rs = _RunningStats(4, feature_axis=1)
    rs.update(latents)
    stats = {
        "format_version": "1",
        "adapter": "test",
        "hf_repo_id": None,
        "latent_shape": [4, 8, 8],
        "kind": "spatial",
        "feature_axis": 1,
        "feature_dim": 4,
        "input_size": 8,
        "scaling_factor": 1.0,
        "suggested_patch_size": 2,
        **rs.to_dict(),
        "images_written": n,
        "wall_seconds": 0.0,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats))


def test_train_val_split_is_disjoint_and_complete(tmp_path: Path) -> None:
    """Train + val together must reconstruct the full set; no leak, no loss."""
    _seed_shard(tmp_path, n=200)
    train = CachedLatentDataset(tmp_path, normalize=False, holdout_fraction=0.1, holdout_seed=7, is_val=False)
    val = CachedLatentDataset(tmp_path, normalize=False, holdout_fraction=0.1, holdout_seed=7, is_val=True)
    assert len(train) == 180
    assert len(val) == 20
    train_idx = set(int(x) for x in train._index_map.tolist())
    val_idx = set(int(x) for x in val._index_map.tolist())
    assert train_idx.isdisjoint(val_idx)
    assert train_idx | val_idx == set(range(200))


def test_split_is_seed_deterministic(tmp_path: Path) -> None:
    _seed_shard(tmp_path, n=200)
    a = CachedLatentDataset(tmp_path, normalize=False, holdout_fraction=0.1, holdout_seed=42, is_val=True)
    b = CachedLatentDataset(tmp_path, normalize=False, holdout_fraction=0.1, holdout_seed=42, is_val=True)
    assert a._index_map.tolist() == b._index_map.tolist()
    c = CachedLatentDataset(tmp_path, normalize=False, holdout_fraction=0.1, holdout_seed=43, is_val=True)
    assert a._index_map.tolist() != c._index_map.tolist()


def test_zero_holdout_preserves_legacy_behavior(tmp_path: Path) -> None:
    _seed_shard(tmp_path, n=50)
    ds = CachedLatentDataset(tmp_path, normalize=False, holdout_fraction=0.0)
    assert ds._index_map is None
    assert len(ds) == 50
    item = ds[0]
    assert item["latent"].shape == (4, 8, 8)


def test_datamodule_val_dataloader_emits(tmp_path: Path) -> None:
    _seed_shard(tmp_path, n=200)
    dm = CachedLatentDataModule(
        shard_dir=str(tmp_path),
        batch_size=8,
        num_workers=0,
        normalize=False,
        holdout_fraction=0.1,
        holdout_seed=42,
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    assert val_loader is not None
    val_batches = list(val_loader)
    train_batches = list(train_loader)
    assert sum(b["latent"].shape[0] for b in train_batches) == 180
    assert sum(b["latent"].shape[0] for b in val_batches) == 20


def test_datamodule_no_holdout_returns_none(tmp_path: Path) -> None:
    _seed_shard(tmp_path, n=20)
    dm = CachedLatentDataModule(
        shard_dir=str(tmp_path), batch_size=4, num_workers=0, normalize=False
    )
    dm.setup()
    assert dm.val_dataloader() is None


def test_callbacks_importable() -> None:
    """Smoke: callbacks can be imported and constructed without GPU/model."""
    from diffmechint.training.callbacks import MiniFIDCallback, SampleCallback

    s = SampleCallback(every_n_steps=1000, n_samples=4, adapter_name="sd_vae", num_classes=1000)
    assert s.every_n_steps == 1000
    assert s.n_samples == 4

    f = MiniFIDCallback(every_n_steps=10000, n_samples=100, adapter_name="sd_vae", num_classes=1000)
    assert f.every_n_steps == 10000
    assert f.n_samples == 100
