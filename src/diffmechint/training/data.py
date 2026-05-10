"""Datamodules for SiT training — synthetic (smoke) + HDF5-cached real latents."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Synthetic — for smoke / unit tests. Standard normal latents with random labels.
# ---------------------------------------------------------------------------
class SyntheticLatentDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        latent_shape: tuple[int, int, int],
        num_classes: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.latent_shape = latent_shape
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        gen = torch.Generator().manual_seed(self.seed + idx)
        z = torch.randn(self.latent_shape, generator=gen)
        y = torch.randint(0, self.num_classes, (1,), generator=gen).item()
        return {"latent": z, "label": int(y)}


class SyntheticLatentDataModule(L.LightningDataModule):
    """For 1k-step smoke runs without real data."""

    def __init__(
        self,
        latent_shape: tuple[int, int, int],
        num_classes: int = 1000,
        n_samples: int = 8192,
        batch_size: int = 32,
        num_workers: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        self.train_set = SyntheticLatentDataset(
            self.hparams.n_samples,
            tuple(self.hparams.latent_shape),
            self.hparams.num_classes,
            self.hparams.seed,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )


# ---------------------------------------------------------------------------
# Cached — read pre-encoded latents shard-by-shard from HDF5.
# Output of `precompute_latents.py`. Each shard has datasets `latents` and `labels`.
# ---------------------------------------------------------------------------
class CachedLatentDataset(Dataset):
    """Memory-mapped read across an arbitrary number of HDF5 shards.

    If `normalize=True` (default) and `stats.json` is present in `shard_dir`,
    applies per-channel z-score `(z - mean) / std` on read. Without this the
    DiT is trained on tokenizer-specific scales (esp. dc_ae_1_0 with
    scaling_factor=1.0) and matched-compute comparisons across conditions
    are no longer apples-to-apples.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        normalize: bool = True,
        holdout_fraction: float = 0.0,
        holdout_seed: int = 42,
        is_val: bool = False,
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        shards = sorted(glob.glob(str(self.shard_dir / "*.h5")))
        if not shards:
            raise FileNotFoundError(f"No *.h5 shards found in {self.shard_dir}")
        # Build cumulative index: (shard_path, offset_within_shard).
        self.shards: list[str] = shards
        self.lengths: list[int] = []
        for s in shards:
            with h5py.File(s, "r") as f:
                self.lengths.append(int(f["latents"].shape[0]))
        self.cumulative = [0]
        for n in self.lengths:
            self.cumulative.append(self.cumulative[-1] + n)
        self._handles: dict[str, h5py.File] = {}

        # Deterministic holdout split. Same seed across (train, val) instances yields
        # complementary index sets. Letting the train side keep all indices when
        # holdout_fraction=0 preserves the original behavior bit-for-bit.
        n_total = self.cumulative[-1]
        if not 0.0 <= holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}")
        self.holdout_fraction = holdout_fraction
        self.is_val = is_val
        if holdout_fraction == 0.0:
            self._index_map: np.ndarray | None = None  # use raw idx
        else:
            n_holdout = int(round(n_total * holdout_fraction))
            rng = np.random.default_rng(holdout_seed)
            holdout_idx = rng.choice(n_total, size=n_holdout, replace=False)
            holdout_set = np.zeros(n_total, dtype=bool)
            holdout_set[holdout_idx] = True
            mask = holdout_set if is_val else ~holdout_set
            self._index_map = np.flatnonzero(mask)

        # Per-feature normalization tensors. Broadcast shape is derived from the
        # `feature_axis` recorded in stats.json so the same code path handles
        # spatial (B,C,H,W) and sequence (B,T,D) layouts uniformly.
        self.normalize = normalize
        self.stats: dict | None = None
        self._mean: Tensor | None = None
        self._inv_std: Tensor | None = None
        if normalize:
            stats_path = self.shard_dir / "stats.json"
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"normalize=True requires {stats_path}; rerun precompute_latents.py "
                    f"with the current code, or pass normalize=False explicitly."
                )
            self.stats = json.loads(stats_path.read_text())
            mean = np.asarray(self.stats["per_feature_mean"], dtype=np.float32)
            std = np.asarray(self.stats["per_feature_std"], dtype=np.float32)
            # Guard against zero-variance features (would create NaNs).
            std = np.where(std < 1e-6, 1.0, std)
            # Build the per-sample broadcast shape (no batch dim here): set the
            # feature axis to -1 (full size) and 1 everywhere else.
            n_dims = len(self.stats["latent_shape"])
            feat_axis_batched = self.stats["feature_axis"]
            feat_axis_sample = (
                feat_axis_batched - 1 if feat_axis_batched >= 0 else feat_axis_batched
            )
            broadcast_shape = [1] * n_dims
            broadcast_shape[feat_axis_sample] = -1
            self._mean = torch.from_numpy(mean).view(*broadcast_shape)
            self._inv_std = torch.from_numpy(1.0 / std).view(*broadcast_shape)

    def __len__(self) -> int:
        if self._index_map is not None:
            return int(self._index_map.shape[0])
        return self.cumulative[-1]

    def _open(self, path: str) -> h5py.File:
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, "r", swmr=True)
            self._handles[path] = h
        return h

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        # Translate local idx → global (only when a holdout split is active).
        if self._index_map is not None:
            idx = int(self._index_map[idx])
        # Locate which shard owns idx.
        shard_idx = max(i for i, c in enumerate(self.cumulative) if c <= idx)
        offset = idx - self.cumulative[shard_idx]
        f = self._open(self.shards[shard_idx])
        z = torch.from_numpy(f["latents"][offset]).float()
        if self._mean is not None:
            z = (z - self._mean) * self._inv_std
        y = int(f["labels"][offset])
        return {"latent": z, "label": y}


class CachedLatentDataModule(L.LightningDataModule):
    """For real training: reads HDF5 shards produced by `precompute_latents.py`.

    With `holdout_fraction>0` a deterministic seeded split carves a held-out
    partition for `val_dataloader()` — kept inside the same shard_dir so we
    don't need a second precompute or an external ground-truth file.
    """

    def __init__(
        self,
        shard_dir: str,
        batch_size: int = 128,
        num_workers: int = 4,
        shuffle: bool = True,
        normalize: bool = True,
        holdout_fraction: float = 0.0,
        holdout_seed: int = 42,
        val_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        self.train_set = CachedLatentDataset(
            self.hparams.shard_dir,
            normalize=self.hparams.normalize,
            holdout_fraction=self.hparams.holdout_fraction,
            holdout_seed=self.hparams.holdout_seed,
            is_val=False,
        )
        if self.hparams.holdout_fraction > 0.0:
            self.val_set = CachedLatentDataset(
                self.hparams.shard_dir,
                normalize=self.hparams.normalize,
                holdout_fraction=self.hparams.holdout_fraction,
                holdout_seed=self.hparams.holdout_seed,
                is_val=True,
            )
        else:
            self.val_set = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_set is None:
            return None
        bs = self.hparams.val_batch_size or self.hparams.batch_size
        return DataLoader(
            self.val_set,
            batch_size=bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
