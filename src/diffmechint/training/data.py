"""Datamodules for SiT training — synthetic (smoke) + HDF5-cached real latents."""

from __future__ import annotations

import glob
from pathlib import Path

import h5py
import lightning as L
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
    """Memory-mapped read across an arbitrary number of HDF5 shards."""

    def __init__(self, shard_dir: str | Path) -> None:
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

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _open(self, path: str) -> h5py.File:
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, "r", swmr=True)
            self._handles[path] = h
        return h

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        # Locate which shard owns idx.
        shard_idx = max(i for i, c in enumerate(self.cumulative) if c <= idx)
        offset = idx - self.cumulative[shard_idx]
        f = self._open(self.shards[shard_idx])
        z = torch.from_numpy(f["latents"][offset]).float()
        y = int(f["labels"][offset])
        return {"latent": z, "label": y}


class CachedLatentDataModule(L.LightningDataModule):
    """For real training: reads HDF5 shards produced by `precompute_latents.py`."""

    def __init__(
        self,
        shard_dir: str,
        batch_size: int = 128,
        num_workers: int = 4,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        self.train_set = CachedLatentDataset(self.hparams.shard_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
