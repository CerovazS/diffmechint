"""Activation buffer — accumulates (layer × t_bin) records, optional shard-to-disk."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch import Tensor


class ActivationBuffer:
    """Holds residual-stream activations keyed by `(layer_idx, t_bin)`.

    By default writes to RAM. When `shard_dir` is set, flushes records that
    exceed `max_records_per_cell` to HDF5 shards on disk:

        <shard_dir>/<layer_idx>_<t_bin>.h5
        ├── activations  (N, T, D) fp16
        └── ...

    Each cell becomes its own file so SAE training can stream a single cell
    without touching the others.
    """

    def __init__(
        self,
        max_records_per_cell: int = 0,
        shard_dir: str | Path | None = None,
        store_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.max_records_per_cell = max_records_per_cell
        self.shard_dir = Path(shard_dir) if shard_dir is not None else None
        if self.shard_dir is not None:
            self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.store_dtype = store_dtype
        # cell -> list of CPU tensors, each shape (T, D)
        self._cells: dict[tuple[int, int], list[Tensor]] = defaultdict(list)

    def __len__(self) -> int:
        return sum(len(v) for v in self._cells.values())

    def keys(self) -> list[tuple[int, int]]:
        return sorted(self._cells.keys())

    def write(self, layer: int, t_bin: int, x: Tensor) -> None:
        """Record `x` of shape (B, T, D) into the (layer, t_bin) cell.

        Splits the batch into B individual records so downstream sampling is
        per-sample (one residual stream per record).
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, D) tensor, got shape {tuple(x.shape)}.")
        records = self._cells[(int(layer), int(t_bin))]
        cpu = x.detach().to("cpu", dtype=self.store_dtype)
        # split along batch
        for i in range(cpu.shape[0]):
            records.append(cpu[i].clone())
        # auto-flush if exceeded budget
        if (
            self.max_records_per_cell > 0
            and len(records) >= self.max_records_per_cell
            and self.shard_dir is not None
        ):
            self._flush_cell(layer, t_bin)

    def get(self, layer: int, t_bin: int) -> list[Tensor]:
        return list(self._cells[(int(layer), int(t_bin))])

    def iter_cells(self) -> Iterator[tuple[tuple[int, int], list[Tensor]]]:
        for k in sorted(self._cells.keys()):
            yield k, self._cells[k]

    def clear(self) -> None:
        self._cells.clear()

    def flush_all(self) -> None:
        if self.shard_dir is None:
            raise RuntimeError("flush_all called but no shard_dir was configured.")
        for layer, t_bin in list(self._cells.keys()):
            self._flush_cell(layer, t_bin)

    def _flush_cell(self, layer: int, t_bin: int) -> None:
        if self.shard_dir is None:
            raise RuntimeError("_flush_cell called but no shard_dir was configured.")
        records = self._cells.pop((layer, t_bin))
        if not records:
            return
        path = self.shard_dir / f"{layer}_{t_bin}.h5"
        new = torch.stack(records, dim=0).numpy()  # (N, T, D)
        np_dtype = new.dtype  # already fp16 from store_dtype
        if path.exists():
            with h5py.File(path, "r+") as f:
                ds = f["activations"]
                old_n = ds.shape[0]
                ds.resize((old_n + new.shape[0], *new.shape[1:]))
                ds[old_n:] = new
        else:
            with h5py.File(path, "w") as f:
                f.create_dataset(
                    "activations",
                    data=new,
                    maxshape=(None, *new.shape[1:]),
                    dtype=np_dtype,
                    chunks=(1, *new.shape[1:]),
                    compression="lzf",
                )

    @staticmethod
    def load_cell(shard_path: str | Path) -> np.ndarray:
        with h5py.File(shard_path, "r") as f:
            return f["activations"][()]
