"""Layer × timestep linear-probe grid (PLAN §9 — Revelio convention).

For one (condition, dit_checkpoint), train one probe per (layer, t_bin)
cell across each concept axis. Output is a 3D matrix indexed by
[concept, layer_idx_in_grid, t_bin] of probe accuracies + the
(layer, t_bin) cell where each concept peaks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from diffmechint.hooks.activation_buffer import ActivationBuffer
from diffmechint.utils import info, ok, warn

from .concepts import ConceptAxis, expand_labels_for_tokens, pool_tokens

import torch  # noqa: E402  (after sklearn to keep import errors clear)


@dataclass
class CellResult:
    layer: int
    t_bin: int
    accuracy: float
    n_train: int
    n_test: int
    n_classes: int
    pool_mode: str


@dataclass
class GridResult:
    concept: str
    cells: list[CellResult] = field(default_factory=list)

    def matrix(self, layers: list[int], t_bins: list[int]) -> np.ndarray:
        """Return a (len(layers), len(t_bins)) accuracy matrix; NaN for missing."""
        arr = np.full((len(layers), len(t_bins)), np.nan)
        idx_l = {l_: i for i, l_ in enumerate(layers)}
        idx_t = {t_: i for i, t_ in enumerate(t_bins)}
        for c in self.cells:
            if c.layer in idx_l and c.t_bin in idx_t:
                arr[idx_l[c.layer], idx_t[c.t_bin]] = c.accuracy
        return arr

    def peak(self) -> CellResult | None:
        if not self.cells:
            return None
        return max(self.cells, key=lambda c: c.accuracy)


def train_probe(
    activations: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    test_size: float = 0.2,
    max_samples: int = 50_000,
    max_iter: int = 1000,
    C: float = 1.0,
    seed: int = 0,
) -> tuple[float, int, int, int]:
    """Train a multinomial logistic-regression probe; return (accuracy, n_train, n_test, n_classes).

    `activations` accepted as np.ndarray or torch.Tensor; shape `(N, D)`.
    Labels must be 1-D `(N,)`.
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    activations = np.asarray(activations).astype(np.float32)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    if activations.shape[0] != labels.shape[0]:
        raise ValueError(
            f"activations (N={activations.shape[0]}) and labels (N={labels.shape[0]}) misaligned."
        )

    # Optional sub-sample to keep probe training tractable.
    if activations.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(activations.shape[0], size=max_samples, replace=False)
        activations = activations[idx]
        labels = labels[idx]

    n_classes = int(np.unique(labels).shape[0])
    if n_classes < 2:
        warn(f"Probe degenerate: only {n_classes} unique labels — returning accuracy=NaN.")
        return float("nan"), int(activations.shape[0]), 0, n_classes

    x_tr, x_te, y_tr, y_te = train_test_split(
        activations, labels, test_size=test_size, random_state=seed, stratify=None
    )
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(x_tr, y_tr)
    acc = float(clf.score(x_te, y_te))
    return acc, int(x_tr.shape[0]), int(x_te.shape[0]), n_classes


def probe_one_cell(
    shard_path: Path | str,
    *,
    pool: str = "tokens",
    max_samples: int = 50_000,
    seed: int = 0,
) -> CellResult | None:
    """Load one HDF5 cell, train a probe, return its result.

    The shard file name is expected to follow the
    `<layer>_<tbin>.h5` convention from `ActivationBuffer`.
    Returns `None` when the cell has no labels (Phase 4 SAE-only shards).
    """
    p = Path(shard_path)
    layer, t_bin = (int(x) for x in p.stem.split("_"))
    acts, labels = ActivationBuffer.load_cell_with_labels(p)
    if labels is None:
        warn(f"{p.name}: no labels in HDF5 — skipping probe.")
        return None
    acts_t = torch.from_numpy(acts.astype(np.float32))
    labels_t = torch.from_numpy(labels.astype(np.int64))
    pooled = pool_tokens(acts_t, mode=pool)
    if pool == "tokens":
        labels_expanded = expand_labels_for_tokens(labels_t, acts_t.shape[1])
    else:
        labels_expanded = labels_t

    acc, n_tr, n_te, n_cls = train_probe(
        pooled, labels_expanded, max_samples=max_samples, seed=seed
    )
    return CellResult(
        layer=layer,
        t_bin=t_bin,
        accuracy=acc,
        n_train=n_tr,
        n_test=n_te,
        n_classes=n_cls,
        pool_mode=pool,
    )


def evaluate_grid(
    shard_dir: Path | str,
    *,
    concept: ConceptAxis,
    layers: list[int],
    t_bins: list[int],
    pool: str = "tokens",
    max_samples: int = 50_000,
    seed: int = 0,
) -> GridResult:
    """Sweep over `(layer, t_bin)` cells under `shard_dir` and probe each."""
    shard_dir = Path(shard_dir)
    out = GridResult(concept=concept.name)
    for layer in layers:
        for t_bin in t_bins:
            shard = shard_dir / f"{layer}_{t_bin}.h5"
            if not shard.exists():
                warn(f"missing cell {shard.name} — skipping.")
                continue
            result = probe_one_cell(shard, pool=pool, max_samples=max_samples, seed=seed)
            if result is not None:
                info(
                    f"  cell ({layer}, {t_bin}): acc={result.accuracy:.4f} "
                    f"n_tr={result.n_train} n_te={result.n_test} cls={result.n_classes}"
                )
                out.cells.append(result)
    if (peak := out.peak()) is not None:
        ok(f"[{concept.name}] peak @ (layer={peak.layer}, t_bin={peak.t_bin}) acc={peak.accuracy:.4f}")
    return out


def write_grid_result(grid: GridResult, out_path: Path | str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "concept": grid.concept,
        "peak": (
            {
                "layer": grid.peak().layer,
                "t_bin": grid.peak().t_bin,
                "accuracy": grid.peak().accuracy,
            }
            if grid.peak() is not None
            else None
        ),
        "cells": [
            {
                "layer": c.layer,
                "t_bin": c.t_bin,
                "accuracy": c.accuracy,
                "n_train": c.n_train,
                "n_test": c.n_test,
                "n_classes": c.n_classes,
                "pool_mode": c.pool_mode,
            }
            for c in grid.cells
        ],
    }
    out.write_text(json.dumps(payload, indent=2))
