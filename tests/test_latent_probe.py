"""Unit tests for the E34a latent probe pipeline (CPU, synthetic data)."""

from __future__ import annotations

import numpy as np
import torch

from diffmechint.analysis.latent_probe import (
    build_features,
    stratified_split,
    torch_probe,
)


def test_stratified_split_is_balanced_and_disjoint():
    labels = np.repeat(np.arange(5), 20)  # 5 classes x 20
    tr, te = stratified_split(labels, n_train=15, n_test=5, seed=0)
    assert len(tr) == 75 and len(te) == 25
    assert set(tr).isdisjoint(set(te))
    # every class represented in both splits
    assert set(labels[tr]) == set(range(5))
    assert set(labels[te]) == set(range(5))


def test_build_features_shapes():
    tokens = torch.randn(7, 4, 3)  # (N,T,D)
    assert build_features(tokens, "mean_pool").shape == (7, 3)
    assert build_features(tokens, "max_pool").shape == (7, 3)
    assert build_features(tokens, "flat").shape == (7, 12)


def test_probe_learns_separable_data():
    """Linearly separable clusters -> probe must beat chance decisively."""
    torch.manual_seed(0)
    n_classes, per, d = 5, 60, 8
    centers = torch.randn(n_classes, d) * 5.0
    xs, ys = [], []
    for c in range(n_classes):
        xs.append(centers[c] + torch.randn(per, d) * 0.5)
        ys.append(torch.full((per,), c))
    x = torch.cat(xs)
    y = torch.cat(ys)
    idx = torch.randperm(len(y))
    x, y = x[idx], y[idx]
    tr, te = stratified_split(y.numpy(), n_train=45, n_test=15, seed=0)
    top1, top5 = torch_probe(
        x[tr], y[tr], x[te], y[te], n_classes,
        device="cpu", epochs=80, lr=1e-2, batch=64, wd=1e-4,
    )
    assert top1 > 0.9, f"separable data should be near-perfect, got {top1}"
    assert top5 == 1.0
