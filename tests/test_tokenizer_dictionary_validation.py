"""Synthetic tests for Phase 4.10 tokenizer dictionary validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffmechint.hooks.timestep_router import timestep_context
from scripts.analysis.tokenizer_dictionary_validation import (
    fit_procrustes_affine,
    linear_cka,
    rsa_spearman,
)
from scripts.eval.cross_tokenizer_sae_substitution_fid import make_transferred_sae_hook


def test_linear_cka_identical_is_one() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(128, 32)).astype(np.float32)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-6)


def test_procrustes_recovers_known_rotation() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(256, 24)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(24, 24)))
    bias = rng.normal(size=(24,)).astype(np.float32)
    y = x @ q.astype(np.float32) + bias
    fitted = fit_procrustes_affine(x, y)
    np.testing.assert_allclose(fitted.apply(x), y, atol=2e-5, rtol=2e-5)


def test_rsa_identical_distances_high() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 16)).astype(np.float32)
    assert rsa_spearman(x, x) == pytest.approx(1.0, abs=1e-6)


class _IdentitySae(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_transferred_sae_hook_preserves_residual_shape() -> None:
    sae = _IdentitySae()
    stats = {"active": 0, "skipped": 0, "no_t": 0}
    hook = make_transferred_sae_hook(
        sae,
        target_to_source=lambda x: x,
        source_to_target=lambda x: x,
        t_center=0.20,
        t_tol=0.01,
        stats=stats,
    )
    x = torch.randn(2, 4, 768)
    with timestep_context(0.20):
        y = hook(None, (), x)
    assert y is not None
    assert y.shape == (2, 4, 768)
    assert stats["active"] == 1
