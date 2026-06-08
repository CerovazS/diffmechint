"""E37 unit tests: band-resolved ridge transfer (synthetic) and broadband backward-compat."""

from __future__ import annotations

import pytest
import torch

from diffmechint.spectral import (
    N_OCTAVE_BANDS,
    band_decompose_tokens,
    band_tokens_from_coeffs,
    octave_band_masks,
    token_dct_coeffs,
)
from scripts.analysis.latent_dit_inheritance import (
    BandedTokens,
    ev,
    parse_bands,
    ridge_apply,
    ridge_fit,
)

GRID = 16
T = GRID * GRID


def _band_ridge_ev(x_tokens: torch.Tensor, y_tokens: torch.Tensor, band: int) -> float:
    """Per-band ridge X->Y EV via the same factored path the script uses."""
    xb = BandedTokens(x_tokens).band(band)
    yb = BandedTokens(y_tokens).band(band)
    x = xb.reshape(-1, xb.shape[-1])
    y = yb.reshape(-1, yb.shape[-1])
    n_train = int(0.8 * x.shape[0])
    w = ridge_fit(x[:n_train], y[:n_train])
    return ev(y[n_train:], ridge_apply(w, x[n_train:]))


def test_band_tokens_from_coeffs_matches_band_decompose_tokens():
    g = torch.Generator().manual_seed(0)
    tokens = torch.randn(3, T, 6, generator=g)
    masks = octave_band_masks(GRID, GRID)
    coeffs = token_dct_coeffs(tokens, GRID, GRID)
    one_by_one = torch.stack([band_tokens_from_coeffs(coeffs, m) for m in masks])
    assert torch.allclose(one_by_one, band_decompose_tokens(tokens, GRID, GRID, masks), atol=1e-5)


def test_banded_tokens_partition_and_coeff_cache():
    g = torch.Generator().manual_seed(1)
    tokens = torch.randn(4, T, 8, generator=g)
    bt = BandedTokens(tokens)
    parts = [bt.band(b) for b in range(N_OCTAVE_BANDS)]
    assert torch.allclose(torch.stack(parts).sum(dim=0), tokens, atol=1e-5)
    assert bt._coeffs is not None  # decomposed once, cached
    cached = bt._coeffs
    bt.band(2)
    assert bt._coeffs is cached


def test_band_minus_one_is_identical_to_raw():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(8, T, 4, generator=g)
    y = torch.randn(8, T, 6, generator=g)
    bt = BandedTokens(x)
    assert bt.band(-1) is x  # exact object: zero-cost, bit-identical E32 path
    assert bt._coeffs is None  # broadband never triggers the DCT

    xf, yf = x.reshape(-1, 4), y.reshape(-1, 6)
    n_train = int(0.8 * xf.shape[0])
    w = ridge_fit(xf[:n_train], yf[:n_train])
    raw_ev = ev(yf[n_train:], ridge_apply(w, xf[n_train:]))
    assert _band_ridge_ev(x, y, -1) == raw_ev


def test_parse_bands():
    assert parse_bands(None) == [-1]
    assert parse_bands(["all"]) == [-1, 0, 1, 2, 3, 4]
    assert parse_bands(["1", "3"]) == [1, 3]
    assert parse_bands(["-1", "4"]) == [-1, 4]
    with pytest.raises(ValueError):
        parse_bands(["5"])
    with pytest.raises(ValueError):
        parse_bands(["-2"])


def test_band_localized_noise_degrades_only_its_band():
    g = torch.Generator().manual_seed(3)
    n, dx, dy = 64, 8, 12
    # White tokens leave ~1/256 of the energy in B0 and the fixed ridge lambda
    # over-shrinks weak bands; add an explicit per-image DC term and scale up so
    # every band carries workable energy (natural latents are low-frequency-heavy).
    x = 4.0 * (torch.randn(n, T, dx, generator=g) + torch.randn(n, 1, dx, generator=g))
    w_true = torch.randn(dx, dy, generator=g) / dx**0.5
    masks = octave_band_masks(GRID, GRID)
    noise_b4 = band_decompose_tokens(torch.randn(n, T, dy, generator=g), GRID, GRID, masks)[4]
    y = x @ w_true + 8.0 * noise_b4

    evs = [_band_ridge_ev(x, y, b) for b in range(N_OCTAVE_BANDS)]
    for b in range(4):
        assert evs[b] > 0.95, f"band {b}: EV {evs[b]}"
    assert evs[4] < 0.5, f"band 4 should be noise-degraded, EV {evs[4]}"
