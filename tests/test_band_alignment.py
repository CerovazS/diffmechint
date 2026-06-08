"""Unit tests for the E36 band-resolved CKA core (synthetic tokens, no cluster data)."""

from __future__ import annotations

import torch

from diffmechint.analysis.band_alignment import band_cka_for_pair, scrambled_band_cka
from diffmechint.spectral import band_decompose_tokens, octave_band_masks, scramble_band_signs

GRID = 16


def test_identical_tokens_give_unit_cka_in_every_band():
    g = torch.Generator().manual_seed(0)
    tokens = torch.randn(32, GRID * GRID, 8, generator=g)
    masks = octave_band_masks(GRID, GRID)
    ckas, n_tokens = band_cka_for_pair(tokens, tokens.clone(), GRID, GRID, masks, token_cap=5000, seed=0)
    assert n_tokens == 5000  # subsample positions shared by both sides
    assert set(ckas) == {"B0", "B1", "B2", "B3", "B4", "all"}
    for band, value in ckas.items():
        assert abs(value - 1.0) < 1e-6, f"{band}: {value}"


def test_shared_low_band_high_b1_low_b4():
    masks = octave_band_masks(GRID, GRID)
    g = torch.Generator().manual_seed(1)
    base = torch.randn(128, GRID * GRID, 8, generator=g)
    other = torch.randn(128, GRID * GRID, 8, generator=g)
    bands_base = band_decompose_tokens(base, GRID, GRID, masks)
    bands_other = band_decompose_tokens(other, GRID, GRID, masks)
    tokens_a = bands_base[1] + bands_base[4]
    tokens_b = bands_base[1] + bands_other[4]  # shares only B1 content with tokens_a
    ckas, _ = band_cka_for_pair(tokens_a, tokens_b, GRID, GRID, masks, token_cap=200_000, seed=0)
    assert ckas["B1"] > 0.9
    assert ckas["B4"] < 0.3


def test_scramble_lower_anchor_via_band_cka():
    masks = octave_band_masks(GRID, GRID)
    g = torch.Generator().manual_seed(2)
    tokens = torch.randn(128, GRID * GRID, 8, generator=g)
    gen = torch.Generator().manual_seed(11)
    scrambled = scramble_band_signs(tokens, GRID, GRID, masks, band=3, generator=gen)
    ckas, _ = band_cka_for_pair(tokens, scrambled, GRID, GRID, masks, token_cap=200_000, seed=0)
    assert ckas["B3"] < 0.3
    for band in ("B0", "B1", "B2", "B4"):
        assert abs(ckas[band] - 1.0) < 1e-4, f"{band}: {ckas[band]}"


def test_scrambled_band_cka_low_in_every_band():
    masks = octave_band_masks(GRID, GRID)
    g = torch.Generator().manual_seed(3)
    tokens = torch.randn(128, GRID * GRID, 8, generator=g)
    ckas, n_tokens = scrambled_band_cka(
        tokens, tokens.clone(), GRID, GRID, masks, token_cap=200_000, seed=0, scramble_seed=5
    )
    assert n_tokens == 128 * GRID * GRID
    assert set(ckas) == {"B0", "B1", "B2", "B3", "B4"}
    for band, value in ckas.items():
        assert value < 0.3, f"{band}: {value}"


def test_token_cap_subsample_is_deterministic():
    g = torch.Generator().manual_seed(4)
    tokens_a = torch.randn(16, GRID * GRID, 4, generator=g)
    tokens_b = torch.randn(16, GRID * GRID, 4, generator=g)
    masks = octave_band_masks(GRID, GRID)
    first, _ = band_cka_for_pair(tokens_a, tokens_b, GRID, GRID, masks, token_cap=1000, seed=7)
    second, _ = band_cka_for_pair(tokens_a, tokens_b, GRID, GRID, masks, token_cap=1000, seed=7)
    assert first == second
