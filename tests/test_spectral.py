"""Unit tests for the DCT-2D spectral framework (program.md E36-E39 test plan)."""

from __future__ import annotations

import torch

from diffmechint.spectral import (
    N_OCTAVE_BANDS,
    band_decompose,
    band_decompose_tokens,
    dct2,
    grid_to_tokens,
    idct2,
    octave_band_masks,
    radial_band_energy,
    radial_bin_map,
    scramble_band_signs,
    tokens_to_grid,
)


def test_dct2_round_trip_and_parseval():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 3, 16, 16, generator=g)
    coeffs = dct2(x)
    assert torch.allclose(idct2(coeffs), x, atol=1e-5)
    assert torch.allclose(coeffs.pow(2).sum(), x.pow(2).sum(), rtol=1e-5)


def test_radial_bin_map_matches_paper_formula():
    m = radial_bin_map(16, 16, 16)
    assert m[0, 0].item() == 0  # DC in bin 0
    rmax = (2 * 15.0**2) ** 0.5
    assert m[15, 15].item() == 15  # corner clamped to last bin
    assert m[0, 8].item() == int(8.0 / rmax * 16)
    assert m[3, 4].item() == int(5.0 / rmax * 16)
    flat = m.reshape(-1)
    rr = torch.sqrt(
        torch.arange(16).view(16, 1).float() ** 2 + torch.arange(16).view(1, 16).float() ** 2
    ).reshape(-1)
    order = torch.argsort(rr)
    assert (flat[order].diff() >= 0).all()  # monotone in radius


def test_octave_masks_partition_and_band_decompose():
    masks = octave_band_masks(16, 16)
    assert masks.shape[0] == N_OCTAVE_BANDS
    assert masks.sum(dim=0).eq(1).all()  # exact partition of the grid
    assert masks[0].sum().item() == 1 and masks[0, 0, 0]
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 2, 16, 16, generator=g)
    parts = band_decompose(x, masks)
    assert torch.allclose(parts.sum(dim=0), x, atol=1e-5)
    energies = parts.pow(2).sum(dim=(1, 2, 3, 4))
    assert torch.allclose(energies.sum(), x.pow(2).sum(), rtol=1e-5)  # orthogonal split


def test_tokens_grid_round_trip_matches_patchify_order():
    g = torch.Generator().manual_seed(2)
    tokens = torch.randn(3, 256, 16, generator=g)
    grid = tokens_to_grid(tokens, 16, 16)
    assert grid.shape == (3, 16, 16, 16)
    assert torch.equal(grid_to_tokens(grid), tokens)
    assert torch.equal(grid[0, :, 0, 1], tokens[0, 1, :])  # token 1 = row 0, col 1 (row-major)


def test_synthetic_band_signal_lands_in_its_band():
    masks = octave_band_masks(16, 16)
    h = torch.arange(16).view(16, 1).float()
    w = torch.arange(16).view(1, 16).float()
    lo = torch.cos(torch.pi * (2 * h + 1) * 1 / 32) * torch.cos(torch.pi * (2 * w + 1) * 1 / 32)
    hi = torch.cos(torch.pi * (2 * h + 1) * 14 / 32) * torch.cos(torch.pi * (2 * w + 1) * 14 / 32)
    x = torch.stack([lo, hi]).unsqueeze(1)  # (2, 1, 16, 16)
    parts = band_decompose(x, masks)
    band_energy = parts.pow(2).sum(dim=(2, 3, 4))  # (B, 2)
    assert band_energy[:, 0].argmax().item() == 1  # (1,1) → rnorm≈0.067 → B1
    assert band_energy[:, 1].argmax().item() == 4  # (14,14) → rnorm≈0.93 → B4
    pure = band_energy / band_energy.sum(dim=0)
    assert pure[1, 0] > 0.99 and pure[4, 1] > 0.99


def test_radial_band_energy_constant_input():
    x = torch.zeros(2, 3, 16, 16)
    x[:, :] = 5.0
    prof = radial_band_energy(dct2(x).pow(2).mean(dim=1), n_bins=16)
    assert prof[:, 0].gt(0).all() and torch.allclose(prof[:, 1:], torch.zeros(2, 15))


def test_scramble_preserves_band_power_and_destroys_alignment():
    g = torch.Generator().manual_seed(3)
    masks = octave_band_masks(16, 16)
    tokens = torch.randn(16, 256, 8, generator=g)
    gen = torch.Generator().manual_seed(7)
    scrambled = scramble_band_signs(tokens, 16, 16, masks, band=3, generator=gen)
    p0 = dct2(tokens_to_grid(tokens, 16, 16)).pow(2)
    p1 = dct2(tokens_to_grid(scrambled, 16, 16)).pow(2)
    assert torch.allclose(p0, p1, atol=1e-4)  # per-coefficient power preserved
    b3 = band_decompose_tokens(tokens, 16, 16, masks)[3]
    b3s = band_decompose_tokens(scrambled, 16, 16, masks)[3]
    flat_a, flat_b = b3.reshape(-1), b3s.reshape(-1)
    corr = torch.corrcoef(torch.stack([flat_a, flat_b]))[0, 1].abs()
    assert corr < 0.2  # alignment destroyed inside the scrambled band
    b1 = band_decompose_tokens(tokens, 16, 16, masks)[1]
    b1s = band_decompose_tokens(scrambled, 16, 16, masks)[1]
    assert torch.allclose(b1, b1s, atol=1e-5)  # other bands untouched
