"""Unit tests for E38 spectral probing on synthetic band-localized class signal (CPU)."""

from __future__ import annotations

import numpy as np
import torch

from diffmechint.analysis.latent_probe import stratified_split, torch_probe
from diffmechint.analysis.spectral_probe import (
    filter_tokens_from_coeffs,
    gather_selected_tokens,
    probe_band_specs,
    spec_to_mask,
    token_selection,
)
from diffmechint.spectral import dct2, idct2, octave_band_masks, tokens_to_grid

GH = 8
D = 4
N_CLASSES = 6
PER_CLASS = 30


def _band2_class_tokens(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens (N,64,D) whose class signal lives ONLY in octave band 2; noise elsewhere."""
    g = torch.Generator().manual_seed(seed)
    masks = octave_band_masks(GH, GH)
    noise_mask = (~masks[2]).float()
    templates = torch.randn(N_CLASSES, D, GH, GH, generator=g) * masks[2].float() * 4.0
    coeffs, labels = [], []
    for c in range(N_CLASSES):
        noise = torch.randn(PER_CLASS, D, GH, GH, generator=g) * noise_mask
        coeffs.append(templates[c] + noise)
        labels.append(torch.full((PER_CLASS,), c))
    grid = idct2(torch.cat(coeffs))  # (N,D,GH,GH)
    tokens = grid.permute(0, 2, 3, 1).reshape(-1, GH * GH, D)
    return tokens, torch.cat(labels).long()


def test_spec_to_mask_partitions_and_lowpass():
    masks = octave_band_masks(GH, GH)
    assert spec_to_mask("broadband", masks) is None
    assert torch.equal(spec_to_mask("B2", masks), masks[2])
    lp2 = spec_to_mask("LP2", masks)
    assert torch.equal(lp2, masks[0] | masks[1] | masks[2])
    lp1 = spec_to_mask("LP1", masks)
    assert not (lp1 & masks[2]).any()  # LP1 excludes band 2


def test_filter_tokens_round_trip_with_full_mask():
    g = torch.Generator().manual_seed(1)
    tokens = torch.randn(5, GH * GH, D, generator=g)
    coeffs = dct2(tokens_to_grid(tokens, GH, GH))
    full = torch.ones(GH, GH, dtype=torch.bool)
    assert torch.allclose(filter_tokens_from_coeffs(coeffs, full), tokens, atol=1e-5)


def test_token_selection_deterministic_and_unique():
    sel1 = token_selection(10, 64, 6, seed=0)
    sel2 = token_selection(10, 64, 6, seed=0)
    assert np.array_equal(sel1, sel2)
    assert sel1.shape == (10, 6)
    for row in sel1:
        assert len(set(row.tolist())) == 6  # no replacement
    tokens = torch.arange(10 * 64 * 2).float().reshape(10, 64, 2)
    gathered = gather_selected_tokens(tokens, sel1)
    assert gathered.shape == (10, 6, 2)
    assert torch.equal(gathered[3, 2], tokens[3, sel1[3, 2]])


def test_band_probe_peaks_at_signal_band():
    """Class signal injected only in B2 -> flat probe peaks at B2, near chance at B4/LP1."""
    tokens, labels = _band2_class_tokens()
    tr, te = stratified_split(labels.numpy(), n_train=20, n_test=10, seed=0)
    rows = probe_band_specs(
        tokens, labels, tr, te,
        band_specs=["broadband", "B2", "B4", "LP1", "LP2"],
        feature_sets=["flat"], n_classes=N_CLASSES, device="cpu",
        epochs=40, lr=1e-2, batch=64, wd=1e-4, seed=0,
    )
    acc = {r["band_spec"]: r["top1"] for r in rows}
    chance = 1.0 / N_CLASSES
    assert acc["B2"] > 0.85, f"signal band should be decodable, got {acc['B2']}"
    assert acc["B4"] < chance + 0.25, f"noise-only band should be near chance, got {acc['B4']}"
    assert acc["B2"] > acc["B4"] + 0.4
    assert acc["LP1"] < chance + 0.25, f"LP1 excludes the signal band, got {acc['LP1']}"
    assert acc["LP2"] > 0.85, f"LP2 includes the signal band, got {acc['LP2']}"
    assert acc["broadband"] > 0.6  # signal present but diluted by full-band noise


def test_probe_band_specs_token_feature_set():
    tokens, labels = _band2_class_tokens(seed=2)
    tr, te = stratified_split(labels.numpy(), n_train=20, n_test=10, seed=0)
    rows = probe_band_specs(
        tokens, labels, tr, te, band_specs=["B2"], feature_sets=["token"],
        n_classes=N_CLASSES, device="cpu", epochs=5, lr=1e-2, batch=128, wd=1e-4,
        token_cap=tokens.shape[0] * 8, seed=0,
    )
    (r,) = rows
    assert r["feature_set"] == "token"
    assert r["dim"] == D
    assert r["n_train"] == len(tr) * 8  # per_img = token_cap // N = 8
    assert r["n_test"] == len(te) * 8


def test_mean_pool_is_degenerate_for_high_bands():
    """Spatial mean of a band-limited (b>0) signal is ~0 -> mean_pool carries no class info."""
    tokens, _labels = _band2_class_tokens(seed=3)
    masks = octave_band_masks(GH, GH)
    coeffs = dct2(tokens_to_grid(tokens, GH, GH))
    filt = filter_tokens_from_coeffs(coeffs, masks[2])
    assert filt.mean(dim=1).abs().max() < 1e-4


def test_torch_probe_reusable_on_band_features():
    """Direct probe on B2-filtered flat features replicates the peak (sanity of reuse)."""
    tokens, labels = _band2_class_tokens(seed=4)
    masks = octave_band_masks(GH, GH)
    coeffs = dct2(tokens_to_grid(tokens, GH, GH))
    filt = filter_tokens_from_coeffs(coeffs, masks[2])
    feats = filt.reshape(filt.shape[0], -1)
    tr, te = stratified_split(labels.numpy(), n_train=20, n_test=10, seed=0)
    top1, _ = torch_probe(
        feats[tr], labels[tr], feats[te], labels[te], N_CLASSES,
        device="cpu", epochs=40, lr=1e-2, batch=64, wd=1e-4,
    )
    assert top1 > 0.85
