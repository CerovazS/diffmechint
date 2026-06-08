"""Unit tests for the E31 latent feature atlas scoring logic."""

from __future__ import annotations

import math

import torch

from diffmechint.analysis.latent_atlas import class_entropy_bits, score_features


def test_class_entropy_pure_and_uniform():
    assert class_entropy_bits([7] * 9) == 0.0
    # 9 distinct labels → log2(9) bits
    assert abs(class_entropy_bits(list(range(9))) - math.log2(9)) < 1e-9


def test_score_features_monosemantic_verdict():
    # 3 features, 30 images over 3 classes: f0 fires on >=9 images of class 2
    # (mono), f1 fires uniformly across classes (poly), f2 never fires (dead).
    # f0 needs >=9 firing images — fewer would pollute the top-9 with
    # zero-activation ties, which the density band excludes in production.
    torch.manual_seed(0)
    n_img, d_sae = 30, 3
    img_max = torch.zeros(n_img, d_sae)
    labels = torch.arange(n_img) % 3
    class2 = (labels == 2).nonzero().flatten()  # 10 images
    img_max[class2, 0] = torch.rand(len(class2)) + 1.0
    img_max[:, 1] = torch.rand(n_img) + 0.1
    scan = {
        "img_max": img_max,
        "labels": labels,
        # densities: f0 in band, f1 in band, f2 zero
        "density": torch.tensor([1e-3, 5e-2, 0.0]),
        "total_tokens": 1000,
    }
    rows = score_features(scan, density_lo=1e-4, density_hi=1e-1, entropy_threshold=1.0)
    assert rows[0]["monosemantic"] == 1
    assert rows[0]["top_class_idx"] == 2
    assert rows[1]["monosemantic"] == 0  # entropy ~log2(3) > 1.0 threshold
    assert rows[2]["live"] == 0 and rows[2]["monosemantic"] == 0
