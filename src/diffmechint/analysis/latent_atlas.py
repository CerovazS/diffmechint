"""Scoring helpers for the latent-token feature atlas (E31/E33)."""

from __future__ import annotations

import math
from collections import Counter

TOP_N = 9


def class_entropy_bits(labels: list[int]) -> float:
    """Shannon entropy (bits) of a label multiset (E11 monosemanticity score)."""
    n = len(labels)
    return -sum(
        (c / n) * math.log2(c / n) for c in Counter(labels).values()
    )


def score_features(
    scan: dict,
    *,
    density_lo: float,
    density_hi: float,
    entropy_threshold: float,
) -> list[dict]:
    """Per-feature top-9 images, class entropy, and monosemanticity verdict."""
    img_max, labels, density = scan["img_max"], scan["labels"], scan["density"]
    d_sae = img_max.shape[1]
    top_act, top_idx = img_max.topk(TOP_N, dim=0)  # (9, d_sae)
    rows = []
    for f in range(d_sae):
        dens = float(density[f])
        live = dens > 0
        idx = top_idx[:, f].tolist()
        lab = labels[top_idx[:, f]].tolist()
        act = [round(float(a), 5) for a in top_act[:, f]]
        ent = class_entropy_bits(lab) if live else float("nan")
        in_band = density_lo <= dens <= density_hi
        mono = live and in_band and ent < entropy_threshold
        top_label = Counter(lab).most_common(1)[0][0] if live else -1
        rows.append({
            "feature": f,
            "density": f"{dens:.3e}",
            "live": int(live),
            "in_density_band": int(in_band),
            "top9_entropy_bits": round(ent, 4) if live else "",
            "monosemantic": int(mono),
            "top_class_idx": top_label,
            "top9_image_idx": " ".join(map(str, idx)),
            "top9_labels": " ".join(map(str, lab)),
            "top9_activations": " ".join(map(str, act)),
        })
    return rows
