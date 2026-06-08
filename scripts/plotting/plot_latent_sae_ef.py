"""EF-ablation curves for Phase 4.19 latent SAEs (E30).

Reads `aggregate.csv` produced by `train_latent_sae.py` and renders, per
metric, one panel per condition with expansion factor on a log x-axis and
one line per k. Palette B, white background.

Usage:
    uv run python scripts/plotting/plot_latent_sae_ef.py \
        --aggregate <run_root>/aggregate.csv --out_dir <run_root>/plots
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette B (rules/flywheel.md)
PB = ["#335C67", "#E09F3E", "#9E2A2B", "#540B0E", "#FFF3B0"]

METRICS = [
    ("val_ev", "Held-out val EV", None),
    ("val_dead_pct", "Dead-feature fraction", None),
    ("val_live_features", "Live features", "log"),
    ("val_recon_cosine", "Reconstruction cosine", None),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()

    with args.aggregate.open() as f:
        rows = [
            {k: float(v) if k not in ("condition",) else v for k, v in r.items()}
            for r in csv.DictReader(f)
        ]
    conditions = sorted({r["condition"] for r in rows})
    ks = sorted({int(r["k"]) for r in rows})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key, label, yscale in METRICS:
        fig, axes = plt.subplots(
            1, len(conditions), figsize=(4.2 * len(conditions), 3.6),
            sharey=True, facecolor="#FFFFFF", squeeze=False,
        )
        axes = axes[0]
        for ax, cond in zip(axes, conditions, strict=True):
            ax.set_facecolor("#FFFFFF")
            series = defaultdict(list)  # k -> [(ef, value)]
            for r in rows:
                if r["condition"] == cond:
                    series[int(r["k"])].append((r["expansion_factor"], r[key]))
            for i, k in enumerate(ks):
                pts = sorted(series.get(k, []))
                if not pts:
                    continue
                ax.plot(
                    [x for x, _ in pts], [y for _, y in pts],
                    marker="o", color=PB[i % len(PB)], label=f"k={k}",
                )
            ax.set_xscale("log", base=2)
            if yscale:
                ax.set_yscale(yscale)
            ax.set_title(cond)
            ax.set_xlabel("expansion factor (d_sae / d_in)")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(label)
        axes[-1].legend(frameon=False)
        fig.suptitle(f"Latent Matryoshka SAEs — {label} vs EF", y=1.02)
        fig.tight_layout()
        out = args.out_dir / f"ef_{key}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
