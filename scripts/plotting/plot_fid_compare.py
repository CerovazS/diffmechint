"""Compare FID curves across multiple runs on a single figure (palette B).

Usage:
    uv run python scripts/plot_fid_compare.py \
        <run_dir1> <run_dir2> ... \
        --labels sd_vae eq_vae repa_e dc_ae_p2 dc_ae_p1 \
        --out outputs/fid_cross_condition.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Palette B (per ~/.claude/rules/swe-stack.md)
PALETTE_B = ["#335C67", "#E09F3E", "#9E2A2B", "#540B0E", "#FFF3B0"]
# Cream (#FFF3B0) is too light against a white background; if more colors are
# needed we extend with ember-light fallback.
EXTRA = ["#3a6080", "#946030", "#7a6820", "#386858"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path,
                   help="Run directories containing metrics/validation/fid.csv")
    p.add_argument("--labels", nargs="+", type=str, required=True,
                   help="Display labels, same order as runs")
    p.add_argument("--out", type=Path, default=Path("outputs/fid_cross_condition.png"))
    p.add_argument("--title", type=str, default="FID curves across K=5 tokenizer conditions")
    args = p.parse_args()

    if len(args.runs) != len(args.labels):
        raise SystemExit(f"got {len(args.runs)} runs but {len(args.labels)} labels")

    # Load all curves
    curves: list[tuple[str, pd.DataFrame]] = []
    for run, lab in zip(args.runs, args.labels):
        fid = run / "metrics" / "validation" / "fid.csv"
        if not fid.exists():
            print(f"  [skip] no fid.csv at {fid}")
            continue
        df = pd.read_csv(fid)
        if df.empty:
            print(f"  [skip] empty fid.csv at {fid}")
            continue
        curves.append((lab, df))

    if not curves:
        raise SystemExit("no FID data found across the given runs")

    colors = (PALETTE_B + EXTRA)[: len(curves)]

    # 3-panel layout: linear (full), log-y (full), linear zoomed (top performers <100)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for (lab, df), c in zip(curves, colors):
        for ax in axes:
            ax.plot(df["step"], df["fid"], "o-", color=c, lw=1.8, ms=6, label=lab)

    # axes[0]: linear full range
    axes[0].set_title("Linear scale (all 5 conditions)")
    axes[0].set_xlabel("global step")
    axes[0].set_ylabel("Clean-FID (5k samples, CFG=1.5)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # axes[1]: log-y so plateau-high curves are visible alongside fast learners
    axes[1].set_yscale("log")
    axes[1].set_title("Log scale (compresses cross-condition range)")
    axes[1].set_xlabel("global step")
    axes[1].set_ylabel("Clean-FID (log)")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(loc="upper right")

    # axes[2]: zoom on top performers (FID < 110)
    axes[2].set_ylim(20, 110)
    axes[2].set_title("Zoom: top performers (FID < 110)")
    axes[2].set_xlabel("global step")
    axes[2].set_ylabel("Clean-FID")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ wrote {args.out}")

    # Print summary table
    print()
    print(f"{'step':>8}  " + "  ".join(f"{lab:>10}" for lab, _ in curves))
    print("-" * (10 + 12 * len(curves)))
    # Get all unique steps
    all_steps = sorted({int(s) for _, df in curves for s in df["step"]})
    for s in all_steps:
        row = [f"{s:>8}"]
        for _, df in curves:
            match = df[df["step"] == s]
            row.append(f"{match['fid'].iloc[0]:>10.2f}" if len(match) else f"{'—':>10}")
        print("  ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
