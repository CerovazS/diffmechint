"""Plot training/validation/FID curves for one run from the standardized
`metrics/` layout. Output PNGs in <run>/plots/.

Palette B (per ~/.claude/rules/swe-stack.md):
  #335C67 (teal)  #FFF3B0 (cream)  #E09F3E (gold)  #9E2A2B (red)  #540B0E (wine)

Usage:
    uv run python scripts/plot_run.py <run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = {
    "teal": "#335C67",
    "gold": "#E09F3E",
    "red":  "#9E2A2B",
    "wine": "#540B0E",
}


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path.relative_to(out_path.parent.parent)}")


def plot_train_loss(run_dir: Path, plots_dir: Path) -> None:
    p = run_dir / "metrics" / "train" / "loss_step.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d["step"], d["loss"], color=PALETTE["teal"], lw=0.8, alpha=0.5, label="loss (raw)")
    if len(d) > 50:
        d["loss_ma"] = d["loss"].rolling(window=max(50, len(d) // 50), min_periods=1).mean()
        ax.plot(d["step"], d["loss_ma"], color=PALETTE["wine"], lw=1.4, label="loss (rolling)")
    ax.set_xlabel("global step")
    ax.set_ylabel("train loss")
    ax.set_title(f"{run_dir.name} — training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, plots_dir / "train_loss.png")


def plot_val_loss(run_dir: Path, plots_dir: Path) -> None:
    p = run_dir / "metrics" / "validation" / "loss.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    if len(d) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d["step"], d["loss"], "o-", color=PALETTE["gold"], lw=1.4, ms=4, label="val loss")
    ax.set_xlabel("global step")
    ax.set_ylabel("val loss")
    ax.set_title(f"{run_dir.name} — validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, plots_dir / "val_loss.png")


def plot_fid(run_dir: Path, plots_dir: Path) -> None:
    p = run_dir / "metrics" / "validation" / "fid.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    if len(d) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d["step"], d["fid"], "s-", color=PALETTE["red"], lw=1.6, ms=6, label=f"FID-{int(d['n_samples'].iloc[0])//1000}k (CFG={d['cfg_scale'].iloc[0]:.1f})")
    ax.set_xlabel("global step")
    ax.set_ylabel("Clean-FID")
    ax.set_title(f"{run_dir.name} — FID curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, plots_dir / "fid.png")


def plot_combined(run_dir: Path, plots_dir: Path) -> None:
    train = run_dir / "metrics" / "train" / "loss_step.csv"
    val = run_dir / "metrics" / "validation" / "loss.csv"
    fid = run_dir / "metrics" / "validation" / "fid.csv"
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    if train.exists():
        d = pd.read_csv(train)
        if len(d) > 50:
            d["loss_ma"] = d["loss"].rolling(window=max(50, len(d) // 50), min_periods=1).mean()
            axes[0].plot(d["step"], d["loss_ma"], color=PALETTE["teal"], lw=1.4)
        else:
            axes[0].plot(d["step"], d["loss"], color=PALETTE["teal"], lw=1.4)
        axes[0].set_ylabel("train loss")
        axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"{run_dir.name}")

    if val.exists():
        d = pd.read_csv(val)
        if len(d):
            axes[1].plot(d["step"], d["loss"], "o-", color=PALETTE["gold"], lw=1.4, ms=4)
        axes[1].set_ylabel("val loss")
        axes[1].grid(True, alpha=0.3)

    if fid.exists():
        d = pd.read_csv(fid)
        if len(d):
            axes[2].plot(d["step"], d["fid"], "s-", color=PALETTE["red"], lw=1.6, ms=6)
        axes[2].set_ylabel("Clean-FID")
        axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("global step")
    _save(fig, plots_dir / "summary.png")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for arg in sys.argv[1:]:
        run_dir = Path(arg)
        if not run_dir.is_dir():
            print(f"  [skip] not a dir: {run_dir}")
            continue
        plots_dir = run_dir / "plots"
        print(f"--- plotting {run_dir.name} ---")
        plot_train_loss(run_dir, plots_dir)
        plot_val_loss(run_dir, plots_dir)
        plot_fid(run_dir, plots_dir)
        plot_combined(run_dir, plots_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
