"""Generate comparison plots for E04 (Matryoshka vs TopK SAE sweep).

Reads per-stage final val/train metrics from both sweep output trees, then
produces three plots in Palette B:
  1. val_ev_scatter.png       — head-to-head scatter, diagonal at y=x
  2. val_ev_grouped_bars.png  — mean val EV per (cond, layer), grouped bars
  3. dead_pct_grouped_bars.png — mean dead% per (cond, layer), grouped bars

Restricted to "production" DiT stages (step >= 50000) where the underlying
DiT model has converged. Pre-convergence stages are saved separately for
the aggregate.csv.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Palette B (rules/flywheel.md)
PB = {
    "win": "#335C67",
    "neutral": "#E09F3E",
    "loss": "#9E2A2B",
    "cream": "#FFF3B0",
    "deep_red": "#540B0E",
}
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

BASE = Path("/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint")
SWEEPS = {"TopK k=128 d=32k": "sae_topk_k128_d32k",
          "Matryoshka k=256 d=32k": "sae_matryoshka_k256_d32k"}
PROD_DIT_STEPS = {50000, 100000, 150000, 200000}
OUT_DIR = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/flywheel/sae/e04_matryoshka_vs_topk/plots")
DATA_DIR = OUT_DIR.parent / "data"
D_SAE = 32768


def last_with(path: Path, key: str) -> dict | None:
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get(key) is not None:
                last = r
    return last


def load_rows() -> list[dict]:
    rows = []
    for sweep_label, dir_ in SWEEPS.items():
        for stage in sorted((BASE / dir_).glob("*/L*_T*/step_*")):
            m = re.match(r"L(\d+)_T(\d+)", stage.parent.name)
            if not m:
                continue
            layer, t_bin = int(m.group(1)), int(m.group(2))
            cond = stage.parent.parent.name
            dit_step = int(stage.name.removeprefix("step_"))
            v = last_with(stage / "metrics" / "val.jsonl", "val/ev") or {}
            t = last_with(stage / "metrics" / "train.jsonl", "metrics/explained_variance") or {}
            rows.append({
                "sweep": sweep_label, "cond": cond, "layer": layer, "t_bin": t_bin,
                "dit_step": dit_step,
                "val_ev": v.get("val/ev"), "val_cos": v.get("val/recon_cosine"),
                "val_mse": v.get("val/mse"), "val_dead": v.get("val/dead_features"),
                "val_l0": v.get("val/l0_mean"),
                "train_ev": t.get("metrics/explained_variance"),
                "train_dead": t.get("sparsity/dead_features"),
            })
    return rows


def plot_scatter(rows: list[dict]) -> None:
    """Head-to-head val EV — TopK on x, Matryoshka on y. Diagonal at y=x."""
    pairs = defaultdict(dict)
    for r in rows:
        if r["dit_step"] not in PROD_DIT_STEPS:
            continue
        key = (r["cond"], r["layer"], r["t_bin"], r["dit_step"])
        pairs[key][r["sweep"]] = r["val_ev"]

    xy = [(p[list(SWEEPS)[0]], p[list(SWEEPS)[1]])
          for p in pairs.values() if all(k in p for k in SWEEPS)]
    xs, ys = zip(*xy)
    xs, ys = np.array(xs), np.array(ys)

    fig, ax = plt.subplots(figsize=(6, 6))
    # Colour points by who wins
    win_b = ys > xs
    ax.scatter(xs[win_b], ys[win_b], s=22, c=PB["win"], alpha=0.75,
               label=f"Matryoshka wins (n={int(win_b.sum())})", zorder=3)
    ax.scatter(xs[~win_b], ys[~win_b], s=22, c=PB["loss"], alpha=0.75,
               label=f"TopK wins (n={int((~win_b).sum())})", zorder=3)
    # Diagonal
    lim = (min(xs.min(), ys.min()) - 0.02, 1.0)
    ax.plot(lim, lim, ls="--", c="#444444", lw=0.8, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"val EV — {list(SWEEPS)[0]}")
    ax.set_ylabel(f"val EV — {list(SWEEPS)[1]}")
    ax.set_title(f"Head-to-head val EV  (production stages, n={len(xs)})")
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "val_ev_scatter.png", dpi=160)
    plt.close(fig)
    print(f"  val_ev_scatter.png  →  Matryo wins {int(win_b.sum())}/{len(xs)}, "
          f"mean Δ = {(ys - xs).mean():+.4f}")


def plot_grouped_bars(rows: list[dict], metric_key: str, ylabel: str,
                       fname: str, ylim: tuple | None = None,
                       value_fn=lambda r, k: r[k]) -> None:
    """Grouped bar chart per (cond, layer); 2 bars per group (TopK | Matryoshka)."""
    by_group = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["dit_step"] not in PROD_DIT_STEPS:
            continue
        v = value_fn(r, metric_key)
        if v is None:
            continue
        by_group[(r["cond"], r["layer"])][r["sweep"]].append(v)

    conds = sorted({k[0] for k in by_group})
    layers = sorted({k[1] for k in by_group})
    sweep_labels = list(SWEEPS)
    bar_colors = {sweep_labels[0]: PB["loss"], sweep_labels[1]: PB["win"]}

    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4), sharey=True)
    if len(conds) == 1:
        axes = [axes]
    x = np.arange(len(layers))
    width = 0.36

    for ax, cond in zip(axes, conds):
        for i, sw in enumerate(sweep_labels):
            vals = [np.mean(by_group[(cond, L)].get(sw, [np.nan])) for L in layers]
            ax.bar(x + (i - 0.5) * width, vals, width=width,
                   color=bar_colors[sw], label=sw if ax is axes[0] else None,
                   edgecolor="#202020", linewidth=0.6)
        ax.set_xticks(x, [f"L{L}" for L in layers])
        ax.set_title(cond)
        ax.set_xlabel("SAE residual-stream layer")
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="lower right" if "EV" in ylabel else "upper right",
                   framealpha=0.92, fontsize=9)
    fig.suptitle(f"{ylabel}   (production DiT stages, mean over t-bin × DiT step)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=160)
    plt.close(fig)
    print(f"  {fname}  →  {len(by_group)} groups")


def plot_val_ev_trajectories(rows: list[dict]) -> None:
    """Mean val EV vs DiT-checkpoint step, one panel per (cond, layer)."""
    by_group = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        v = r["val_ev"]
        if v is None:
            continue
        by_group[(r["cond"], r["layer"])][r["sweep"]][r["dit_step"]].append(v)

    conds = sorted({k[0] for k in by_group})
    layers = sorted({k[1] for k in by_group})

    fig, axes = plt.subplots(len(layers), len(conds), figsize=(4.5 * len(conds), 3 * len(layers)),
                             sharex=True, sharey=False)
    sweep_styles = {list(SWEEPS)[0]: dict(color=PB["loss"], marker="o", linestyle="-"),
                    list(SWEEPS)[1]: dict(color=PB["win"],  marker="s", linestyle="-")}
    for i, L in enumerate(layers):
        for j, cond in enumerate(conds):
            ax = axes[i, j] if (len(layers) > 1 and len(conds) > 1) else (
                axes[max(i, j)] if (len(layers) == 1 or len(conds) == 1) else axes)
            g = by_group.get((cond, L), {})
            for sw, style in sweep_styles.items():
                d = g.get(sw, {})
                if not d:
                    continue
                steps = sorted(d)
                means = [np.mean(d[s]) for s in steps]
                ax.plot(steps, means, label=sw if (i == 0 and j == 0) else None, **style)
            ax.axvspan(0, 50000, color="#cccccc", alpha=0.25, lw=0)
            ax.set_title(f"{cond}  •  L{L}", fontsize=9)
            ax.set_ylim(-0.2, 1.0)
            if j == 0:
                ax.set_ylabel("mean val EV")
            if i == len(layers) - 1:
                ax.set_xlabel("DiT checkpoint step")
    axes[0, 0].legend(loc="lower right", framealpha=0.92, fontsize=8)
    fig.suptitle("val EV vs DiT-checkpoint step  (shaded = pre-convergence, step < 50k)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "val_ev_vs_dit_step.png", dpi=160)
    plt.close(fig)
    print("  val_ev_vs_dit_step.png  →  3×3 grid")


def write_aggregates(rows: list[dict]) -> None:
    """Persist aggregate JSON + CSV for the empirical node."""
    import csv
    with open(DATA_DIR / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # Per-group summary
    by = defaultdict(list)
    for r in rows:
        if r["dit_step"] not in PROD_DIT_STEPS:
            continue
        by[(r["sweep"], r["cond"], r["layer"])].append(r)
    summary = []
    for (sw, cond, L), g in by.items():
        evs = [r["val_ev"] for r in g if r["val_ev"] is not None]
        coss = [r["val_cos"] for r in g if r["val_cos"] is not None]
        deads = [r["val_dead"] for r in g if r["val_dead"] is not None]
        summary.append({
            "sweep": sw, "cond": cond, "layer": L, "n": len(g),
            "val_ev_mean": float(np.mean(evs)),
            "val_ev_min": float(np.min(evs)),
            "val_ev_max": float(np.max(evs)),
            "val_cos_mean": float(np.mean(coss)),
            "dead_pct_mean": float(100 * np.mean(deads) / D_SAE),
        })
    summary.sort(key=lambda r: (r["sweep"], r["cond"], r["layer"]))
    with open(DATA_DIR / "summary_production.json", "w") as fh:
        json.dump({"production_dit_steps": sorted(PROD_DIT_STEPS),
                   "d_sae": D_SAE, "groups": summary}, fh, indent=2)
    print(f"  aggregate.csv  →  {len(rows)} rows")
    print(f"  summary_production.json  →  {len(summary)} groups")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    print(f"Loaded {len(rows)} stages across {len(SWEEPS)} sweeps")
    plot_scatter(rows)
    plot_grouped_bars(rows, "val_ev", "mean val EV",
                       "val_ev_grouped_bars.png", ylim=(0.7, 1.0))
    plot_grouped_bars(rows, "val_dead", "mean dead %",
                       "dead_pct_grouped_bars.png", ylim=(0, 80),
                       value_fn=lambda r, k: 100 * r[k] / D_SAE if r[k] is not None else None)
    plot_val_ev_trajectories(rows)
    write_aggregates(rows)


if __name__ == "__main__":
    main()
