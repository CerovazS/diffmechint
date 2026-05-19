"""Generate E05 plots: BatchTopK vs Matryoshka (with TopK for context).

Three plots in Palette B (#335C67 win, #9E2A2B regression, #E09F3E mixed):
  1. matryo_vs_batchtopk_scatter.png — head-to-head, diagonal y=x.
  2. three_sweep_grouped_bars.png    — mean val EV per (cond, layer), 3 bars.
  3. three_sweep_trajectories.png    — val EV vs DiT step, 3×3 grid.

Plus aggregate.csv + summary_production.json with the three-sweep group means.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PB = {"win": "#335C67", "neutral": "#E09F3E", "loss": "#9E2A2B"}
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

BASE = Path("/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint")
SWEEPS = {
    "TopK k=128 d=32k":         "sae_topk_k128_d32k",
    "BatchTopK k=128 d=32k":    "sae_batch_topk_k128_d32k",
    "Matryoshka k=256 d=32k":   "sae_matryoshka_k256_d32k",
}
PROD = {50000, 100000, 150000, 200000}
D_SAE = 32768
OUT = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/flywheel/sae/e05_batchtopk_vs_matryoshka")
PLOTS, DATA = OUT / "plots", OUT / "data"


def last_with(path: Path, key: str):
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get(key) is not None: last = r
    return last


def load_rows():
    rows = []
    for sweep, d in SWEEPS.items():
        for stage in sorted((BASE / d).glob("*/L*_T*/step_*")):
            m = re.match(r"L(\d+)_T(\d+)", stage.parent.name)
            if not m: continue
            layer, tbin = int(m.group(1)), int(m.group(2))
            cond = stage.parent.parent.name
            dit_step = int(stage.name.removeprefix("step_"))
            v = last_with(stage / "metrics" / "val.jsonl", "val/ev") or {}
            rows.append({
                "sweep": sweep, "cond": cond, "layer": layer, "t_bin": tbin,
                "dit_step": dit_step,
                "val_ev": v.get("val/ev"), "val_cos": v.get("val/recon_cosine"),
                "val_mse": v.get("val/mse"), "val_dead": v.get("val/dead_features"),
            })
    return rows


def plot_scatter(rows):
    """Matryoshka (y) vs BatchTopK (x). Diagonal y=x. Production stages only."""
    pairs = defaultdict(dict)
    for r in rows:
        if r["dit_step"] not in PROD: continue
        key = (r["cond"], r["layer"], r["t_bin"], r["dit_step"])
        pairs[key][r["sweep"]] = r["val_ev"]
    bt, ma = "BatchTopK k=128 d=32k", "Matryoshka k=256 d=32k"
    xy = [(p[bt], p[ma]) for p in pairs.values() if bt in p and ma in p]
    xs, ys = np.array([x for x, _ in xy]), np.array([y for _, y in xy])
    win_m = ys > xs

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs[win_m], ys[win_m], s=22, c=PB["win"], alpha=0.75,
               label=f"Matryoshka wins (n={int(win_m.sum())})", zorder=3)
    ax.scatter(xs[~win_m], ys[~win_m], s=22, c=PB["loss"], alpha=0.75,
               label=f"BatchTopK wins (n={int((~win_m).sum())})", zorder=3)
    lim = (min(xs.min(), ys.min()) - 0.02, 1.0)
    ax.plot(lim, lim, ls="--", c="#444", lw=0.8, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("val EV — BatchTopK k=128 d=32k")
    ax.set_ylabel("val EV — Matryoshka k=256 d=32k")
    ax.set_title(f"Head-to-head val EV  (production stages, n={len(xs)})")
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(PLOTS / "matryo_vs_batchtopk_scatter.png", dpi=160)
    plt.close(fig)
    print(f"  scatter → Matryo wins {int(win_m.sum())}/{len(xs)}, mean Δ={(ys-xs).mean():+.4f}")


def plot_grouped_bars(rows):
    """3 bars per (cond, layer): TopK | BatchTopK | Matryoshka."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["dit_step"] not in PROD: continue
        if r["val_ev"] is None: continue
        by[(r["cond"], r["layer"])][r["sweep"]].append(r["val_ev"])

    conds = sorted({k[0] for k in by})
    layers = sorted({k[1] for k in by})
    sweeps = list(SWEEPS)
    colors = {sweeps[0]: PB["loss"], sweeps[1]: PB["neutral"], sweeps[2]: PB["win"]}

    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4), sharey=True)
    if len(conds) == 1: axes = [axes]
    x = np.arange(len(layers))
    width = 0.27
    for ax, cond in zip(axes, conds):
        for i, sw in enumerate(sweeps):
            vals = [np.mean(by[(cond, L)].get(sw, [np.nan])) for L in layers]
            ax.bar(x + (i - 1) * width, vals, width=width, color=colors[sw],
                   label=sw if ax is axes[0] else None,
                   edgecolor="#202020", linewidth=0.6)
        ax.set_xticks(x, [f"L{L}" for L in layers])
        ax.set_title(cond)
        ax.set_xlabel("SAE residual-stream layer")
        ax.set_ylim(0.7, 1.0)
    axes[0].set_ylabel("mean val EV")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.92)
    fig.suptitle("Three-sweep mean val EV  (production DiT stages)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "three_sweep_grouped_bars.png", dpi=160)
    plt.close(fig)
    print(f"  grouped_bars → {len(by)} groups")


def plot_trajectories(rows):
    """val EV vs DiT step, 3×3 grid (cond × layer)."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["val_ev"] is None: continue
        by[(r["cond"], r["layer"])][r["sweep"]][r["dit_step"]].append(r["val_ev"])

    conds = sorted({k[0] for k in by})
    layers = sorted({k[1] for k in by})
    sweeps = list(SWEEPS)
    styles = {
        sweeps[0]: dict(color=PB["loss"],    marker="o", linestyle="-"),
        sweeps[1]: dict(color=PB["neutral"], marker="^", linestyle="-"),
        sweeps[2]: dict(color=PB["win"],     marker="s", linestyle="-"),
    }

    fig, axes = plt.subplots(len(layers), len(conds),
                             figsize=(4.5 * len(conds), 3 * len(layers)),
                             sharex=True, sharey=False)
    for i, L in enumerate(layers):
        for j, cond in enumerate(conds):
            ax = axes[i, j]
            for sw, st in styles.items():
                d = by.get((cond, L), {}).get(sw, {})
                if not d: continue
                steps = sorted(d)
                means = [np.mean(d[s]) for s in steps]
                ax.plot(steps, means, label=sw if (i==0 and j==0) else None, **st)
            ax.axvspan(0, 50000, color="#cccccc", alpha=0.25, lw=0)
            ax.set_title(f"{cond}  •  L{L}", fontsize=9)
            ax.set_ylim(-0.2, 1.0)
            if j == 0: ax.set_ylabel("mean val EV")
            if i == len(layers) - 1: ax.set_xlabel("DiT checkpoint step")
    axes[0, 0].legend(loc="lower right", fontsize=8, framealpha=0.92)
    fig.suptitle("val EV vs DiT-checkpoint step  (shaded = pre-convergence, step <50k)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "three_sweep_trajectories.png", dpi=160)
    plt.close(fig)
    print("  trajectories → 3×3 grid")


def write_aggregates(rows):
    with open(DATA / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    # Group summary
    by = defaultdict(list)
    for r in rows:
        if r["dit_step"] not in PROD: continue
        by[(r["sweep"], r["cond"], r["layer"])].append(r)
    summary = []
    for (sw, cond, L), g in by.items():
        evs = [r["val_ev"] for r in g if r["val_ev"] is not None]
        deads = [r["val_dead"] for r in g if r["val_dead"] is not None]
        summary.append({
            "sweep": sw, "cond": cond, "layer": L, "n": len(g),
            "val_ev_mean": float(np.mean(evs)),
            "val_ev_min": float(np.min(evs)),
            "val_ev_max": float(np.max(evs)),
            "dead_pct_mean": float(100 * np.mean(deads) / D_SAE) if deads else None,
        })
    summary.sort(key=lambda r: (r["sweep"], r["cond"], r["layer"]))
    with open(DATA / "summary_production.json", "w") as fh:
        json.dump({"production_dit_steps": sorted(PROD), "d_sae": D_SAE,
                   "groups": summary}, fh, indent=2)
    # Head-to-head head record
    h2h = {}
    for s1, s2 in [("TopK k=128 d=32k", "BatchTopK k=128 d=32k"),
                   ("TopK k=128 d=32k", "Matryoshka k=256 d=32k"),
                   ("BatchTopK k=128 d=32k", "Matryoshka k=256 d=32k")]:
        p = defaultdict(dict)
        for r in rows:
            if r["dit_step"] not in PROD: continue
            if r["sweep"] not in (s1, s2): continue
            p[(r["cond"], r["layer"], r["t_bin"], r["dit_step"])][r["sweep"]] = r["val_ev"]
        diffs = [v[s2] - v[s1] for v in p.values() if s1 in v and s2 in v]
        h2h[f"{s2} - {s1}"] = {
            "n": len(diffs),
            "wins_for_s2": sum(1 for d in diffs if d > 0),
            "mean_delta": float(np.mean(diffs)),
            "median_delta": float(sorted(diffs)[len(diffs)//2]),
        }
    with open(DATA / "head_to_head.json", "w") as fh:
        json.dump(h2h, fh, indent=2)
    print(f"  aggregate.csv → {len(rows)} rows")
    print(f"  head_to_head.json → {len(h2h)} comparisons")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    print(f"Loaded {len(rows)} stages across {len(SWEEPS)} sweeps")
    plot_scatter(rows)
    plot_grouped_bars(rows)
    plot_trajectories(rows)
    write_aggregates(rows)


if __name__ == "__main__":
    main()
