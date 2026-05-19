"""E06 — substitution-FID plots from Phase 4.5a aggregate.csv.

Three outputs (Palette B: #335C67 win, #E09F3E mixed, #9E2A2B regression):
  1. delta_fid_heatmap.png — 3 panels (one per cond), 3×3 (L, T) ΔFID heatmap.
  2. delta_fid_bar.png     — grouped bars: 3 conds × 9 cells, color by L.
  3. fid_vs_val_ev.png     — scatter ΔFID vs val EV from E05 aggregate (if found).

Also writes:
  data/headline.json  — baseline FID, mean/max/min ΔFID, faithful-count.
  data/aggregate_with_delta.csv — flat per-row with the ΔFID column.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PB = {"win": "#335C67", "neutral": "#E09F3E", "loss": "#9E2A2B",
      "cream": "#FFF3B0", "dark": "#540B0E"}
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
})

ROOT = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint")
AGG_CSV = ROOT / "outputs/phase4_5a_subst_fid/aggregate.csv"
OUT_DIR = ROOT / "flywheel/sae/e06_substitution_fid"
PLOTS = OUT_DIR / "plots"
DATA = OUT_DIR / "data"

CONDS = ["sd_vae", "repa_e", "eq_vae"]
LAYERS = [3, 6, 9]
T_BINS = [0, 1, 2]
T_CENTERS = {0: 0.025, 1: 0.20, 2: 0.50}


def load_rows():
    rows = []
    with open(AGG_CSV) as fh:
        for r in csv.DictReader(fh):
            row = {
                "cond": r["condition"],
                "layer": int(r["layer"]) if r["layer"] not in ("", "None") else None,
                "t_bin": int(r["t_bin"]) if r["t_bin"] not in ("", "None") else None,
                "fid": float(r["fid"]),
                "substituted": r["substituted"] == "True",
                "hook_stats": json.loads(r["hook_stats"]),
                "gen_min": float(r["gen_minutes"]),
            }
            rows.append(row)
    return rows


def split_baselines(rows):
    base = {r["cond"]: r["fid"] for r in rows if not r["substituted"]}
    subs = [r for r in rows if r["substituted"]]
    return base, subs


def plot_heatmap(base, subs):
    """3 panels: one heatmap per cond, 3×3 (L × T) ΔFID."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    grids = {}
    for cond in CONDS:
        g = np.full((3, 3), np.nan)
        for r in subs:
            if r["cond"] != cond:
                continue
            li = LAYERS.index(r["layer"])
            ti = T_BINS.index(r["t_bin"])
            g[li, ti] = r["fid"] - base[cond]
        grids[cond] = g

    vmin = min(g.min() for g in grids.values())
    vmax = max(g.max() for g in grids.values())
    span = max(abs(vmin), abs(vmax))

    # Custom diverging colormap centered at 0: cream-to-orange-to-deep-red for + side,
    # cream-to-teal for - side (mostly + here).
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "pb_div",
        [(0.0, PB["win"]),   (0.5, PB["cream"]),  (1.0, PB["loss"])],
        N=256,
    )

    for ax, cond in zip(axes, CONDS):
        g = grids[cond]
        im = ax.imshow(g, cmap=cmap, vmin=-span, vmax=span, aspect="equal")
        ax.set_xticks(range(3), [f"T{t}\n{T_CENTERS[t]:.3f}" for t in T_BINS])
        ax.set_yticks(range(3), [f"L{l}" for l in LAYERS])
        ax.set_title(f"{cond}   (baseline FID = {base[cond]:.2f})", fontsize=11)
        for i in range(3):
            for j in range(3):
                v = g[i, j]
                txtcolor = "#202020" if abs(v) < 0.6 * span else "#fafafa"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color=txtcolor, fontsize=11, fontweight="bold")
        ax.set_xlabel("diffusion timestep bin")
        if cond == CONDS[0]:
            ax.set_ylabel("SAE residual-stream layer")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                        label="ΔFID (substituted − baseline)")
    cbar.ax.axhline(0, color="#444", lw=0.6)
    cbar.ax.axhline(2.0, color="#444", lw=0.6, ls=":")
    cbar.ax.text(1.05, 2.0, "  faithful gate", transform=cbar.ax.get_yaxis_transform(),
                 va="center", fontsize=8, color="#444")
    fig.suptitle("ΔFID heatmap — Matryoshka SAE substitution at one (layer, t-bin) "
                 "during sampling (DiT step 200k, 5 000 samples, ODE-dopri5 250, CFG 1.5)",
                 fontsize=11)
    fig.savefig(PLOTS / "delta_fid_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  heatmap → {PLOTS / 'delta_fid_heatmap.png'}")


def plot_bars(base, subs):
    """Grouped bars: 3 conds × (3 layers × 3 t_bins) = 9 bars per cond, colour by layer."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    layer_colors = {3: PB["win"], 6: PB["neutral"], 9: PB["loss"]}
    x_positions = []
    x_labels = []
    for ti in T_BINS:
        for li, l in enumerate(LAYERS):
            x_positions.append(ti * 3.5 + li)
            x_labels.append(f"L{l}\nT{ti}")

    for ax, cond in zip(axes, CONDS):
        deltas = []
        colors = []
        for ti in T_BINS:
            for l in LAYERS:
                for r in subs:
                    if r["cond"] == cond and r["layer"] == l and r["t_bin"] == ti:
                        deltas.append(r["fid"] - base[cond])
                        colors.append(layer_colors[l])
                        break
        ax.bar(x_positions, deltas, color=colors, edgecolor="#202020", linewidth=0.5)
        ax.axhline(0, color="#444", lw=0.8)
        ax.axhline(2.0, color="#444", lw=0.6, ls=":", label="faithful gate (ΔFID = 2)")
        ax.set_xticks(x_positions, x_labels, fontsize=8)
        ax.set_title(f"{cond}   (baseline FID = {base[cond]:.2f})", fontsize=11)
        if cond == CONDS[0]:
            ax.set_ylabel("ΔFID")
            from matplotlib.patches import Patch
            handles = [Patch(color=layer_colors[l], label=f"L{l}") for l in LAYERS]
            handles.append(plt.Line2D([0], [0], color="#444", ls=":", label="faithful gate (2.0)"))
            ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.92)
    fig.suptitle("Per-cell ΔFID at DiT step 200k  (5 000 samples, ODE-dopri5 250, CFG 1.5)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "delta_fid_bar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  bars → {PLOTS / 'delta_fid_bar.png'}")


def plot_vs_val_ev(base, subs):
    """Scatter ΔFID vs val EV (from E05's matryoshka aggregate) — does reconstruction
    quality on raw activations predict causal faithfulness?"""
    e05_csv = ROOT / "flywheel/sae/e05_batchtopk_vs_matryoshka/data/aggregate.csv"
    if not e05_csv.exists():
        print(f"  [skip] {e05_csv} not found")
        return
    # E05 aggregate columns: sweep, cond, layer, t_bin, dit_step, val_ev, ...
    matryo_ev = {}
    with open(e05_csv) as fh:
        for r in csv.DictReader(fh):
            if r["sweep"] != "Matryoshka k=256 d=32k":
                continue
            if int(r["dit_step"]) != 200000:
                continue
            if r["val_ev"] in ("", "None"):
                continue
            key = (r["cond"], int(r["layer"]), int(r["t_bin"]))
            matryo_ev[key] = float(r["val_ev"])

    xs, ys, cs = [], [], []
    for r in subs:
        key = (r["cond"], r["layer"], r["t_bin"])
        if key not in matryo_ev:
            continue
        xs.append(matryo_ev[key])
        ys.append(r["fid"] - base[r["cond"]])
        cs.append({"sd_vae": PB["loss"], "repa_e": PB["neutral"], "eq_vae": PB["win"]}[r["cond"]])

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for cond, col in [("sd_vae", PB["loss"]), ("repa_e", PB["neutral"]), ("eq_vae", PB["win"])]:
        m = [i for i, r in enumerate(subs) if r["cond"] == cond
             and (r["cond"], r["layer"], r["t_bin"]) in matryo_ev]
        if not m:
            continue
        ax.scatter([xs[i] for i in m], [ys[i] for i in m],
                   s=55, c=col, alpha=0.85, edgecolor="#202020", linewidth=0.5,
                   label=cond)
    ax.axhline(0, color="#444", lw=0.6)
    ax.axhline(2.0, color="#444", lw=0.6, ls=":")
    ax.set_xlabel("val EV (Matryoshka SAE, raw activations)")
    ax.set_ylabel("ΔFID (causal: SAE substitution during sampling)")
    ax.set_title("Reconstruction-EV vs causal-faithfulness  (27 cells, DiT step 200k)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92, title="condition")
    # Pearson correlation
    if len(xs) >= 3:
        cor = float(np.corrcoef(xs, ys)[0, 1])
        ax.text(0.02, 0.96, f"Pearson r = {cor:+.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="#fafafa", edgecolor="#888", alpha=0.92))
    fig.tight_layout()
    fig.savefig(PLOTS / "fid_vs_val_ev.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  fid_vs_val_ev → {PLOTS / 'fid_vs_val_ev.png'}")


def write_data(base, subs):
    # Flat CSV with delta column.
    with open(DATA / "aggregate_with_delta.csv", "w", newline="") as fh:
        cols = ["cond", "layer", "t_bin", "t_center", "fid_baseline",
                "fid_sub", "delta_fid", "classification", "hook_active"]
        w = csv.writer(fh)
        w.writerow(cols)
        for r in sorted(subs, key=lambda x: (x["cond"], x["layer"], x["t_bin"])):
            d = r["fid"] - base[r["cond"]]
            cls = "faithful" if d < 2.0 else ("marginal" if d < 5.0 else "unreliable")
            w.writerow([r["cond"], r["layer"], r["t_bin"], T_CENTERS[r["t_bin"]],
                        f"{base[r['cond']]:.4f}", f"{r['fid']:.4f}", f"{d:+.4f}",
                        cls, r["hook_stats"]["active"]])
    print(f"  csv → {DATA / 'aggregate_with_delta.csv'}")

    deltas = [r["fid"] - base[r["cond"]] for r in subs]
    head = {
        "n_cells": len(subs),
        "n_baselines": len(base),
        "baselines": {c: base[c] for c in CONDS},
        "delta_fid": {
            "mean": float(np.mean(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
            "median": float(np.median(deltas)),
        },
        "classification_counts": {
            "faithful_lt_2": int(sum(1 for d in deltas if d < 2.0)),
            "marginal_2_to_5": int(sum(1 for d in deltas if 2.0 <= d < 5.0)),
            "unreliable_ge_5": int(sum(1 for d in deltas if d >= 5.0)),
        },
        "ref_name": "imagenet_val_50k",
        "n_samples_per_cell": 5000,
        "sampler": "ode-dopri5",
        "sample_steps": 250,
        "cfg": 1.5,
        "dit_step": 200000,
        "sae_variant": "matryoshka_batchtopk",
        "sae_k": 256,
        "sae_d": 32768,
        "matryoshka_widths": [4096, 8192, 16384, 32768],
    }
    (DATA / "headline.json").write_text(json.dumps(head, indent=2))
    print(f"  headline → {DATA / 'headline.json'}")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    base, subs = split_baselines(rows)
    print(f"loaded {len(rows)} rows → {len(base)} baselines + {len(subs)} substitution cells")
    plot_heatmap(base, subs)
    plot_bars(base, subs)
    plot_vs_val_ev(base, subs)
    write_data(base, subs)


if __name__ == "__main__":
    main()
