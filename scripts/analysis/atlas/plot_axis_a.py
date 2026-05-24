"""Axis A — "Where does semantics live?" — geometry of the diffusion residual.

Produces (under outputs/phase4_8_atlas/plots/):

    a1_mono_heatmap.png        — 3×9 heatmap of mono_count per (cond, L, T)
    a2_mono_vs_tbin.png        — mono_count vs t_bin, 3 panels (one per layer)
    a3_mono_vs_layer.png       — mono_count vs layer, 3 panels (one per t_bin)
    a4_entropy_boxplot.png     — entropy of all LIVE features, 27 boxes
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ATLAS = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/phase4_8_atlas")
PLOTS = ATLAS / "plots"

PB = {
    "bg": "#FFF3B0",
    "ink": "#1c1b19",
    "teal": "#335C67",
    "amber": "#E09F3E",
    "red": "#9E2A2B",
    "cream": "#FFF3B0",
    "wine": "#540B0E",
}
COND_COLORS = {"sd_vae": PB["red"], "repa_e": PB["teal"], "eq_vae": PB["amber"]}
COND_ORDER = ["sd_vae", "repa_e", "eq_vae"]
LAYER_ORDER = [3, 6, 9]
TBIN_ORDER = [0, 1, 2]
T_LABELS = {0: "T0 = 0.025", 1: "T1 = 0.20", 2: "T2 = 0.50"}
COND_LABEL = {"sd_vae": "SD-VAE", "repa_e": "REPA-E", "eq_vae": "EQ-VAE"}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": PB["ink"],
    "axes.labelcolor": PB["ink"],
    "xtick.color": PB["ink"],
    "ytick.color": PB["ink"],
    "text.color": PB["ink"],
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.5,
})


def plot_a1(df_cell: pd.DataFrame) -> None:
    """3×9 heatmap of mono_count.  Rows = condition; columns = 9 (layer,t_bin)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # 3×9 matrix
    M = np.zeros((3, 9))
    col_labels = []
    for j_layer, L in enumerate(LAYER_ORDER):
        for k_tbin, T in enumerate(TBIN_ORDER):
            col_labels.append(f"L{L}·T{T}")
            for i_cond, cond in enumerate(COND_ORDER):
                row = df_cell[(df_cell["cond"] == cond) & (df_cell["layer"] == L)
                              & (df_cell["tbin"] == T)]
                M[i_cond, j_layer * 3 + k_tbin] = row["mono_count_recomputed"].iloc[0]

    # Custom colormap: cream → teal (low to high)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cream_teal", [PB["cream"], PB["amber"], PB["teal"], PB["wine"]])
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(9))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(range(3))
    ax.set_yticklabels([COND_LABEL[c] for c in COND_ORDER])
    # Annotate each cell with the count
    for i in range(3):
        for j in range(9):
            v = int(M[i, j])
            color = "white" if M[i, j] > M.max() * 0.55 else PB["ink"]
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=11)
    # Vertical separators between layer groups
    for x in (2.5, 5.5):
        ax.axvline(x, color=PB["ink"], lw=1.2)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("mono feature count", rotation=270, labelpad=15)
    ax.set_title("A1 — Monosemantic feature count per cell (y-null counterfactual)",
                 fontsize=12, color=PB["ink"], pad=10)
    ax.set_xlabel("layer · timestep")
    fig.tight_layout()
    out = PLOTS / "a1_mono_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_a2(df_cell: pd.DataFrame) -> None:
    """3 sub-panels (one per layer), each plots mono_count vs t_bin for 3 conds."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for j, L in enumerate(LAYER_ORDER):
        ax = axes[j]
        for cond in COND_ORDER:
            sub = df_cell[(df_cell["cond"] == cond) & (df_cell["layer"] == L)].sort_values("tbin")
            ax.plot(sub["tbin"], sub["mono_count_recomputed"],
                    marker="o", lw=2.2, markersize=8,
                    label=COND_LABEL[cond], color=COND_COLORS[cond])
        ax.set_title(f"Layer L{L}", fontsize=11)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["T0\n0.025", "T1\n0.20", "T2\n0.50"])
        ax.set_xlabel("timestep")
        if j == 0:
            ax.set_ylabel("mono feature count")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle("A2 — Monosemantic count vs timestep (per layer)",
                 fontsize=12, color=PB["ink"])
    fig.tight_layout()
    out = PLOTS / "a2_mono_vs_tbin.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_a3(df_cell: pd.DataFrame) -> None:
    """3 sub-panels (one per t_bin), each plots mono_count vs layer for 3 conds."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for j, T in enumerate(TBIN_ORDER):
        ax = axes[j]
        for cond in COND_ORDER:
            sub = df_cell[(df_cell["cond"] == cond) & (df_cell["tbin"] == T)].sort_values("layer")
            ax.plot(sub["layer"], sub["mono_count_recomputed"],
                    marker="s", lw=2.2, markersize=8,
                    label=COND_LABEL[cond], color=COND_COLORS[cond])
        ax.set_title(T_LABELS[T], fontsize=11)
        ax.set_xticks(LAYER_ORDER)
        ax.set_xlabel("layer (DiT block index)")
        if j == 0:
            ax.set_ylabel("mono feature count")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle("A3 — Monosemantic count vs layer (per timestep)",
                 fontsize=12, color=PB["ink"])
    fig.tight_layout()
    out = PLOTS / "a3_mono_vs_layer.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_a4(df_cell: pd.DataFrame) -> None:
    """Boxplot of entropy distributions across all LIVE features, 27 boxes
    grouped into 3 facets by condition.  Lower entropy = cleaner features."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for j, cond in enumerate(COND_ORDER):
        ax = axes[j]
        cell_names = []
        data = []
        positions = []
        colors = []
        pos = 0
        for L in LAYER_ORDER:
            for T in TBIN_ORDER:
                cell = f"{cond}_L{L}_T{T}"
                arr_path = ATLAS / "entropy_dists" / f"{cell}.npy"
                if not arr_path.exists():
                    continue
                arr = np.load(arr_path)
                data.append(arr)
                cell_names.append(f"L{L}·T{T}")
                positions.append(pos)
                # Darken color slightly with t_bin to indicate noise level
                shades = [PB["amber"], PB["red"], PB["wine"]]
                colors.append(shades[T])
                pos += 1
            pos += 0.4  # gap between layer groups
        bp = ax.boxplot(data, positions=positions, widths=0.7,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": PB["ink"], "linewidth": 1.3},
                        whiskerprops={"color": PB["ink"]},
                        capprops={"color": PB["ink"]})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(PB["ink"])
        ax.axhline(2.5, ls="--", lw=1.2, color=PB["teal"], alpha=0.7,
                   label="mono threshold (2.5 bit)")
        ax.set_xticks(positions)
        ax.set_xticklabels(cell_names, rotation=45, ha="right", fontsize=8)
        ax.set_title(COND_LABEL[cond], fontsize=11)
        if j == 0:
            ax.set_ylabel("entropy (bits) over top-9 classes")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("A4 — Per-cell entropy distribution over live features  "
                 "(below 2.5 bit → monosemantic)", fontsize=12, color=PB["ink"])
    fig.tight_layout()
    out = PLOTS / "a4_entropy_boxplot.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> int:
    df_cell = pd.read_csv(ATLAS / "atlas_per_cell.csv")
    print(f"loaded atlas_per_cell.csv ({len(df_cell)} rows)")
    plot_a1(df_cell)
    plot_a2(df_cell)
    plot_a3(df_cell)
    plot_a4(df_cell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
