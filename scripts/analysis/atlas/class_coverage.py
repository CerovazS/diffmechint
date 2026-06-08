"""Class coverage analysis (#2 of todo.md).

For each of 1000 ImageNet classes, compute per cell how strongly the class is
"covered" by some feature — best_count(k, cell) = max over all live features
of #(class-k images in feature's top-9).

Produces:
    outputs/phase4_8_atlas/class_coverage.npy        — 1000 x 27 int matrix
    outputs/phase4_8_atlas/class_winners.csv         — per-class winning cell
    outputs/phase4_8_atlas/plots/c4a_class_coverage_heatmap.png
    outputs/phase4_8_atlas/plots/c4b_coverage_curve.png
    outputs/phase4_8_atlas/plots/c4c_class_winners.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffmechint.analysis.atlas import CELL_RE, COND_LABEL, COND_ORDER, TBIN_ORDER

ATLAS_DEFAULT = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/phase4_8_atlas")
ATLAS = ATLAS_DEFAULT
TOP9 = ATLAS / "top9"
PLOTS = ATLAS / "plots"
TIMM_LEMMAS = Path(
    "/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/.venv/lib/python3.11/"
    "site-packages/timm/data/_info/imagenet_synset_to_lemma.txt"
)
TIMM_SYNSETS = Path(
    "/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/.venv/lib/python3.11/"
    "site-packages/timm/data/_info/imagenet_synsets.txt"
)

LAYER_ORDER = [3, 6, 9]
PB = {
    "ink": "#1c1b19", "teal": "#335C67", "amber": "#E09F3E",
    "red": "#9E2A2B", "cream": "#FFF3B0", "wine": "#540B0E",
}
COND_COLORS = {"sd_vae": PB["red"], "repa_e": PB["teal"], "eq_vae": PB["amber"]}
N_CLASSES = 1000
COVERAGE_THRESHOLD = 5  # at least 5 of 9 top-images of class k → "covered"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": PB["ink"], "axes.labelcolor": PB["ink"],
    "xtick.color": PB["ink"], "ytick.color": PB["ink"], "text.color": PB["ink"],
})


def load_class_names() -> list[str]:
    syns = [s.strip() for s in TIMM_SYNSETS.read_text().splitlines() if s.strip()]
    lemmas: dict[str, str] = {}
    for line in TIMM_LEMMAS.read_text().splitlines():
        if line.strip():
            s, lemma = line.split("\t", 1)
            lemmas[s.strip()] = lemma.strip()
    return [lemmas[s] for s in syns]


def build_coverage_matrix(cells: list[str]) -> np.ndarray:
    """For each (class k, cell c) compute the max #(class-k images in any
    live feature's top-9). Returns (1000, n_cells) int16."""
    M = np.zeros((N_CLASSES, len(cells)), dtype=np.int16)
    for j, cell in enumerate(cells):
        path = TOP9 / f"{cell}.npz"
        if not path.exists():
            print(f"  WARN: missing {path.name}", file=sys.stderr)
            continue
        z = np.load(path)
        top9_cls = z["top9_class_idx"]  # (N_features, 9), -1 = missing
        if top9_cls.size == 0:
            continue
        # Per feature, count occurrences of each class in its top-9.
        # Vectorise: for each class k, count rows where top9_cls == k.
        # Then max along features = best_count(k, cell).
        for k in range(N_CLASSES):
            counts = (top9_cls == k).sum(axis=1)  # (N_features,)
            if counts.size:
                M[k, j] = counts.max()
    return M


def plot_c4a(M: np.ndarray, cells: list[str], labels: list[str]) -> None:
    """1000 x 27 heatmap, sorted by row sum (total coverage)."""
    row_sum = M.sum(axis=1)
    order = np.argsort(-row_sum)
    M_sorted = M[order]

    fig, ax = plt.subplots(figsize=(12, 16))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "white_to_wine", ["#ffffff", PB["amber"], PB["teal"], PB["wine"]])
    im = ax.imshow(M_sorted, cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c.replace("_", "·") for c in cells], rotation=90, fontsize=7)
    # Y ticks: show every 20th class
    yticks = list(range(0, N_CLASSES, 25))
    ax.set_yticks(yticks)
    ax.set_yticklabels([labels[order[i]] for i in yticks], fontsize=6)
    for x in (8.5, 17.5):
        ax.axvline(x, color=PB["ink"], lw=1.3)
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("max # class-k images in a feature's top-9 (0..9)",
                   rotation=270, labelpad=15)
    ax.set_title("C4a — Class coverage matrix: 1 000 ImageNet classes (rows, sorted by total)\n"
                 "x 27 cells (columns). Cell value = strongest dedicated detector for that class.",
                 fontsize=11, pad=10)
    fig.tight_layout()
    out = PLOTS / "c4a_class_coverage_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_c4b(M: np.ndarray, cells: list[str]) -> None:
    """For each threshold t ∈ 1..9, count classes with best_count >= t per cell.
    3 facets, one per condition."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    thresholds = list(range(1, 10))
    for j_panel, cond in enumerate(COND_ORDER):
        ax = axes[j_panel]
        for L in LAYER_ORDER:
            for T in TBIN_ORDER:
                cell = f"{cond}_L{L}_T{T}"
                if cell not in cells:
                    continue
                col = cells.index(cell)
                counts = [int((M[:, col] >= t).sum()) for t in thresholds]
                # Style: solid lines for T2, dashed T1, dotted T0; color by layer.
                styles = [":", "--", "-"]
                palette = [PB["teal"], PB["amber"], PB["red"], PB["wine"]]
                colors = {layer: palette[i % len(palette)]
                          for i, layer in enumerate(LAYER_ORDER)}
                ax.plot(thresholds, counts, ls=styles[T], color=colors[L],
                        lw=1.8, marker="o", markersize=4,
                        label=f"L{L}·T{T}" if j_panel == 0 else None)
        ax.set_title(COND_LABEL[cond], fontsize=11)
        ax.set_xticks(thresholds)
        ax.set_xlabel("threshold (min # class-k images in top-9)")
        if j_panel == 0:
            ax.set_ylabel("# ImageNet classes covered")
        ax.grid(True, color="#cccccc", lw=0.4)
        ax.axvline(COVERAGE_THRESHOLD, color=PB["wine"], lw=1.0, ls="--", alpha=0.5)
    fig.suptitle("C4b — Class coverage curve: how many classes are covered as the "
                 "threshold relaxes?", fontsize=12)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="upper right", fontsize=7, framealpha=0.95, ncol=3)
    fig.tight_layout()
    out = PLOTS / "c4b_coverage_curve.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_c4c(M: np.ndarray, cells: list[str]) -> pd.DataFrame:
    """For each class, find the (single) winning cell at threshold=5.
    Then bar chart: how many classes won per condition."""
    rows = []
    for k in range(N_CLASSES):
        best = M[k].max()
        if best < COVERAGE_THRESHOLD:
            continue
        winners = np.where(M[k] == best)[0]
        # If multiple winners, attribute proportional fractional wins.
        for w in winners:
            cell = cells[int(w)]
            m = CELL_RE.match(cell)
            rows.append({
                "class_idx": k, "cell": cell,
                "cond": m["cond"], "layer": int(m["layer"]), "tbin": int(m["tbin"]),
                "best_count": int(best), "share": 1.0 / len(winners),
            })
    df = pd.DataFrame(rows)
    by_cond = df.groupby("cond")["share"].sum().reindex(COND_ORDER, fill_value=0)
    by_layer = df.groupby(["cond", "layer"])["share"].sum().unstack(fill_value=0).reindex(COND_ORDER)
    by_tbin = df.groupby(["cond", "tbin"])["share"].sum().unstack(fill_value=0).reindex(COND_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    # Panel 1: total per condition
    axes[0].bar(range(3), by_cond.values,
                color=[COND_COLORS[c] for c in COND_ORDER],
                edgecolor=PB["ink"], lw=0.5)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
    axes[0].set_ylabel(f"# ImageNet classes won (threshold ≥ {COVERAGE_THRESHOLD})")
    axes[0].set_title("Total classes won (of 1 000)", fontsize=11)
    for j, v in enumerate(by_cond.values):
        axes[0].text(j, v + 5, f"{v:.0f}", ha="center", fontsize=10)
    # Panel 2: breakdown per layer
    bottoms = np.zeros(3)
    palette = [PB["teal"], PB["amber"], PB["red"], PB["wine"]]
    layer_colors = {layer: palette[i % len(palette)]
                    for i, layer in enumerate(LAYER_ORDER)}
    for L in LAYER_ORDER:
        vals = by_layer[L].values if L in by_layer.columns else np.zeros(3)
        axes[1].bar(range(3), vals, bottom=bottoms,
                    color=layer_colors[L], label=f"L{L}", edgecolor=PB["ink"], lw=0.4)
        bottoms += vals
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
    axes[1].set_title("Stacked by layer", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.95)
    # Panel 3: breakdown per t_bin
    tbin_colors = {0: "#F2E089", 1: PB["amber"], 2: PB["wine"]}
    tbin_labels = {0: "T0 (quasi-clean)", 1: "T1 (mid)", 2: "T2 (high-noise)"}
    bottoms = np.zeros(3)
    for T in [0, 1, 2]:
        vals = by_tbin[T].values if T in by_tbin.columns else np.zeros(3)
        axes[2].bar(range(3), vals, bottom=bottoms,
                    color=tbin_colors[T], label=tbin_labels[T], edgecolor=PB["ink"], lw=0.4)
        bottoms += vals
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
    axes[2].set_title("Stacked by timestep", fontsize=11)
    axes[2].legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle(f"C4c — Class winners (threshold ≥ {COVERAGE_THRESHOLD} of 9 images in best feature's top-9)",
                 fontsize=12)
    fig.tight_layout()
    out = PLOTS / "c4c_class_winners.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")
    return df


def main() -> int:
    global ATLAS, TOP9, PLOTS, LAYER_ORDER
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard_root", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--atlas_root", type=Path, default=ATLAS_DEFAULT)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9])
    args = parser.parse_args()
    ATLAS = args.atlas_root
    TOP9 = ATLAS / "top9"
    PLOTS = ATLAS / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)
    LAYER_ORDER = list(args.layers)

    labels = load_class_names()
    cells = [f"{c}_L{L}_T{T}" for c in COND_ORDER for L in LAYER_ORDER for T in TBIN_ORDER]
    print(f"loading top9 npz for {len(cells)} cells...")
    M = build_coverage_matrix(cells)
    np.save(ATLAS / "class_coverage.npy", M)
    print(f"wrote class_coverage.npy  shape={M.shape}")
    plot_c4a(M, cells, labels)
    plot_c4b(M, cells)
    df_winners = plot_c4c(M, cells)
    df_winners.to_csv(ATLAS / "class_winners.csv", index=False)
    print(f"wrote class_winners.csv ({len(df_winners)} rows)")

    # Print summary stats
    print("\n=== KEY STATS ===")
    covered_any = int((M.max(axis=1) >= COVERAGE_THRESHOLD).sum())
    print(f"Classes covered by at least one cell (threshold ≥{COVERAGE_THRESHOLD}): {covered_any}/{N_CLASSES}")
    print("\nClasses covered per cell:")
    for j, cell in enumerate(cells):
        n = int((M[:, j] >= COVERAGE_THRESHOLD).sum())
        print(f"  {cell:22s}  {n:>4} / 1000")
    print("\nClasses covered by ALL 3 conditions (at threshold 5):")
    by_cond_mat = {}
    for cond in COND_ORDER:
        cols = [cells.index(f"{cond}_L{L}_T{T}") for L in LAYER_ORDER for T in TBIN_ORDER]
        by_cond_mat[cond] = M[:, cols].max(axis=1) >= COVERAGE_THRESHOLD
    intersect = by_cond_mat["sd_vae"] & by_cond_mat["repa_e"] & by_cond_mat["eq_vae"]
    print(f"  intersection (all 3): {int(intersect.sum())}/1000")
    for cond in COND_ORDER:
        print(f"  {cond} alone covers: {int(by_cond_mat[cond].sum())}/1000")

    return 0


if __name__ == "__main__":
    sys.exit(main())
