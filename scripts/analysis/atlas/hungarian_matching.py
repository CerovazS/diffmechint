"""Hungarian matching cross-cell (#3 of todo.md).

For curated pairs of cells, compute the optimal one-to-one assignment between
their MONO features under the cost metric:
    C[i, j] = 1 - Jaccard(top9_dataset_idx(i_a), top9_dataset_idx(j_b))

The dataset_idx are universal indices into the shared ImageNet val set, so this
cost is mathematically well-defined across conditions.

Curated pairs (54 total):
- 27 cross-condition same-(L, T)
- 9 within-condition cross-layer same-t
- 18 within-condition cross-timestep same-L (only L=6 and L=9, T1 vs T2)

Outputs:
    outputs/phase4_8_atlas/hungarian_summary.csv
    outputs/phase4_8_atlas/plots/h1_cost_heatmap.png        (27 x 27 lower-tri)
    outputs/phase4_8_atlas/plots/h2_cost_histogram.png      (headline pair)
    outputs/phase4_8_atlas/plots/h3_top_matches.png         (top-10 best matches example)
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations, pairwise
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from diffmechint.analysis.atlas import COND_ORDER, TBIN_ORDER

ATLAS_DEFAULT = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/phase4_8_atlas")
ATLAS = ATLAS_DEFAULT
TOP9 = ATLAS / "top9"
PLOTS = ATLAS / "plots"

LAYER_ORDER = [3, 6, 9]
ALL_CELLS = [f"{c}_L{L}_T{T}"
             for c in COND_ORDER for L in LAYER_ORDER for T in TBIN_ORDER]
PB = {
    "ink": "#1c1b19", "teal": "#335C67", "amber": "#E09F3E",
    "red": "#9E2A2B", "cream": "#FFF3B0", "wine": "#540B0E",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": PB["ink"], "axes.labelcolor": PB["ink"],
    "xtick.color": PB["ink"], "ytick.color": PB["ink"], "text.color": PB["ink"],
})


def load_mono_top9(cell: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (mono_feature_ids, top9_dataset_idx_for_mono_only)."""
    z = np.load(TOP9 / f"{cell}.npz")
    mask = z["is_mono"]
    return z["feature_ids"][mask], z["top9_dataset_idx"][mask]


def jaccard_cost_matrix(top9_a: np.ndarray, top9_b: np.ndarray) -> np.ndarray:
    """For each pair (i, j) compute 1 - Jaccard(top9_a[i], top9_b[j]).

    top9 arrays are (N, 9) int32. We treat each row as a SET (ignoring -1 padding).
    Vectorised: convert to set membership tensor for speed on moderate sizes.
    """
    na, nb = top9_a.shape[0], top9_b.shape[0]
    if na == 0 or nb == 0:
        return np.empty((na, nb), dtype=np.float32)

    # All possible dataset indices that appear in either side. Build dense
    # incidence matrices: A[i, k] = 1 iff k in top9_a[i].
    union_indices = np.unique(np.concatenate([top9_a.ravel(), top9_b.ravel()]))
    union_indices = union_indices[union_indices >= 0]
    if union_indices.size == 0:
        return np.ones((na, nb), dtype=np.float32)
    idx_to_pos = {int(v): i for i, v in enumerate(union_indices)}
    n_universe = union_indices.size

    A = np.zeros((na, n_universe), dtype=bool)
    for i in range(na):
        for v in top9_a[i]:
            if v >= 0:
                A[i, idx_to_pos[int(v)]] = True
    B = np.zeros((nb, n_universe), dtype=bool)
    for i in range(nb):
        for v in top9_b[i]:
            if v >= 0:
                B[i, idx_to_pos[int(v)]] = True

    inter = (A.astype(np.uint8) @ B.T.astype(np.uint8)).astype(np.float32)
    sizes_a = A.sum(axis=1).astype(np.float32)  # (na,)
    sizes_b = B.sum(axis=1).astype(np.float32)  # (nb,)
    union = sizes_a[:, None] + sizes_b[None, :] - inter
    jaccard = np.where(union > 0, inter / union, 0.0)
    return (1.0 - jaccard).astype(np.float32)


def curated_pairs() -> list[tuple[str, str, str]]:
    """Return list of (cell_a, cell_b, kind) for the 54 curated pairs."""
    pairs: list[tuple[str, str, str]] = []
    # 1. Cross-condition same (L, T): 27 pairs
    for L in LAYER_ORDER:
        for T in TBIN_ORDER:
            for c1, c2 in combinations(COND_ORDER, 2):
                pairs.append((f"{c1}_L{L}_T{T}", f"{c2}_L{L}_T{T}", "cross_cond_same_LT"))
    # 2. Within-condition cross-layer same-t: adjacent layer pairs.
    for c in COND_ORDER:
        for T in TBIN_ORDER:
            for l_a, l_b in pairwise(LAYER_ORDER):
                pairs.append((f"{c}_L{l_a}_T{T}", f"{c}_L{l_b}_T{T}",
                              "within_cond_cross_layer"))
    # 3. Within-condition cross-timestep same-L: T1 vs T2 only
    #    (T0 has too few mono features to be informative)
    for c in COND_ORDER:
        for L in LAYER_ORDER:
            pairs.append((f"{c}_L{L}_T1", f"{c}_L{L}_T2", "within_cond_cross_timestep"))
    return pairs


def hungarian_pair(cell_a: str, cell_b: str) -> dict:
    fids_a, top9_a = load_mono_top9(cell_a)
    fids_b, top9_b = load_mono_top9(cell_b)
    na, nb = len(fids_a), len(fids_b)
    if na == 0 or nb == 0:
        return {
            "cell_a": cell_a, "cell_b": cell_b, "n_a": na, "n_b": nb,
            "mean_cost": np.nan, "median_cost": np.nan,
            "frac_good_match": np.nan, "min_cost": np.nan,
        }
    C = jaccard_cost_matrix(top9_a, top9_b)
    row_ind, col_ind = linear_sum_assignment(C)
    match_costs = C[row_ind, col_ind]
    return {
        "cell_a": cell_a, "cell_b": cell_b, "n_a": na, "n_b": nb,
        "mean_cost": float(match_costs.mean()),
        "median_cost": float(np.median(match_costs)),
        "min_cost": float(match_costs.min()),
        "frac_good_match": float((match_costs < 0.7).mean()),
        "_row_ind": row_ind, "_col_ind": col_ind, "_costs": match_costs,
        "_fids_a": fids_a, "_fids_b": fids_b,
    }


def plot_h1(df: pd.DataFrame) -> None:
    """27 x 27 lower-triangular heatmap of mean Hungarian cost.

    Only curated pairs are filled; rest is NaN/blank.
    """
    n = len(ALL_CELLS)
    cell_to_idx = {c: i for i, c in enumerate(ALL_CELLS)}
    M = np.full((n, n), np.nan)
    for _, r in df.iterrows():
        i, j = cell_to_idx[r["cell_a"]], cell_to_idx[r["cell_b"]]
        # Place value at lower triangle: M[max(i,j), min(i,j)]
        M[max(i, j), min(i, j)] = r["mean_cost"]

    fig, ax = plt.subplots(figsize=(12, 11))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "good_to_bad", [PB["teal"], PB["amber"], PB["red"], PB["wine"]])
    cmap.set_bad("#f5f5f0")
    im = ax.imshow(M, cmap=cmap, vmin=0.5, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([c.replace("_", "·") for c in ALL_CELLS], rotation=90, fontsize=7)
    ax.set_yticklabels([c.replace("_", "·") for c in ALL_CELLS], fontsize=7)
    for x in (8.5, 17.5):
        ax.axvline(x, color=PB["ink"], lw=1.0, alpha=0.4)
        ax.axhline(x, color=PB["ink"], lw=1.0, alpha=0.4)
    # Annotate non-NaN cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if v > 0.78 else PB["ink"])
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("mean Hungarian match cost (1 - Jaccard on top-9 images)",
                   rotation=270, labelpad=14)
    ax.set_title("H1 — Hungarian match cost between cell pairs\n"
                 "(lower = more shared concept structure; 0.9-1.0 = essentially no overlap)",
                 fontsize=11, pad=10)
    fig.tight_layout()
    out = PLOTS / "h1_cost_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_h2(headline_pair: dict) -> None:
    """Histogram of match costs for one headline pair."""
    costs = headline_pair["_costs"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _, bins, patches = ax.hist(costs, bins=30, color=PB["amber"],
                                edgecolor=PB["ink"], lw=0.5)
    # Color the "good match" bin in teal
    for patch, bin_low in zip(patches, bins[:-1], strict=False):
        if bin_low < 0.7:
            patch.set_facecolor(PB["teal"])
        elif bin_low > 0.9:
            patch.set_facecolor(PB["wine"])
    ax.axvline(0.7, color=PB["teal"], lw=1.5, ls="--", alpha=0.6, label="good match threshold")
    ax.set_xlabel("Hungarian match cost (1 - Jaccard)")
    ax.set_ylabel("# matched feature pairs")
    ax.set_title(f"H2 — Cost distribution for headline pair: "
                 f"{headline_pair['cell_a']}  ↔  {headline_pair['cell_b']}\n"
                 f"n={len(costs)}  ·  mean={costs.mean():.3f}  ·  "
                 f"frac<0.7 = {(costs < 0.7).mean():.1%}",
                 fontsize=11, pad=10)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, color="#cccccc", lw=0.4)
    fig.tight_layout()
    out = PLOTS / "h2_cost_histogram.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_h3(headline_pair: dict, df_captions: pd.DataFrame) -> pd.DataFrame:
    """Top-10 best matches example table (rendered as PNG + saved as CSV)."""
    costs = headline_pair["_costs"]
    row_ind, col_ind = headline_pair["_row_ind"], headline_pair["_col_ind"]
    order = np.argsort(costs)[:10]
    cell_a, cell_b = headline_pair["cell_a"], headline_pair["cell_b"]
    fids_a = headline_pair["_fids_a"][row_ind]
    fids_b = headline_pair["_fids_b"][col_ind]

    cap_a = df_captions[df_captions["cell"] == cell_a].set_index("feature_id")["vlm_interpretation"]
    cap_b = df_captions[df_captions["cell"] == cell_b].set_index("feature_id")["vlm_interpretation"]

    rows = []
    for rank, k in enumerate(order):
        fa, fb, cost = int(fids_a[k]), int(fids_b[k]), float(costs[k])
        rows.append({
            "rank": rank + 1, "cost": cost,
            "feature_a": fa,
            "caption_a": cap_a.get(fa, ""),
            "feature_b": fb,
            "caption_b": cap_b.get(fb, ""),
        })
    df_top = pd.DataFrame(rows)
    df_top.to_csv(ATLAS / "hungarian_top10_matches.csv", index=False)

    # Render as figure
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")
    cell_text = []
    for r in rows:
        cell_text.append([
            str(r["rank"]),
            f"{r['cost']:.3f}",
            f"#{r['feature_a']}",
            (r["caption_a"] or "")[:38],
            f"#{r['feature_b']}",
            (r["caption_b"] or "")[:38],
        ])
    table = ax.table(cellText=cell_text,
                     colLabels=["rank", "cost", f"{cell_a} feat", "caption (A)",
                                f"{cell_b} feat", "caption (B)"],
                     cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    # Color header
    for col in range(6):
        cell = table[0, col]
        cell.set_facecolor(PB["teal"])
        cell.set_text_props(color="white", weight="bold")
    # Color rank 1-3 cells
    for r in range(1, min(4, len(rows) + 1)):
        for col in range(6):
            table[r, col].set_facecolor("#e0ecef")
    ax.set_title(f"H3 — Top-10 best Hungarian matches  ·  {cell_a}  ↔  {cell_b}",
                 fontsize=12, pad=20, color=PB["ink"])
    fig.tight_layout()
    out = PLOTS / "h3_top_matches.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")
    return df_top


def main() -> int:
    global ATLAS, TOP9, PLOTS, LAYER_ORDER, ALL_CELLS
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
    ALL_CELLS = [f"{c}_L{L}_T{T}"
                 for c in COND_ORDER for L in LAYER_ORDER for T in TBIN_ORDER]

    pairs = curated_pairs()
    print(f"Computing Hungarian matching on {len(pairs)} curated pairs...")
    results = []
    for k, (a, b, kind) in enumerate(pairs):
        try:
            r = hungarian_pair(a, b)
            r["kind"] = kind
            results.append(r)
            if k % 10 == 0 or k == len(pairs) - 1:
                print(f"  [{k+1:2d}/{len(pairs)}] {a} ↔ {b}  "
                      f"n=({r['n_a']},{r['n_b']})  mean={r['mean_cost']:.3f}")
        except Exception as e:
            print(f"  [{k+1:2d}/{len(pairs)}] {a} ↔ {b}  FAILED: {e}", file=sys.stderr)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results])
    df.to_csv(ATLAS / "hungarian_summary.csv", index=False)
    print(f"\nwrote hungarian_summary.csv ({len(df)} rows)")

    # Summary stats
    print("\n=== KEY STATS ===")
    for kind in ["cross_cond_same_LT", "within_cond_cross_layer", "within_cond_cross_timestep"]:
        sub = df[df["kind"] == kind]
        valid = sub["mean_cost"].dropna()
        if len(valid):
            print(f"  {kind:30s}  n={len(valid):>2}  "
                  f"mean_cost={valid.mean():.3f}  median={valid.median():.3f}  "
                  f"frac<0.7={(valid < 0.7).mean():.1%}")

    # Plot H1
    plot_h1(df)

    # Choose headline pair: best mean cost from cross-cond pairs.
    cross_cond = df[df["kind"] == "cross_cond_same_LT"].dropna(subset=["mean_cost"])
    if cross_cond.empty:
        print("no cross-cond pairs", file=sys.stderr)
        return 1
    headline_idx = cross_cond["mean_cost"].idxmin()
    headline = results[headline_idx]
    print(f"\nHeadline pair: {headline['cell_a']} ↔ {headline['cell_b']}  "
          f"mean_cost={headline['mean_cost']:.3f}")
    plot_h2(headline)

    # Plot H3: need captions from atlas_mono_features.csv
    df_cap = pd.read_csv(ATLAS / "atlas_mono_features.csv",
                         usecols=["cell", "feature_id", "vlm_interpretation"])
    df_top10 = plot_h3(headline, df_cap)
    print("\n=== TOP-10 BEST MATCHES (headline pair) ===")
    print(df_top10.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
