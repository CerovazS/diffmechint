"""Axis B — "Which concepts did the model learn?" — semantic interpretability.

Produces (under outputs/phase4_8_atlas/plots/):

    b0_concept_emergence.png   — top-30 concepts x 27 cells heatmap
    b1_concept_categories.png  — 5-category bar chart, 3 cond facets
    b2_wordclouds.png          — 3 panels, top-50 concepts per condition
    b3_jaccard.png             — Jaccard similarity matrix of concept sets
    b5_saturation.png          — unique concepts vs feature count (Heaps' law check)

All plots use Palette B + a CSV-derived dataframe — no GPU, no extra downloads.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ATLAS_DEFAULT = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/phase4_8_atlas")
ATLAS = ATLAS_DEFAULT
PLOTS = ATLAS / "plots"

PB = {
    "ink": "#1c1b19",
    "teal": "#335C67",
    "amber": "#E09F3E",
    "red": "#9E2A2B",
    "cream": "#FFF3B0",
    "wine": "#540B0E",
}
COND_COLORS = {"sd_vae": PB["red"], "repa_e": PB["teal"], "eq_vae": PB["amber"]}
COND_ORDER = ["sd_vae", "repa_e", "eq_vae"]
COND_LABEL = {"sd_vae": "SD-VAE", "repa_e": "REPA-E", "eq_vae": "EQ-VAE"}
LAYER_ORDER = [3, 6, 9]
TBIN_ORDER = [0, 1, 2]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": PB["ink"],
    "axes.labelcolor": PB["ink"],
    "xtick.color": PB["ink"],
    "ytick.color": PB["ink"],
    "text.color": PB["ink"],
    "axes.grid": False,
})

# Keyword classifier — caption → coarse category.
CATEGORIES: dict[str, list[str]] = {
    "animals": [
        r"\bdog", r"\bcat\b", r"\bbird", r"\bfish", r"\bsnake", r"\bspider",
        r"\binsect", r"\bbeetle", r"\bbutterfly", r"\bpanda", r"\bbear\b",
        r"\bmonkey", r"\belephant", r"\bhorse", r"\bwhale", r"\bshark",
        r"\blizard", r"\bfrog", r"\bturtle", r"\bwolf", r"\bfox\b", r"\bape\b",
        r"\blion", r"\btiger", r"\bowl\b", r"\bduck", r"\bgoose", r"\beagle",
        r"\bcrocodile", r"\bdeer", r"\brabbit", r"\bhamster", r"\bsquirrel",
        r"\bpuppy", r"\bkitten", r"\banimals?\b", r"\bmarine\b", r"\bwildlife",
        r"\bcreature", r"\bbeast", r"\bcrab", r"\bjellyfish", r"\boctopus",
        r"\bsnail", r"\bworm\b", r"\bbeak", r"\bmammals?", r"\breptile",
        r"\brodent", r"\bprimate", r"\bequine", r"\bcanine", r"\bfeline",
    ],
    "vehicles": [
        r"\bcar\b", r"\bcars\b", r"\btruck", r"\btrain", r"\bplane",
        r"\bairplane", r"\baircraft", r"\bship\b", r"\bboat", r"\bmotorcycle",
        r"\bambulance", r"\bvehicle", r"\btram", r"\bbicycle", r"\bbike\b",
        r"\btractor", r"\bbus\b", r"\btaxi", r"\bvan\b", r"\brail",
        r"\bsubmarine", r"\bscooter", r"\bjet\b", r"\bhelicopter",
        r"\blocomotive", r"\bcaravan", r"\btransport", r"\bautomobile",
    ],
    "food": [
        r"\bfood\b", r"\bbread", r"\bfruit", r"\bdessert", r"\bdrink",
        r"\bmeal\b", r"\bcake\b", r"\bpizza", r"\bsalad", r"\bcheese",
        r"\bmeat\b", r"\bsoup\b", r"\bbeverage", r"\bcuisine", r"\bvegetable",
        r"\bberry", r"\bwine\b", r"\bcoffee", r"\btea\b", r"\bsushi",
        r"\bbacon", r"\bsteak", r"\bdish\b", r"\bgrocery",
    ],
    "scenes/landscape": [
        r"\blandscape", r"\bbuilding", r"\barchitecture", r"\bmountain",
        r"\bsky\b", r"\bwater\b", r"\bforest", r"\bfield\b", r"\bocean",
        r"\bbeach", r"\bgarden", r"\bcity", r"\bvillage", r"\binterior",
        r"\boutdoor", r"\bscene\b", r"\bview\b", r"\bplace\b",
        r"\benvironment", r"\btower", r"\bbridge", r"\bchurch", r"\btemple",
        r"\bhouse\b", r"\bmosque", r"\bcastle", r"\bstreet", r"\broom\b",
        r"\bsea\b", r"\briver", r"\blake\b", r"\bdesert", r"\bsnow",
        r"\bvalley", r"\bsunset", r"\bhill\b",
    ],
    "objects/tools": [
        r"\bmachine", r"\binstrument", r"\bdevice", r"\btool\b", r"\bfurniture",
        r"\bclothing", r"\bfabric", r"\bweapon", r"\bequipment", r"\bappliance",
        r"\bcontainer", r"\bbottle", r"\bcup\b", r"\bcamera", r"\bphone",
        r"\bcomputer", r"\bkeyboard", r"\bmonitor", r"\bclock", r"\bguitar",
        r"\bpiano", r"\bumbrella", r"\bhelmet", r"\bbag\b", r"\bbasket",
        r"\bbox\b", r"\bbook\b", r"\bball\b", r"\bchair", r"\btable\b",
        r"\bbed\b", r"\bsofa", r"\blamp\b", r"\bmirror",
    ],
}
CAT_ORDER = ["animals", "vehicles", "food", "scenes/landscape", "objects/tools", "other"]
CAT_COLORS = {
    "animals": PB["teal"],
    "vehicles": PB["red"],
    "food": PB["amber"],
    "scenes/landscape": PB["wine"],
    "objects/tools": "#7a9b8f",
    "other": "#bbbbbb",
}

# Word stopper for word-cloud / B0 normalization: caption stem.
STOPWORDS = {
    "and", "or", "with", "of", "in", "the", "a", "an", "on", "at", "to",
    "for", "from", "various", "different", "kinds", "kind", "photo", "image",
    "mixed", "diverse", "types", "type",
}


def categorize(caption: str | None) -> str:
    if not caption or not isinstance(caption, str):
        return "other"
    text = caption.lower()
    for cat in CAT_ORDER[:-1]:
        for pat in CATEGORIES[cat]:
            if re.search(pat, text):
                return cat
    return "other"


def normalize_caption(c: str | None) -> str:
    """Strip articles, lowercase, keep ≤3 word stem for matching."""
    if not c or not isinstance(c, str):
        return ""
    c = c.lower().strip()
    c = re.sub(r"[^\w\s]", " ", c)
    tokens = [t for t in c.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens[:3])  # keep most-informative head


def plot_b0(df_mono: pd.DataFrame) -> None:
    """Heatmap: top-30 concepts x 27 cells.  Cell value = # mono features with
    that concept (after normalization)."""
    df = df_mono.copy()
    df["concept"] = df["vlm_interpretation"].map(normalize_caption)
    df = df[df["concept"] != ""]
    # Top-30 concepts by total frequency across all cells.
    top = df["concept"].value_counts().head(30).index.tolist()

    # Build the heatmap matrix: rows = concepts (sorted), columns = 27 cells.
    cells = []
    for c in COND_ORDER:
        for L in LAYER_ORDER:
            for T in TBIN_ORDER:
                cells.append(f"{c}_L{L}_T{T}")
    M = np.zeros((len(top), len(cells)), dtype=int)
    for i, concept in enumerate(top):
        for j, cell in enumerate(cells):
            M[i, j] = int(((df["concept"] == concept) & (df["cell"] == cell)).sum())

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "white_teal", ["#ffffff", PB["amber"], PB["teal"], PB["wine"]])
    im = ax.imshow(M, cmap=cmap, aspect="auto",
                   norm=mpl.colors.PowerNorm(gamma=0.6))
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c.replace("_", "·") for c in cells], rotation=90, fontsize=7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top, fontsize=9)
    # Vertical separators between conditions
    for x in (8.5, 17.5):
        ax.axvline(x, color=PB["ink"], lw=1.3)
    # Vertical separators between layer groups (every 3 cells)
    for x in (2.5, 5.5, 11.5, 14.5, 20.5, 23.5):
        ax.axvline(x, color=PB["ink"], lw=0.4, alpha=0.4)
    # Annotate non-zero cells with the count
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > M.max() * 0.5 else PB["ink"],
                        fontsize=6)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("# mono features (gamma=0.6)", rotation=270, labelpad=15)
    ax.set_title("B0 — Concept emergence: top-30 concepts x 27 cells\n"
                 "(rows = concept stem; columns grouped by condition / layer / timestep)",
                 fontsize=12, color=PB["ink"], pad=10)
    fig.tight_layout()
    out = PLOTS / "b0_concept_emergence.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_b1(df_mono: pd.DataFrame) -> None:
    """Concept-category bar chart: 5 categories + 'other', stacked per cell, 3 facets (one per cond)."""
    df = df_mono.copy()
    df["category"] = df["vlm_interpretation"].map(categorize)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for j, cond in enumerate(COND_ORDER):
        ax = axes[j]
        sub = df[df["cond"] == cond]
        cell_keys = [(L, T) for L in LAYER_ORDER for T in TBIN_ORDER]
        n = len(cell_keys)
        running = np.zeros(n)
        for cat in CAT_ORDER:
            counts = np.zeros(n)
            for k, (L, T) in enumerate(cell_keys):
                counts[k] = int(((sub["layer"] == L) & (sub["tbin"] == T)
                                 & (sub["category"] == cat)).sum())
            ax.bar(range(n), counts, bottom=running,
                   color=CAT_COLORS[cat], label=cat, edgecolor=PB["ink"], lw=0.4)
            running += counts
        ax.set_title(COND_LABEL[cond], fontsize=11)
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"L{L}·T{T}" for L, T in cell_keys], rotation=45, ha="right", fontsize=8)
        if j == 0:
            ax.set_ylabel("# mono features")
        if j == 2:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.95)
    fig.suptitle("B1 — Concept categories per (cond, layer, t) — stacked", fontsize=12, color=PB["ink"])
    fig.tight_layout()
    out = PLOTS / "b1_concept_categories.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_b2(df_mono: pd.DataFrame) -> None:
    """Word-frequency 'cloud' as a horizontal bar chart of top-30 stems per condition."""
    df = df_mono.copy()
    df["concept"] = df["vlm_interpretation"].map(normalize_caption)
    df = df[df["concept"] != ""]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharex=False)
    for j, cond in enumerate(COND_ORDER):
        sub = df[df["cond"] == cond]
        ctr = Counter(sub["concept"])
        top = ctr.most_common(30)
        labels = [w for w, _ in top][::-1]
        counts = [n for _, n in top][::-1]
        ax = axes[j]
        ax.barh(range(len(labels)), counts, color=COND_COLORS[cond], edgecolor=PB["ink"], lw=0.4)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("# mono features", fontsize=9)
        ax.set_title(f"{COND_LABEL[cond]}  ({sub['concept'].nunique()} unique)",
                     fontsize=11, color=COND_COLORS[cond])
    fig.suptitle("B2 — Top-30 concepts per condition (caption stems)", fontsize=12, color=PB["ink"])
    fig.tight_layout()
    out = PLOTS / "b2_wordclouds.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_b3(df_mono: pd.DataFrame) -> None:
    """Jaccard similarity matrix between concept sets at the middle layer."""
    mid_layer = LAYER_ORDER[len(LAYER_ORDER) // 2]
    df = df_mono[df_mono["layer"] == mid_layer].copy()
    df["concept"] = df["vlm_interpretation"].map(normalize_caption)
    df = df[df["concept"] != ""]
    cells = []
    sets: list[set] = []
    for c in COND_ORDER:
        for T in TBIN_ORDER:
            cells.append(f"{c}·T{T}")
            s = set(df[(df["cond"] == c) & (df["tbin"] == T)]["concept"])
            sets.append(s)
    n = len(cells)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = sets[i], sets[j]
            if not a and not b:
                M[i, j] = 0.0
            elif not (a | b):
                M[i, j] = 0.0
            else:
                M[i, j] = len(a & b) / max(1, len(a | b))

    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cream_wine", ["#ffffff", PB["amber"], PB["red"], PB["wine"]])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(0.1, float(M[M < 1].max() if (M < 1).any() else 0.1)))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cells, fontsize=9)
    # block separators between conditions
    for x in (2.5, 5.5):
        ax.axvline(x, color=PB["ink"], lw=1.4)
        ax.axhline(x, color=PB["ink"], lw=1.4)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if v > 0.3 else PB["ink"])
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Jaccard similarity", rotation=270, labelpad=14)
    ax.set_title(f"B3 — Concept-set Jaccard similarity at L={mid_layer}\n"
                 "(diagonal blocks = within-condition; off-diagonal = across-condition)",
                 fontsize=11, color=PB["ink"], pad=10)
    fig.tight_layout()
    out = PLOTS / "b3_jaccard.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_b5(df_mono: pd.DataFrame) -> None:
    """Saturation: cumulative unique concepts vs cumulative mono feature count, per condition."""
    df = df_mono.copy()
    df["concept"] = df["vlm_interpretation"].map(normalize_caption)
    df = df[df["concept"] != ""]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cond in COND_ORDER:
        sub = df[df["cond"] == cond].sample(frac=1.0, random_state=0).reset_index(drop=True)
        seen: set = set()
        unique_at_step = []
        for c in sub["concept"]:
            seen.add(c)
            unique_at_step.append(len(seen))
        x = np.arange(1, len(sub) + 1)
        ax.plot(x, unique_at_step, lw=2.4, color=COND_COLORS[cond],
                label=f"{COND_LABEL[cond]}  (mono={len(sub)}, unique={len(seen)})")
    ax.set_xlabel("# mono features sampled (cumulative)")
    ax.set_ylabel("# unique concept stems discovered")
    ax.set_title("B5 — Concept-vocabulary saturation curve (Heaps' law check)",
                 fontsize=12, color=PB["ink"], pad=10)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, color="#cccccc", lw=0.4)
    fig.tight_layout()
    out = PLOTS / "b5_saturation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> int:
    global ATLAS, PLOTS, LAYER_ORDER
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard_root", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--atlas_root", type=Path, default=ATLAS_DEFAULT)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9])
    args = parser.parse_args()
    ATLAS = args.atlas_root
    PLOTS = ATLAS / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)
    LAYER_ORDER = list(args.layers)

    df_mono = pd.read_csv(ATLAS / "atlas_mono_features.csv")
    df_mono = df_mono[df_mono["vlm_interpretation"].notna()]
    print(f"loaded atlas_mono_features.csv  ({len(df_mono)} rows with VLM)")
    plot_b0(df_mono)
    plot_b1(df_mono)
    plot_b2(df_mono)
    plot_b3(df_mono)
    plot_b5(df_mono)
    return 0


if __name__ == "__main__":
    sys.exit(main())
