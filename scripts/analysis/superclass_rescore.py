"""E34b — superclass-entropy re-scoring of latent and DiT SAE dictionaries.

Re-scores already-saved top-9 class lists at coarser semantic granularities
(WordNet hypernym superclasses) to test whether features that look polysemantic
over 1000 fine ImageNet labels become monosemantic over coarser groups. Applies
the *identical* metric to the latent-token SAE atlases (p=2 and p=4) and to the
SiT-B/2 y-null DiT cells, so the latent/DiT ratio — not the absolute count — is
the result. No model evaluation: pure re-scoring of existing artifacts. CPU-only.

Outputs (under --out_dir):
    wordnet_superclass_map.json   fine class_idx -> {medium, coarse} group ids
    superclass_mono_counts.csv    granularity x threshold x source x condition
    superclass_latent_vs_dit.png  comparison plot
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log2
from pathlib import Path

import numpy as np

from diffmechint.utils import info, ok, warn

IMAGEFOLDER_ROOT = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/imagenet_val_imagefolder"
)
DIT_VIZ_ROOT = Path(
    "/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/phase4_5b_feature_viz_ynull"
)
LATENT_P2 = Path(
    "outputs/phase4_20_latent_feature_atlas_20260606_221917/metrics"
)
LATENT_P4 = Path(
    "outputs/phase4_22_latent_atlas_p4_20260606_224945/metrics"
)
CONDITIONS = ["sd_vae", "eq_vae", "repa_e"]

# Same monosemanticity band as E07/E11/E31 (density window unchanged across
# granularities; only the entropy threshold and label resolution vary).
DENSITY_MIN = 1e-4
DENSITY_MAX = 0.10
THRESHOLDS = [2.5, 1.5]

# Coarse anchors: WordNet synsets covering the standard ImageNet high-level
# groups. Each fine class maps to the FIRST anchor that is one of its ancestors
# (priority order, specific before general); unmatched -> "other".
COARSE_ANCHORS = [
    ("dog", "dog.n.01"),
    ("cat", "feline.n.01"),
    ("primate", "primate.n.02"),
    ("ungulate", "ungulate.n.01"),
    ("rodent", "rodent.n.01"),
    ("marine_mammal", "aquatic_mammal.n.01"),
    ("mammal_other", "mammal.n.01"),
    ("bird", "bird.n.01"),
    ("reptile", "reptile.n.01"),
    ("amphibian", "amphibian.n.03"),
    ("fish", "fish.n.01"),
    ("insect", "insect.n.01"),
    ("invertebrate_other", "invertebrate.n.01"),
    ("plant", "plant.n.02"),
    ("food", "food.n.01"),
    ("vehicle", "vehicle.n.01"),
    ("clothing", "clothing.n.01"),
    ("container", "container.n.01"),
    ("furniture", "furniture.n.01"),
    ("musical_instrument", "musical_instrument.n.01"),
    ("device", "device.n.01"),
    ("structure", "structure.n.01"),
    ("covering", "covering.n.01"),
    ("instrumentality_other", "instrumentality.n.03"),
    ("artifact_other", "artifact.n.01"),
    ("geological", "geological_formation.n.01"),
]


# ----------------------------------------------------------------------------
# WordNet fine -> coarse maps
# ----------------------------------------------------------------------------
def synsets_in_classidx_order() -> list:
    """1000 WordNet synsets in ImageFolder class_idx order (sorted dir names)."""
    from nltk.corpus import wordnet as wn

    dirs = sorted(p.name for p in IMAGEFOLDER_ROOT.iterdir() if p.is_dir())
    if len(dirs) != 1000:
        raise RuntimeError(f"expected 1000 synset dirs, got {len(dirs)}")
    out = []
    for name in dirs:
        offset = int(name[1:])
        out.append(wn.synset_from_pos_and_offset("n", offset))
    return out


def build_maps(synsets: list) -> tuple[list[str], list[str], dict]:
    """Return (medium_group[1000], coarse_group[1000], meta).

    Medium: ancestor on the primary hypernym path at a fixed depth from the
    root, depth chosen to land nearest ~100 groups (reported in meta).
    Coarse: first matching anchor in COARSE_ANCHORS priority order, else 'other'.
    """
    from nltk.corpus import wordnet as wn

    anchor_synsets = [(name, wn.synset(s)) for name, s in COARSE_ANCHORS]

    coarse = []
    for s in synsets:
        ancestors = {a for path in s.hypernym_paths() for a in path}
        label = "other"
        for name, anc in anchor_synsets:
            if anc in ancestors:
                label = name
                break
        coarse.append(label)

    # Medium cut: sweep depth, pick the one closest to 100 distinct groups.
    best_depth, best_groups, best_diff = None, None, None
    for depth in range(5, 13):
        groups = []
        for s in synsets:
            path = s.hypernym_paths()[0]  # root -> synset
            node = path[min(depth, len(path) - 1)]
            groups.append(node.name())
        ndistinct = len(set(groups))
        diff = abs(ndistinct - 100)
        if best_diff is None or diff < best_diff:
            best_depth, best_groups, best_diff = depth, groups, ndistinct
    medium = best_groups

    meta = {
        "medium_depth": best_depth,
        "medium_n_groups": len(set(medium)),
        "coarse_n_groups": len(set(coarse)),
        "coarse_anchors": [n for n, _ in COARSE_ANCHORS],
    }
    return medium, coarse, meta


# ----------------------------------------------------------------------------
# entropy
# ----------------------------------------------------------------------------
def entropy_bits(group_ids: list) -> float:
    c = Counter(g for g in group_ids if g is not None)
    n = sum(c.values())
    if n == 0:
        return 0.0
    return -sum((v / n) * log2(v / n) for v in c.values())


def score_source(
    densities: np.ndarray,
    top9_classidx: np.ndarray,
    medium: list[str],
    coarse: list[str],
) -> dict:
    """Count monosemantic features per (granularity, threshold).

    densities: (N,) per-feature token density (live features only).
    top9_classidx: (N, 9) int, -1 for missing slots.
    Returns dict keyed 'fine'/'medium'/'coarse' -> {thr: count} plus 'live'.
    """
    band = (densities >= DENSITY_MIN) & (densities <= DENSITY_MAX)
    out = {"live": len(densities), "in_band": int(band.sum())}
    for gran, mapper in (("fine", None), ("medium", medium), ("coarse", coarse)):
        ents = np.empty(len(densities), dtype=np.float64)
        for i in range(len(densities)):
            cids = [int(c) for c in top9_classidx[i] if c >= 0]
            if mapper is None:
                ents[i] = entropy_bits(cids)
            else:
                ents[i] = entropy_bits([mapper[c] for c in cids])
        for thr in THRESHOLDS:
            mono = band & (ents < thr)
            out[f"{gran}@{thr}"] = int(mono.sum())
    return out


# ----------------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------------
def load_latent(metrics_dir: Path, cond: str) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader((metrics_dir / f"latent_atlas_{cond}.csv").open()))
    dens, top9 = [], []
    for r in rows:
        if r["live"] != "1":
            continue
        dens.append(float(r["density"]))
        labs = [int(x) for x in r["top9_labels"].split()]
        labs = (labs + [-1] * 9)[:9]
        top9.append(labs)
    return np.asarray(dens), np.asarray(top9, dtype=np.int64)


def _read_dit_cell(cell_dir_str: str) -> tuple[list[float], list[list[int]]]:
    cell_dir = Path(cell_dir_str)
    dens, top9 = [], []
    for f in sorted((cell_dir / "features").iterdir()):
        if not (f.name.startswith("feature_") and f.name.endswith(".json")):
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not d.get("live"):
            continue
        dens.append(float(d.get("density", 0.0)))
        cls = [int(t.get("class_idx", -1)) for t in (d.get("top") or [])[:9]]
        cls = (cls + [-1] * 9)[:9]
        top9.append(cls)
    return dens, top9


def load_dit_cells(cond: str, layers: list[int], tbin: int) -> tuple[np.ndarray, np.ndarray]:
    """Pool all (layer) cells of a condition at the given t-bin."""
    cell_dirs = [DIT_VIZ_ROOT / f"{cond}_L{layer}_T{tbin}" for layer in layers]
    cell_dirs = [str(c) for c in cell_dirs if c.is_dir()]
    dens_all, top9_all = [], []
    with ProcessPoolExecutor(max_workers=min(8, len(cell_dirs) or 1)) as ex:
        for fut in as_completed([ex.submit(_read_dit_cell, c) for c in cell_dirs]):
            dens, top9 = fut.result()
            dens_all.extend(dens)
            top9_all.extend(top9)
    return np.asarray(dens_all), np.asarray(top9_all, dtype=np.int64)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--dit_layers", type=int, nargs="+", default=[3, 6, 9])
    ap.add_argument("--dit_tbin", type=int, default=2, help="t-bin for DiT cells (2 = most data-like).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.out_dir / "metrics"
    plots_dir = args.out_dir / "plots"
    metrics_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    info("building WordNet fine->coarse maps...")
    synsets = synsets_in_classidx_order()
    medium, coarse, meta = build_maps(synsets)
    ok(f"medium: {meta['medium_n_groups']} groups (depth {meta['medium_depth']}); "
       f"coarse: {meta['coarse_n_groups']} groups")
    (metrics_dir / "wordnet_superclass_map.json").write_text(json.dumps({
        "meta": meta,
        "synset_classidx_order": [s.name() for s in synsets],
        "medium_group": medium,
        "coarse_group": coarse,
    }, indent=1))

    sources = []  # (source, patch, cond, loader)
    for cond in args.conditions:
        sources.append(("latent", "p2", cond, lambda c=cond: load_latent(LATENT_P2, c)))
        sources.append(("latent", "p4", cond, lambda c=cond: load_latent(LATENT_P4, c)))
        sources.append(("dit", "p2", cond, lambda c=cond: load_dit_cells(c, args.dit_layers, args.dit_tbin)))

    rows = []
    for source, patch, cond, loader in sources:
        try:
            dens, top9 = loader()
        except FileNotFoundError as e:
            warn(f"skip {source}/{patch}/{cond}: {e}")
            continue
        if len(dens) == 0:
            warn(f"skip {source}/{patch}/{cond}: no live features")
            continue
        sc = score_source(dens, top9, medium, coarse)
        info(f"{source:6s} {patch} {cond:7s}  live={sc['live']:6d} band={sc['in_band']:4d}  "
             f"fine@2.5={sc['fine@2.5']:3d}  med@2.5={sc['medium@2.5']:3d}  "
             f"coarse@2.5={sc['coarse@2.5']:3d}  coarse@1.5={sc['coarse@1.5']:3d}")
        for gran in ("fine", "medium", "coarse"):
            for thr in THRESHOLDS:
                rows.append({
                    "source": source, "patch": patch, "condition": cond,
                    "granularity": gran, "threshold": thr,
                    "live": sc["live"], "in_band": sc["in_band"],
                    "mono_count": sc[f"{gran}@{thr}"],
                    "mono_pct_of_live": 100.0 * sc[f"{gran}@{thr}"] / sc["live"],
                })

    csv_path = metrics_dir / "superclass_mono_counts.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok(f"wrote {csv_path} ({len(rows)} rows)")

    _plot(rows, plots_dir / "superclass_latent_vs_dit.png", meta)
    ok("done")


def _plot(rows: list[dict], out_path: Path, meta: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PB = ["#335C67", "#E09F3E", "#9E2A2B", "#540B0E", "#FFF3B0"]
    grans = ["fine", "medium", "coarse"]
    # Aggregate over conditions: mono_pct_of_live summed counts / summed live.
    def agg(source, patch, gran, thr):
        sel = [r for r in rows if r["source"] == source and r["patch"] == patch
               and r["granularity"] == gran and r["threshold"] == thr]
        live = sum(r["live"] for r in sel)
        mono = sum(r["mono_count"] for r in sel)
        return 100.0 * mono / live if live else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white", squeeze=False)
    for col, thr in enumerate(THRESHOLDS):
        ax = axes[0][col]
        ax.set_facecolor("white")
        groups = [("latent p2", "latent", "p2"), ("latent p4", "latent", "p4"), ("DiT B/2", "dit", "p2")]
        x = np.arange(len(grans))
        width = 0.25
        for gi, (label, source, patch) in enumerate(groups):
            vals = [agg(source, patch, g, thr) for g in grans]
            ax.bar(x + (gi - 1) * width, vals, width, label=label, color=PB[gi])
        ax.set_xticks(x)
        ax.set_xticklabels(["fine\n(1000)", f"medium\n(~{meta['medium_n_groups']})",
                            f"coarse\n(~{meta['coarse_n_groups']})"])
        ax.set_ylabel("monosemantic % of live")
        ax.set_title(f"entropy threshold < {thr} bits")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Superclass-entropy monosemanticity: latent SAEs vs SiT-B/2 DiT", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
