"""Aggregate the 27-cell y-null SAE feature atlas into tidy CSVs for analysis.

Reads from outputs/phase4_5b_feature_viz_ynull/<cell>/{stats.json, features/, vlm_log.jsonl}
and writes:

    outputs/phase4_8_atlas/atlas_per_cell.csv       (27 rows — one per cell)
    outputs/phase4_8_atlas/atlas_mono_features.csv  (~2400 rows — mono features only)
    outputs/phase4_8_atlas/entropy_dists/<cell>.npy (per-cell entropy arrays of live features)

The CSV `atlas_mono_features.csv` is the canonical input for all Axis-A / Axis-B plots.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from diffmechint.analysis.atlas import (
    MONO_DENSITY_MAX,
    MONO_DENSITY_MIN,
    MONO_ENTROPY_MAX,
    parse_cell_name,
    select_cell_dirs,
)

ATLAS_ROOT_DEFAULT = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/phase4_5b_feature_viz_ynull")
OUT_ROOT_DEFAULT = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/phase4_8_atlas")

T_VALUES = {0: 0.025, 1: 0.20, 2: 0.50}


def process_cell(cell_dir_str: str) -> tuple[dict, list[dict], np.ndarray]:
    cell_dir = Path(cell_dir_str)
    cell = cell_dir.name
    cond, layer, tbin = parse_cell_name(cell)

    stats = json.loads((cell_dir / "stats.json").read_text())

    mono_rows: list[dict] = []
    entropies: list[float] = []

    for f in sorted((cell_dir / "features").iterdir()):
        if not f.name.startswith("feature_") or not f.name.endswith(".json"):
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not d.get("live"):
            continue

        ent = d.get("entropy")
        if ent is None or not np.isfinite(ent):
            continue
        entropies.append(float(ent))

        dens = float(d.get("density", 0.0))
        if not (MONO_DENSITY_MIN <= dens <= MONO_DENSITY_MAX):
            continue
        if ent >= MONO_ENTROPY_MAX:
            continue

        top = d.get("top") or []
        top0 = top[0] if top else {}
        mono_rows.append({
            "cell": cell,
            "cond": cond,
            "layer": layer,
            "tbin": tbin,
            "t_val": T_VALUES[tbin],
            "feature_id": int(d["feature_id"]),
            "density": dens,
            "entropy": float(ent),
            "unique_classes": int(d.get("unique_classes") or 0),
            "mean_act": float(d.get("mean_act") or 0.0),
            "vlm_interpretation": d.get("vlm_interpretation"),
            "top_label": top0.get("label"),
            "top_class_idx": top0.get("class_idx"),
            "top_synset": top0.get("synset"),
            "top_activation": float(top0.get("activation") or 0.0),
        })

    per_cell = {
        "cell": cell,
        "cond": cond,
        "layer": layer,
        "tbin": tbin,
        "t_val": T_VALUES[tbin],
        "live_features": int(stats.get("live_features") or 0),
        "monosemantic_count_stats": int(stats.get("monosemantic_count") or 0),
        "mono_count_recomputed": len(mono_rows),
        "mean_activation_stats": float(stats.get("mean_activation") or 0.0),
        "mean_entropy_live": float(np.mean(entropies)) if entropies else float("nan"),
        "median_entropy_live": float(np.median(entropies)) if entropies else float("nan"),
        "p25_entropy_live": float(np.percentile(entropies, 25)) if entropies else float("nan"),
        "p75_entropy_live": float(np.percentile(entropies, 75)) if entropies else float("nan"),
        "n_live_entropies": len(entropies),
    }

    return per_cell, mono_rows, np.asarray(entropies, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard_root", type=Path,
                        default=ATLAS_ROOT_DEFAULT,
                        dest="dashboard_root",
                        help="Feature dashboard root containing <cond>_L<L>_T<T> dirs.")
    parser.add_argument("--atlas_root", "--out_root", type=Path,
                        default=OUT_ROOT_DEFAULT,
                        dest="out_root",
                        help="Output root for atlas CSVs and entropy arrays.")
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9])
    parser.add_argument("--conditions", nargs="+", default=["sd_vae", "repa_e", "eq_vae"])
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "entropy_dists").mkdir(exist_ok=True)

    layers = set(args.layers)
    conditions = set(args.conditions)
    cell_dirs = select_cell_dirs(args.dashboard_root, conditions, layers)
    expected = len(conditions) * len(layers) * 3
    if len(cell_dirs) != expected:
        print(f"WARN: found {len(cell_dirs)} cells, expected {expected}", file=sys.stderr)
    print(f"processing {len(cell_dirs)} cells (parallel workers=8)...")

    per_cell_rows: list[dict] = []
    mono_rows_all: list[dict] = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=8) as ex:
        fut_to_dir = {ex.submit(process_cell, str(d)): d for d in cell_dirs}
        for i, fut in enumerate(as_completed(fut_to_dir), start=1):
            cell_dir = fut_to_dir[fut]
            per_cell, mono_rows, entropies = fut.result()
            per_cell_rows.append(per_cell)
            mono_rows_all.extend(mono_rows)
            np.save(args.out_root / "entropy_dists" / f"{cell_dir.name}.npy", entropies)
            print(f"  [{i:2d}/{len(cell_dirs)}] {cell_dir.name:22s}  "
                  f"live={per_cell['live_features']:>5}  "
                  f"mono={per_cell['mono_count_recomputed']:>4}")

    df_cell = pd.DataFrame(per_cell_rows).sort_values(["cond", "layer", "tbin"])
    df_mono = pd.DataFrame(mono_rows_all).sort_values(["cond", "layer", "tbin", "feature_id"])
    df_cell.to_csv(args.out_root / "atlas_per_cell.csv", index=False)
    df_mono.to_csv(args.out_root / "atlas_mono_features.csv", index=False)
    print(f"wrote atlas_per_cell.csv ({len(df_cell)} rows)")
    print(f"wrote atlas_mono_features.csv ({len(df_mono)} rows)")
    print("wrote 27 entropy_dists/*.npy")
    print(f"total time: {(time.perf_counter() - t0):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
