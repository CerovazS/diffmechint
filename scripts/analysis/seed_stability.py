"""Cross-seed stability utilities for the Matryoshka SAE replicate."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

CELL_RE = re.compile(r"^(?P<cond>[a-z_0-9]+?)_L(?P<layer>\d+)_T(?P<tbin>\d+)$")
CONDS = ["sd_vae", "repa_e", "eq_vae"]
LAYERS = [3, 6, 9]
T_BINS = [0, 1, 2]
PB = {
    "ink": "#1c1b19",
    "teal": "#335C67",
    "amber": "#E09F3E",
    "red": "#9E2A2B",
    "cream": "#FFF3B0",
    "wine": "#540B0E",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _normalise_eval_df(df: pd.DataFrame, label: str, sweep_filter: str | None) -> pd.DataFrame:
    if sweep_filter is not None and "sweep" in df.columns:
        df = df[df["sweep"] == sweep_filter].copy()
    rename = {
        "cond": "condition",
        "val_ev": "ev",
        "val_cos": "recon_cosine",
        "val_dead": "dead_features",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    needed = ["condition", "layer", "t_bin", "dit_step", "ev", "recon_cosine"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}")
    keep = [c for c in [
        "condition",
        "layer",
        "t_bin",
        "dit_step",
        "stage",
        "ev",
        "recon_cosine",
        "mse",
        "dead_pct",
        "dead_features",
        "live_features",
        "l0_mean",
        "ckpt_dir",
    ] if c in df.columns]
    out = df[keep].copy()
    out["condition"] = out["condition"].astype(str)
    out["layer"] = out["layer"].astype(int)
    out["t_bin"] = out["t_bin"].astype(int)
    out["dit_step"] = out["dit_step"].astype(int)
    if "stage" in out.columns:
        out = out[out["stage"].fillna("final") == "final"].copy()
    out["seed_label"] = label
    return out


def cmd_compare_eval(args: argparse.Namespace) -> int:
    a = _normalise_eval_df(_read_csv(args.seed0_csv), args.seed0_label, args.seed0_sweep)
    b = _normalise_eval_df(_read_csv(args.seed1_csv), args.seed1_label, args.seed1_sweep)
    if args.only_dit_step is not None:
        a = a[a["dit_step"] == args.only_dit_step].copy()
        b = b[b["dit_step"] == args.only_dit_step].copy()

    keys = ["condition", "layer", "t_bin", "dit_step"]
    merged = a.merge(b, on=keys, suffixes=("_seed0", "_seed1"), how="inner")
    for col in ["ev", "recon_cosine", "dead_pct", "dead_features", "live_features", "l0_mean"]:
        c0, c1 = f"{col}_seed0", f"{col}_seed1"
        if c0 in merged.columns and c1 in merged.columns:
            merged[f"delta_{col}"] = merged[c1] - merged[c0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired_csv = args.out_dir / "paired_eval_metrics.csv"
    merged.to_csv(paired_csv, index=False)

    summary: dict[str, object] = {
        "seed0_csv": str(args.seed0_csv),
        "seed1_csv": str(args.seed1_csv),
        "seed0_label": args.seed0_label,
        "seed1_label": args.seed1_label,
        "only_dit_step": args.only_dit_step,
        "n_seed0_rows": len(a),
        "n_seed1_rows": len(b),
        "n_paired_rows": len(merged),
        "deltas": {},
    }
    for col in [c for c in merged.columns if c.startswith("delta_")]:
        vals = merged[col].dropna().astype(float)
        summary["deltas"][col] = {
            "mean": float(vals.mean()) if len(vals) else math.nan,
            "median": float(vals.median()) if len(vals) else math.nan,
            "min": float(vals.min()) if len(vals) else math.nan,
            "max": float(vals.max()) if len(vals) else math.nan,
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if len(merged) and "delta_ev" in merged.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = np.arange(len(merged))
        colors = [PB["teal"] if v >= 0 else PB["red"] for v in merged["delta_ev"]]
        ax.bar(xs, merged["delta_ev"], color=colors, edgecolor=PB["ink"], linewidth=0.4)
        ax.axhline(0, color=PB["ink"], linewidth=0.8)
        ax.set_ylabel("seed1 - seed0 EV")
        ax.set_xlabel("paired cell")
        ax.set_title("Cross-seed held-out EV delta")
        fig.tight_layout()
        fig.savefig(args.out_dir / "delta_ev_bar.png", dpi=160)
        plt.close(fig)

    print(f"wrote {paired_csv}")
    print(json.dumps(summary, indent=2))
    return 0


def _parse_cell(path: Path) -> tuple[str, int, int] | None:
    m = CELL_RE.match(path.name)
    if not m:
        return None
    return m["cond"], int(m["layer"]), int(m["tbin"])


def _iter_feature_docs(feat_dir: Path):
    rg = shutil.which("rg")
    if rg is None:
        for p in sorted(feat_dir.glob("feature_*.json")):
            yield p.read_text()
        return

    proc = subprocess.Popen(
        [
            rg,
            "--no-messages",
            "--with-filename",
            "--line-number",
            "--color",
            "never",
            "-F",
            '"feature_id"',
            str(feat_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        parts = line.split(":", 2)
        if len(parts) == 3:
            yield parts[2]
    proc.wait()
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"rg failed while reading {feat_dir} (exit={proc.returncode})")


def _feature_rows(cell_dir: Path, *, mono_only: bool, max_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    rows = []
    feat_dir = cell_dir / "features"
    if not feat_dir.is_dir():
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 9), dtype=np.int32),
            np.zeros((0, 9), dtype=np.int16),
            False,
        )
    for text in _iter_feature_docs(feat_dir):
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not d.get("live"):
            continue
        dens = float(d.get("density", 0.0))
        ent = float(d.get("entropy", 99.0))
        is_mono = 1e-4 <= dens <= 0.10 and ent < 2.5
        if mono_only and not is_mono:
            continue
        top = d.get("top") or []
        ds = [int(t.get("dataset_idx", -1)) for t in top[:9]]
        cls = [int(t.get("class_idx", -1)) for t in top[:9]]
        while len(ds) < 9:
            ds.append(-1)
            cls.append(-1)
        rows.append((int(d["feature_id"]), ent, -dens, ds, cls))
    rows.sort(key=lambda r: (r[1], r[2], r[0]))
    truncated = len(rows) > max_features
    rows = rows[:max_features]
    if not rows:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 9), dtype=np.int32),
            np.zeros((0, 9), dtype=np.int16),
            truncated,
        )
    fids = np.asarray([r[0] for r in rows], dtype=np.int32)
    ds = np.asarray([r[3] for r in rows], dtype=np.int32)
    cls = np.asarray([r[4] for r in rows], dtype=np.int16)
    return fids, ds, cls, truncated


def _jaccard_cost_blocked(a: np.ndarray, b: np.ndarray, block: int = 256) -> np.ndarray:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    b_valid = b >= 0
    b_count = b_valid.sum(axis=1).astype(np.float32)
    for lo in range(0, a.shape[0], block):
        hi = min(lo + block, a.shape[0])
        aa = a[lo:hi]
        a_valid = aa >= 0
        a_count = a_valid.sum(axis=1).astype(np.float32)
        eq = (aa[:, None, :, None] == b[None, :, None, :]) & a_valid[:, None, :, None] & b_valid[None, :, None, :]
        inter = eq.any(axis=3).sum(axis=2).astype(np.float32)
        union = a_count[:, None] + b_count[None, :] - inter
        jac = np.where(union > 0, inter / union, 0.0)
        out[lo:hi] = 1.0 - jac
    return out


def _cell_dirs(root: Path) -> dict[str, Path]:
    out = {}
    for p in root.iterdir() if root.exists() else []:
        if p.is_dir() and _parse_cell(p) is not None:
            out[p.name] = p
    return out


def cmd_hungarian(args: argparse.Namespace) -> int:
    cells_a = _cell_dirs(args.seed0_features)
    cells_b = _cell_dirs(args.seed1_features)
    cells = sorted(set(cells_a).intersection(cells_b))
    if args.cells:
        requested = set(args.cells)
        cells = [c for c in cells if c in requested]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    top_rows = []
    for i, cell in enumerate(cells, start=1):
        f_a, ds_a, cls_a, trunc_a = _feature_rows(cells_a[cell], mono_only=args.mono_only, max_features=args.max_features_per_cell)
        f_b, ds_b, cls_b, trunc_b = _feature_rows(cells_b[cell], mono_only=args.mono_only, max_features=args.max_features_per_cell)
        cond, layer, tbin = _parse_cell(cells_a[cell]) or ("", -1, -1)
        if len(f_a) == 0 or len(f_b) == 0:
            rows.append({
                "cell": cell,
                "condition": cond,
                "layer": layer,
                "t_bin": tbin,
                "n_seed0": len(f_a),
                "n_seed1": len(f_b),
                "truncated_seed0": trunc_a,
                "truncated_seed1": trunc_b,
                "mean_dataset_cost": math.nan,
                "mean_class_cost": math.nan,
                "mean_combined_cost": math.nan,
                "frac_good": math.nan,
                "min_combined_cost": math.nan,
            })
            continue
        dataset_cost = _jaccard_cost_blocked(ds_a, ds_b, block=args.block_size)
        class_cost = _jaccard_cost_blocked(cls_a.astype(np.int32), cls_b.astype(np.int32), block=args.block_size)
        combined = args.dataset_weight * dataset_cost + (1.0 - args.dataset_weight) * class_cost
        r, c = linear_sum_assignment(combined)
        matched_combined = combined[r, c]
        matched_dataset = dataset_cost[r, c]
        matched_class = class_cost[r, c]
        rows.append({
            "cell": cell,
            "condition": cond,
            "layer": layer,
            "t_bin": tbin,
            "n_seed0": len(f_a),
            "n_seed1": len(f_b),
            "n_matched": len(r),
            "truncated_seed0": trunc_a,
            "truncated_seed1": trunc_b,
            "mean_dataset_cost": float(matched_dataset.mean()),
            "mean_class_cost": float(matched_class.mean()),
            "mean_combined_cost": float(matched_combined.mean()),
            "median_combined_cost": float(np.median(matched_combined)),
            "min_combined_cost": float(matched_combined.min()),
            "frac_good": float((matched_combined < args.good_threshold).mean()),
        })
        order = np.argsort(matched_combined)[: args.top_k_matches]
        for rank, j in enumerate(order, start=1):
            top_rows.append({
                "cell": cell,
                "rank": rank,
                "feature_seed0": int(f_a[r[j]]),
                "feature_seed1": int(f_b[c[j]]),
                "dataset_cost": float(matched_dataset[j]),
                "class_cost": float(matched_class[j]),
                "combined_cost": float(matched_combined[j]),
            })
        print(f"[{i:2d}/{len(cells)}] {cell}: n=({len(f_a)},{len(f_b)}) mean={rows[-1]['mean_combined_cost']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "hungarian_cross_seed_summary.csv", index=False)
    pd.DataFrame(top_rows).to_csv(args.out_dir / "hungarian_cross_seed_top_matches.csv", index=False)
    summary = {
        "seed0_features": str(args.seed0_features),
        "seed1_features": str(args.seed1_features),
        "n_cells": len(df),
        "mono_only": bool(args.mono_only),
        "max_features_per_cell": int(args.max_features_per_cell),
        "dataset_weight": float(args.dataset_weight),
        "good_threshold": float(args.good_threshold),
        "mean_combined_cost": float(df["mean_combined_cost"].dropna().mean()) if len(df) else math.nan,
        "mean_frac_good": float(df["frac_good"].dropna().mean()) if len(df) else math.nan,
        "n_truncated_cells": int((df.get("truncated_seed0", False).fillna(False) | df.get("truncated_seed1", False).fillna(False)).sum()) if len(df) else 0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if len(df):
        fig, ax = plt.subplots(figsize=(8, 4))
        valid = df.dropna(subset=["mean_combined_cost"]).copy()
        ax.bar(np.arange(len(valid)), valid["mean_combined_cost"], color=PB["amber"], edgecolor=PB["ink"], linewidth=0.4)
        ax.axhline(0.910, color=PB["red"], linestyle=":", linewidth=1.2, label="E12 cross-autoencoder mean")
        ax.set_ylabel("mean Hungarian cost")
        ax.set_xlabel("same-cell seed0 ↔ seed3407 pairs")
        ax.set_title("Cross-seed feature matching cost")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / "hungarian_cross_seed_costs.png", dpi=160)
        plt.close(fig)

    print(json.dumps(summary, indent=2))
    return 0


def _atlas_rows(root: Path, label: str) -> pd.DataFrame:
    rows = []
    summary_path = root / "atlas_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for cell, stats in summary.items():
            m = CELL_RE.match(cell)
            if not m:
                continue
            rows.append({
                "cell": cell,
                "condition": m["cond"],
                "layer": int(m["layer"]),
                "t_bin": int(m["tbin"]),
                "live_features": stats.get("live"),
                "dead_features": None,
                "live_pct": stats.get("live_pct"),
                "monosemantic_count": stats.get("mono_count"),
                "very_monosemantic_count": stats.get("very_mono_count"),
                "pure_monosemantic_count": None,
                "mean_entropy": stats.get("mean_entropy"),
                "median_entropy": None,
                "mean_density": stats.get("mean_density"),
                "median_density": None,
                "label": label,
            })
        if rows:
            return pd.DataFrame(rows)

    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir() and CELL_RE.match(p.name)):
        stats_path = cell_dir / "stats.json"
        if not stats_path.exists():
            continue
        try:
            stats = json.loads(stats_path.read_text())
        except json.JSONDecodeError:
            continue
        cond, layer, tbin = _parse_cell(cell_dir) or ("", -1, -1)
        ent = stats.get("entropy_stats") or {}
        dens = stats.get("density_stats") or {}
        rows.append({
            "cell": cell_dir.name,
            "condition": cond,
            "layer": layer,
            "t_bin": tbin,
            "live_features": stats.get("live_features"),
            "dead_features": stats.get("dead_features"),
            "live_pct": stats.get("live_pct"),
            "monosemantic_count": stats.get("monosemantic_count"),
            "very_monosemantic_count": stats.get("very_monosemantic_count"),
            "pure_monosemantic_count": stats.get("pure_monosemantic_count"),
            "mean_entropy": ent.get("mean"),
            "median_entropy": ent.get("median"),
            "mean_density": dens.get("mean"),
            "median_density": dens.get("median"),
            "label": label,
        })
    return pd.DataFrame(rows)


def cmd_compare_atlas(args: argparse.Namespace) -> int:
    a = _atlas_rows(args.seed0_features, args.seed0_label)
    b = _atlas_rows(args.seed1_features, args.seed1_label)
    keys = ["cell", "condition", "layer", "t_bin"]
    merged = a.merge(b, on=keys, suffixes=("_seed0", "_seed1"), how="inner")
    for col in [
        "live_features",
        "dead_features",
        "live_pct",
        "monosemantic_count",
        "very_monosemantic_count",
        "pure_monosemantic_count",
        "mean_entropy",
        "median_entropy",
        "mean_density",
        "median_density",
    ]:
        c0, c1 = f"{col}_seed0", f"{col}_seed1"
        if c0 in merged.columns and c1 in merged.columns:
            merged[f"delta_{col}"] = merged[c1].astype(float) - merged[c0].astype(float)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "paired_atlas_stats.csv", index=False)
    a.to_csv(args.out_dir / "seed0_atlas_stats.csv", index=False)
    b.to_csv(args.out_dir / "seed1_atlas_stats.csv", index=False)

    correlations = {}
    for col in ["monosemantic_count", "very_monosemantic_count", "mean_entropy", "live_pct"]:
        c0, c1 = f"{col}_seed0", f"{col}_seed1"
        if c0 in merged.columns and c1 in merged.columns:
            sub = merged[[c0, c1]].dropna().astype(float)
            correlations[col] = float(sub[c0].corr(sub[c1])) if len(sub) >= 2 else math.nan

    summary = {
        "seed0_features": str(args.seed0_features),
        "seed1_features": str(args.seed1_features),
        "n_seed0_cells": len(a),
        "n_seed1_cells": len(b),
        "n_paired_cells": len(merged),
        "correlations": correlations,
        "delta_monosemantic_count_mean": (
            float(merged["delta_monosemantic_count"].mean())
            if "delta_monosemantic_count" in merged.columns and len(merged)
            else math.nan
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if len(merged) and "monosemantic_count_seed0" in merged.columns:
        fig, ax = plt.subplots(figsize=(5, 5))
        color_by_cond = {"sd_vae": PB["red"], "repa_e": PB["teal"], "eq_vae": PB["amber"]}
        colors = [color_by_cond.get(c, PB["ink"]) for c in merged["condition"]]
        ax.scatter(
            merged["monosemantic_count_seed0"],
            merged["monosemantic_count_seed1"],
            c=colors,
            edgecolor=PB["ink"],
            linewidth=0.4,
            s=55,
        )
        max_v = float(max(merged["monosemantic_count_seed0"].max(), merged["monosemantic_count_seed1"].max()))
        ax.plot([0, max_v], [0, max_v], color=PB["ink"], linestyle=":", linewidth=1)
        ax.set_xlabel("seed0 monosemantic count")
        ax.set_ylabel("seed3407 monosemantic count")
        ax.set_title("Monosemantic count stability")
        fig.tight_layout()
        fig.savefig(args.out_dir / "mono_count_scatter.png", dpi=160)
        plt.close(fig)

    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare-eval")
    c.add_argument("--seed0_csv", type=Path, required=True)
    c.add_argument("--seed1_csv", type=Path, required=True)
    c.add_argument("--out_dir", type=Path, required=True)
    c.add_argument("--seed0_label", default="seed0")
    c.add_argument("--seed1_label", default="seed3407")
    c.add_argument("--seed0_sweep", default=None)
    c.add_argument("--seed1_sweep", default=None)
    c.add_argument("--only_dit_step", type=int, default=None)
    c.set_defaults(func=cmd_compare_eval)

    h = sub.add_parser("hungarian")
    h.add_argument("--seed0_features", type=Path, required=True)
    h.add_argument("--seed1_features", type=Path, required=True)
    h.add_argument("--out_dir", type=Path, required=True)
    h.add_argument("--cells", nargs="*", default=None)
    h.add_argument("--mono_only", action="store_true", default=True)
    h.add_argument("--all_live", dest="mono_only", action="store_false")
    h.add_argument("--max_features_per_cell", type=int, default=5000)
    h.add_argument("--dataset_weight", type=float, default=0.7)
    h.add_argument("--good_threshold", type=float, default=0.5)
    h.add_argument("--top_k_matches", type=int, default=20)
    h.add_argument("--block_size", type=int, default=256)
    h.set_defaults(func=cmd_hungarian)

    a = sub.add_parser("compare-atlas")
    a.add_argument("--seed0_features", type=Path, required=True)
    a.add_argument("--seed1_features", type=Path, required=True)
    a.add_argument("--out_dir", type=Path, required=True)
    a.add_argument("--seed0_label", default="seed0")
    a.add_argument("--seed1_label", default="seed3407")
    a.set_defaults(func=cmd_compare_atlas)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
