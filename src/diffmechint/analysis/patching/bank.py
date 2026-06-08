"""Feature-bank construction from dashboard JSON or atlas CSV."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from diffmechint.analysis.alignment import T_CENTERS
from diffmechint.utils import info, make_run_dir, ok, warn, write_summary_md

from .common import _decoder_norm_map, read_csv, selected_cells, write_csv


def _feature_dir(dashboard_root: Path, condition: str, layer: int, t_bin: int) -> Path:
    return dashboard_root / f"{condition}_L{layer}_T{t_bin}" / "features"


def _load_feature_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_monosemantic(
    payload: dict,
    *,
    min_density: float,
    max_density: float,
    max_entropy: float,
) -> bool:
    density = float(payload.get("density", 0.0))
    entropy = float(payload.get("entropy", math.inf))
    top = payload.get("top") or []
    return (
        bool(payload.get("live", True))
        and min_density <= density <= max_density
        and entropy <= max_entropy
        and len(top) >= 9
    )


def feature_payload_to_row(condition: str, layer: int, t_bin: int, payload: dict) -> dict:
    top = payload.get("top") or []
    top0 = top[0] if top else {}
    top9_class_idx = [int(x.get("class_idx", -1)) for x in top]
    top9_dataset_idx = [int(x.get("dataset_idx", -1)) for x in top]
    top9_activation = [float(x.get("activation", 0.0)) for x in top]
    top9_token_pos = [int(x.get("token_pos", -1)) for x in top]
    return {
        "condition": condition,
        "layer": int(layer),
        "t_bin": int(t_bin),
        "t_center": T_CENTERS[int(t_bin)],
        "feature_id": int(payload["feature_id"]),
        "density": float(payload.get("density", 0.0)),
        "density_count": int(payload.get("density_count", 0)),
        "entropy": float(payload.get("entropy", math.nan)),
        "unique_classes": int(payload.get("unique_classes", 0)),
        "mean_act": float(payload.get("mean_act", 0.0)),
        "top_activation": float(top0.get("activation", 0.0)),
        "top_label": str(top0.get("label", "")),
        "top_class_idx": int(top0.get("class_idx", -1)),
        "top_synset": str(top0.get("synset", "")),
        "vlm_interpretation": str(payload.get("vlm_interpretation", "")),
        "top9_class_idx": top9_class_idx,
        "top9_dataset_idx": top9_dataset_idx,
        "top9_activation": top9_activation,
        "top9_token_pos": top9_token_pos,
        "n_top_examples": len(top),
        "decoder_norm": "",
    }


def run_build_bank(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id, resume=args.resume)
    rows: list[dict] = []
    if args.atlas_csv is not None:
        cell_set = set(selected_cells(args))
        for row in read_csv(args.atlas_csv):
            condition = str(row.get("cond", row.get("condition", "")))
            layer = int(row.get("layer", -1))
            t_bin = int(row.get("tbin", row.get("t_bin", -1)))
            if condition not in args.conditions or (layer, t_bin) not in cell_set:
                continue
            if args.hydrate_dashboard_json:
                feature_json = _feature_dir(args.dashboard_root, condition, layer, t_bin) / (
                    f"feature_{int(row['feature_id'])}.json"
                )
                if feature_json.exists():
                    rows.append(feature_payload_to_row(condition, layer, t_bin, _load_feature_json(feature_json)))
                    continue
                warn(f"dashboard JSON missing for atlas feature: {feature_json}; using compact atlas row")
            rows.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "t_bin": t_bin,
                    "t_center": float(row.get("t_val", T_CENTERS[t_bin])),
                    "feature_id": int(row["feature_id"]),
                    "density": float(row["density"]),
                    "density_count": "",
                    "entropy": float(row["entropy"]),
                    "unique_classes": int(row.get("unique_classes", 0)),
                    "mean_act": float(row.get("mean_act", 0.0)),
                    "top_activation": float(row.get("top_activation", 0.0)),
                    "top_label": str(row.get("top_label", "")),
                    "top_class_idx": int(row.get("top_class_idx", -1)),
                    "top_synset": str(row.get("top_synset", "")),
                    "vlm_interpretation": str(row.get("vlm_interpretation", "")),
                    "top9_class_idx": [int(row.get("top_class_idx", -1))],
                    "top9_dataset_idx": [],
                    "top9_activation": [float(row.get("top_activation", 0.0))],
                    "top9_token_pos": [],
                    "n_top_examples": 1,
                    "decoder_norm": "",
                }
            )
        if args.max_features_per_cell:
            capped = []
            grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
            for row in rows:
                grouped[(row["condition"], int(row["layer"]), int(row["t_bin"]))].append(row)
            for cell_rows in grouped.values():
                cell_rows.sort(key=lambda r: (float(r["entropy"]), -float(r["top_activation"])))
                capped.extend(cell_rows[: args.max_features_per_cell])
            rows = capped
        if args.with_decoder_norms and rows:
            device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
            grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
            for row in rows:
                grouped[(row["condition"], int(row["layer"]), int(row["t_bin"]))].append(row)
            for (condition, layer, t_bin), cell_rows in grouped.items():
                norms = _decoder_norm_map(
                    args.sae_root,
                    condition,
                    layer,
                    t_bin,
                    args.dit_step,
                    [int(row["feature_id"]) for row in cell_rows],
                    device,
                )
                for row in cell_rows:
                    row["decoder_norm"] = norms.get(int(row["feature_id"]), "")
        write_csv(out_dir / "metrics" / "feature_bank.csv", rows)
        summary = {
            "analysis": "feature-bank",
            "source": str(args.atlas_csv),
            "n_features": len(rows),
            "conditions": list(args.conditions),
            "cells": [f"L{layer}_T{t_bin}" for layer, t_bin in selected_cells(args)],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_summary_md(
            out_dir,
            "Feature Bank",
            [
                f"**Atlas monosemantic features** yielded =={len(rows)}== compact bank rows.",
                "This fast path reuses Phase 4.8 atlas metadata derived from y-null dashboards.",
                "Top-9 overlap is reduced to top-class identity when using the atlas CSV.",
            ],
        )
        ok(f"feature bank complete from atlas: {out_dir}")
        return 0
    for condition in args.conditions:
        for layer, t_bin in selected_cells(args):
            fdir = _feature_dir(args.dashboard_root, condition, layer, t_bin)
            if not fdir.exists():
                raise FileNotFoundError(f"feature directory missing: {fdir}")
            cell_rows = []
            for path in sorted(fdir.glob("feature_*.json")):
                payload = _load_feature_json(path)
                if args.monosemantic_only and not is_monosemantic(
                    payload,
                    min_density=args.min_density,
                    max_density=args.max_density,
                    max_entropy=args.max_entropy,
                ):
                    continue
                cell_rows.append(feature_payload_to_row(condition, layer, t_bin, payload))
            cell_rows.sort(key=lambda r: (float(r["entropy"]), -float(r["top_activation"])))
            if args.max_features_per_cell:
                cell_rows = cell_rows[: args.max_features_per_cell]
            if args.with_decoder_norms and cell_rows:
                device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
                norms = _decoder_norm_map(
                    args.sae_root,
                    condition,
                    layer,
                    t_bin,
                    args.dit_step,
                    [int(row["feature_id"]) for row in cell_rows],
                    device,
                )
                for row in cell_rows:
                    row["decoder_norm"] = norms.get(int(row["feature_id"]), "")
            info(f"bank {condition} L{layer}/T{t_bin}: {len(cell_rows)} features")
            rows.extend(cell_rows)
    write_csv(out_dir / "metrics" / "feature_bank.csv", rows)
    summary = {
        "analysis": "feature-bank",
        "n_features": len(rows),
        "conditions": list(args.conditions),
        "cells": [f"L{layer}_T{t_bin}" for layer, t_bin in selected_cells(args)],
        "monosemantic_only": bool(args.monosemantic_only),
        "filters": {
            "min_density": args.min_density,
            "max_density": args.max_density,
            "max_entropy": args.max_entropy,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(
        out_dir,
        "Feature Bank",
        [
            f"**Monosemantic dashboard features** yielded =={len(rows)}== compact bank rows.",
            "The bank is metadata-only: no thumbnails or source images are read.",
            "Rows are ranked by entropy and activation strength within each cell.",
        ],
    )
    ok(f"feature bank complete: {out_dir}")
    return 0
