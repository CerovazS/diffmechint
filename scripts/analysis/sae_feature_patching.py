"""Phase 4.11 feature-level SAE activation patching analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffmechint.utils import info, ok, warn  # noqa: E402
from scripts.analysis.tokenizer_dictionary_validation import (  # noqa: E402
    ACTIVATIONS_YNULL,
    SAE_ROOT,
    T_CENTERS,
    _aligned_positions,
    _cell_path,
    _read_rows,
    alignment_metrics,
    fit_ridge_affine,
)

DEFAULT_DASHBOARD_ROOT = Path("outputs/phase4_5b_feature_viz_ynull")
DEFAULT_OUT_ROOT = Path("outputs/phase4_11_feature_activation_patching")
DEFAULT_CONDITIONS = ("sd_vae", "repa_e", "eq_vae")
LAYERS = (3, 6, 9)
T_BINS = (0, 1, 2)

PB = {
    "blue": "#335C67",
    "cream": "#FFF3B0",
    "gold": "#E09F3E",
    "red": "#9E2A2B",
    "dark": "#540B0E",
}


@dataclass(frozen=True)
class FeatureRow:
    condition: str
    layer: int
    t_bin: int
    feature_id: int
    density: float
    density_count: int | None
    entropy: float
    unique_classes: int
    mean_act: float
    top_activation: float
    top_class_idx: int
    top_label: str
    top_synset: str
    top9_class_idx: tuple[int, ...]
    top9_dataset_idx: tuple[int, ...]
    top9_activation: tuple[float, ...]
    top9_token_pos: tuple[int, ...]
    vlm_interpretation: str
    decoder_norm: float | None = None


class MetricAccumulator:
    """Online reconstruction metrics for a fixed target activation matrix."""

    def __init__(
        self,
        target_sum: np.ndarray | None = None,
        target_sumsq: np.ndarray | None = None,
        n: int = 0,
    ) -> None:
        self.target_sum = None if target_sum is None else target_sum.astype(np.float64)
        self.target_sumsq = None if target_sumsq is None else target_sumsq.astype(np.float64)
        self.n = int(n)
        self.sse = 0.0
        self.cos_sum = 0.0
        self.rows = 0

    @property
    def sst(self) -> float:
        if self.target_sum is None or self.target_sumsq is None:
            return 0.0
        return float(self.target_sumsq.sum() - (self.target_sum**2).sum() / max(self.n, 1))

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred32 = pred.detach().float()
        target32 = target.detach().float()
        target_np = target32.detach().cpu().numpy().astype(np.float64)
        add_sum = target_np.sum(axis=0)
        add_sumsq = (target_np * target_np).sum(axis=0)
        self.target_sum = add_sum if self.target_sum is None else self.target_sum + add_sum
        self.target_sumsq = add_sumsq if self.target_sumsq is None else self.target_sumsq + add_sumsq
        self.n += int(target32.shape[0])
        err = target32 - pred32
        self.sse += float((err * err).sum().item())
        denom = pred32.norm(dim=1) * target32.norm(dim=1)
        cos = (pred32 * target32).sum(dim=1) / denom.clamp_min(1e-12)
        self.cos_sum += float(cos.sum().item())
        self.rows += int(target32.shape[0])

    def finalize(self) -> dict[str, float]:
        denom = max(self.rows, 1)
        dim = int(self.target_sum.shape[0]) if self.target_sum is not None else 1
        mse = self.sse / max(self.rows * dim, 1)
        return {
            "mse": float(mse),
            "cosine": float(self.cos_sum / denom),
            "ev": float(1.0 - self.sse / self.sst) if self.sst > 0 else float("nan"),
        }


def make_run_dir(out_root: Path, run_id: str | None, *, resume: bool = False) -> Path:
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / run_id
    if out_dir.exists() and not resume:
        raise FileExistsError(f"run directory already exists: {out_dir}")
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "artifacts").mkdir(exist_ok=True)
    return out_dir


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        warn(f"no rows to write for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (list, tuple, dict)) else v for k, v in row.items()}
            )


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def selected_cells(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.cells:
        out = []
        for raw in args.cells:
            cleaned = raw.upper().replace("L", "").replace("T", "").replace(",", "_")
            layer, t_bin = cleaned.split("_", 1)
            out.append((int(layer), int(t_bin)))
        return out
    return [(int(layer), int(t_bin)) for layer in args.layers for t_bin in args.t_bins]


def _feature_dir(dashboard_root: Path, condition: str, layer: int, t_bin: int) -> Path:
    return dashboard_root / f"{condition}_L{layer}_T{t_bin}" / "features"


def _load_feature_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_optional_float(raw: object) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def _decoder_weight(sae: torch.nn.Module) -> torch.Tensor:
    """Return decoder directions as `(d_sae, d_in)` rows."""
    weight = getattr(sae, "W_dec", None)
    if weight is None and hasattr(sae, "decoder"):
        weight = getattr(sae.decoder, "weight", None)
    if weight is None:
        for name, param in sae.named_parameters():
            if name.endswith("W_dec") or name.endswith("decoder.weight"):
                weight = param
                break
    if weight is None:
        raise AttributeError("could not locate SAE decoder weight")
    out = weight.detach()
    d_sae = int(getattr(getattr(sae, "cfg", None), "d_sae", out.shape[0]))
    if out.ndim != 2:
        raise ValueError(f"decoder weight must be 2-D, got {tuple(out.shape)}")
    if out.shape[0] == d_sae:
        return out
    if out.shape[1] == d_sae:
        return out.T
    return out


def _decoder_norm_map(
    sae_root: Path,
    condition: str,
    layer: int,
    t_bin: int,
    dit_step: int,
    feature_ids: list[int],
    device: torch.device,
) -> dict[int, float]:
    if not feature_ids:
        return {}
    sae = _load_matryoshka_sae(_resolve_sae_ckpt(sae_root, condition, layer, t_bin, dit_step), device)
    dec = _decoder_weight(sae).float()
    ids = torch.as_tensor(feature_ids, device=dec.device, dtype=torch.long)
    norms = dec.index_select(0, ids).norm(dim=1).detach().cpu().numpy()
    del sae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {int(fid): float(norm) for fid, norm in zip(feature_ids, norms, strict=True)}


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
        _write_summary_md(
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
    _write_summary_md(
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


def _parse_json_list(raw: str | list | tuple) -> tuple[int, ...]:
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    if raw == "":
        return ()
    return tuple(int(x) for x in json.loads(raw))


def _parse_json_float_list(raw: str | list | tuple) -> tuple[float, ...]:
    if isinstance(raw, (list, tuple)):
        return tuple(float(x) for x in raw)
    if raw == "":
        return ()
    return tuple(float(x) for x in json.loads(raw))


def _row_to_feature(row: dict) -> FeatureRow:
    return FeatureRow(
        condition=str(row["condition"]),
        layer=int(row["layer"]),
        t_bin=int(row["t_bin"]),
        feature_id=int(row["feature_id"]),
        density=float(row["density"]),
        density_count=int(row["density_count"]) if str(row.get("density_count", "")).strip() else None,
        entropy=float(row["entropy"]),
        unique_classes=int(row.get("unique_classes", 0) or 0),
        mean_act=float(row["mean_act"]),
        top_activation=float(row["top_activation"]),
        top_class_idx=int(row["top_class_idx"]),
        top_label=str(row["top_label"]),
        top_synset=str(row["top_synset"]),
        top9_class_idx=_parse_json_list(row["top9_class_idx"]),
        top9_dataset_idx=_parse_json_list(row["top9_dataset_idx"]),
        top9_activation=_parse_json_float_list(row.get("top9_activation", "")),
        top9_token_pos=_parse_json_list(row.get("top9_token_pos", "")),
        vlm_interpretation=str(row.get("vlm_interpretation", "")),
        decoder_norm=_parse_optional_float(row.get("decoder_norm", "")),
    )


def _set_jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set = set(x for x in left if x >= 0)
    right_set = set(x for x in right if x >= 0)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _ratio_score(left: float | None, right: float | None, *, floor: float = 1e-12) -> float:
    if left is None or right is None:
        return 0.0
    left = max(float(left), floor)
    right = max(float(right), floor)
    return float(math.exp(-abs(math.log10(left) - math.log10(right))))


def feature_match_score(
    source: FeatureRow,
    target: FeatureRow,
    *,
    activation_corr: float | None = None,
    mapped_decoder_cosine: float | None = None,
) -> dict[str, float]:
    class_jaccard = _set_jaccard(source.top9_class_idx, target.top9_class_idx)
    image_jaccard = _set_jaccard(source.top9_dataset_idx, target.top9_dataset_idx)
    same_top = 1.0 if source.top_class_idx == target.top_class_idx else 0.0
    entropy_score = math.exp(-abs(source.entropy - target.entropy))
    density_score = _ratio_score(source.density, target.density)
    norm_score = _ratio_score(source.decoder_norm, target.decoder_norm)
    density_norm_compatibility = 0.5 * density_score + 0.5 * norm_score if norm_score > 0 else density_score
    act_score = min(source.top_activation, target.top_activation) / max(
        source.top_activation, target.top_activation, 1e-12
    )
    activation_component = max(float(activation_corr), 0.0) if activation_corr is not None and not math.isnan(float(activation_corr)) else 0.0
    decoder_component = max(float(mapped_decoder_cosine), 0.0) if mapped_decoder_cosine is not None and not math.isnan(float(mapped_decoder_cosine)) else 0.0
    score = (
        0.35 * activation_component
        + 0.20 * image_jaccard
        + 0.15 * class_jaccard
        + 0.20 * decoder_component
        + 0.10 * density_norm_compatibility
    )
    metadata_score = (
        0.45 * same_top
        + 0.30 * class_jaccard
        + 0.12 * entropy_score
        + 0.08 * density_score
        + 0.05 * act_score
    )
    return {
        "match_score": float(score),
        "metadata_score": float(metadata_score),
        "same_top_class": same_top,
        "top_image_jaccard": float(image_jaccard),
        "top_class_jaccard": float(class_jaccard),
        "top9_class_jaccard": float(class_jaccard),
        "activation_corr": float(activation_corr) if activation_corr is not None else float("nan"),
        "mapped_decoder_cosine": float(mapped_decoder_cosine) if mapped_decoder_cosine is not None else float("nan"),
        "entropy_score": float(entropy_score),
        "density_score": float(density_score),
        "decoder_norm_score": float(norm_score),
        "density_norm_compatibility": float(density_norm_compatibility),
        "activation_score": float(act_score),
    }


def _group_bank(rows: list[dict]) -> dict[tuple[str, int, int], list[FeatureRow]]:
    grouped: dict[tuple[str, int, int], list[FeatureRow]] = defaultdict(list)
    for row in rows:
        feat = _row_to_feature(row)
        grouped[(feat.condition, feat.layer, feat.t_bin)].append(feat)
    return grouped


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    corr = spearmanr(a, b).correlation
    return float(corr) if corr is not None else float("nan")


def _feature_profiles(
    sae: torch.nn.Module,
    acts: np.ndarray,
    feature_ids: list[int],
    *,
    device: torch.device,
    batch_images: int,
) -> dict[int, np.ndarray]:
    """Per-image max SAE coefficient for selected features."""
    if not feature_ids:
        return {}
    sae_dtype = next(sae.parameters()).dtype
    ids_t = torch.as_tensor(feature_ids, device=device, dtype=torch.long)
    values = np.zeros((acts.shape[0], len(feature_ids)), dtype=np.float32)
    for start in range(0, acts.shape[0], batch_images):
        chunk = acts[start : start + batch_images]
        flat = torch.from_numpy(chunk.reshape(-1, chunk.shape[-1]).astype(np.float32)).to(
            device=device, dtype=sae_dtype
        )
        z = sae.encode(flat).index_select(1, ids_t)
        z_img = z.reshape(chunk.shape[0], chunk.shape[1], len(feature_ids)).float()
        values[start : start + chunk.shape[0]] = z_img.max(dim=1).values.detach().cpu().numpy()
    return {int(fid): values[:, i] for i, fid in enumerate(feature_ids)}


def _decoder_rows(
    sae: torch.nn.Module,
    feature_ids: list[int],
    *,
    device: torch.device,
) -> dict[int, np.ndarray]:
    if not feature_ids:
        return {}
    dec = _decoder_weight(sae).float().to(device)
    ids = torch.as_tensor(feature_ids, device=device, dtype=torch.long)
    rows = dec.index_select(0, ids).detach().cpu().numpy().astype(np.float32)
    return {int(fid): rows[i] for i, fid in enumerate(feature_ids)}


def _cosine_np(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denom)


def _classify_precausal_match(row: dict, args: argparse.Namespace) -> str:
    score = float(row.get("match_score", 0.0))
    activation_corr = float(row.get("activation_corr", float("nan")))
    class_jaccard = float(row.get("top_class_jaccard", row.get("top9_class_jaccard", 0.0)))
    if (
        score >= args.shared_score_threshold
        and activation_corr >= args.shared_activation_corr_threshold
        and class_jaccard >= args.shared_class_jaccard_threshold
    ):
        return "one_to_one_shared_precausal"
    if class_jaccard >= args.shared_class_jaccard_threshold:
        return "semantic_only_or_weak_causal_candidate"
    if score >= args.min_hungarian_score:
        return "ambiguous"
    return "unmatched"


def run_match(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id, resume=args.resume)
    bank_rows = read_csv(args.feature_bank)
    grouped = _group_bank(bank_rows)
    candidate_rows: list[dict] = []
    hungarian_rows: list[dict] = []
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cells = selected_cells(args)
    for layer, t_bin in cells:
        for source, target in permutations(args.conditions, 2):
            src_feats = grouped.get((source, layer, t_bin), [])
            tgt_feats = grouped.get((target, layer, t_bin), [])
            if not src_feats or not tgt_feats:
                warn(f"match skip {source}->{target} L{layer}/T{t_bin}: empty feature set")
                continue
            src_profiles: dict[int, np.ndarray] = {}
            tgt_profiles: dict[int, np.ndarray] = {}
            src_decoders: dict[int, np.ndarray] = {}
            tgt_decoders: dict[int, np.ndarray] = {}
            src_to_tgt = None
            if not args.metadata_only:
                src_pos, tgt_pos, _ = _aligned_positions(
                    args.activations_root, source, target, args.dit_step, args.max_images, args.seed
                )
                src_acts = _read_rows(_cell_path(args.activations_root, source, args.dit_step, layer, t_bin), src_pos)
                tgt_acts = _read_rows(_cell_path(args.activations_root, target, args.dit_step, layer, t_bin), tgt_pos)
                src_sae = _load_matryoshka_sae(
                    _resolve_sae_ckpt(args.sae_root, source, layer, t_bin, args.dit_step), device
                )
                tgt_sae = _load_matryoshka_sae(
                    _resolve_sae_ckpt(args.sae_root, target, layer, t_bin, args.dit_step), device
                )
                src_ids = [feat.feature_id for feat in src_feats]
                tgt_ids = [feat.feature_id for feat in tgt_feats]
                src_profiles = _feature_profiles(
                    src_sae,
                    src_acts,
                    src_ids,
                    device=device,
                    batch_images=args.profile_batch_images,
                )
                tgt_profiles = _feature_profiles(
                    tgt_sae,
                    tgt_acts,
                    tgt_ids,
                    device=device,
                    batch_images=args.profile_batch_images,
                )
                if args.cache_profiles:
                    cache_dir = out_dir / "artifacts" / "feature_profiles"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    for fid, profile in src_profiles.items():
                        np.save(cache_dir / f"{source}_L{layer}_T{t_bin}_F{fid}.npy", profile)
                    for fid, profile in tgt_profiles.items():
                        np.save(cache_dir / f"{target}_L{layer}_T{t_bin}_F{fid}.npy", profile)
                src_decoders = _decoder_rows(src_sae, src_ids, device=device)
                tgt_decoders = _decoder_rows(tgt_sae, tgt_ids, device=device)
                src_fit, _ = _sample_tokens_np(src_acts, args.max_fit_tokens, args.seed + 51)
                tgt_fit, _ = _sample_tokens_np(tgt_acts, args.max_fit_tokens, args.seed + 51)
                src_to_tgt = fit_ridge_affine(src_fit, tgt_fit, alpha=args.ridge_alpha)
                del src_sae, tgt_sae
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            matrix = np.full((len(src_feats), len(tgt_feats)), -1e6, dtype=np.float64)
            for i, src in enumerate(src_feats):
                for j, tgt in enumerate(tgt_feats):
                    if args.same_top_class_only and src.top_class_idx != tgt.top_class_idx:
                        continue
                    activation_corr = None
                    activation_spearman = float("nan")
                    mapped_decoder_cosine = None
                    if not args.metadata_only:
                        left_profile = src_profiles.get(src.feature_id)
                        right_profile = tgt_profiles.get(tgt.feature_id)
                        if left_profile is not None and right_profile is not None:
                            activation_corr = _pearson(left_profile, right_profile)
                            activation_spearman = _spearman(left_profile, right_profile)
                        if src_to_tgt is not None and src.feature_id in src_decoders and tgt.feature_id in tgt_decoders:
                            mapped = src_decoders[src.feature_id] @ src_to_tgt.weight
                            mapped_decoder_cosine = _cosine_np(mapped, tgt_decoders[tgt.feature_id])
                    parts = feature_match_score(
                        src,
                        tgt,
                        activation_corr=activation_corr,
                        mapped_decoder_cosine=mapped_decoder_cosine,
                    )
                    if args.metadata_only:
                        parts["match_score"] = parts["metadata_score"]
                    if parts["match_score"] < args.min_candidate_score:
                        continue
                    matrix[i, j] = parts["match_score"]
                    candidate_rows.append(
                        {
                            "source": source,
                            "target": target,
                            "layer": layer,
                            "t_bin": t_bin,
                            "t_center": T_CENTERS[int(t_bin)],
                            "source_feature_id": src.feature_id,
                            "target_feature_id": tgt.feature_id,
                            "source_top_label": src.top_label,
                            "target_top_label": tgt.top_label,
                            "source_top_class_idx": src.top_class_idx,
                            "target_top_class_idx": tgt.top_class_idx,
                            "source_entropy": src.entropy,
                            "target_entropy": tgt.entropy,
                            "source_density": src.density,
                            "target_density": tgt.density,
                            "source_decoder_norm": "" if src.decoder_norm is None else src.decoder_norm,
                            "target_decoder_norm": "" if tgt.decoder_norm is None else tgt.decoder_norm,
                            "source_mean_act": src.mean_act,
                            "target_mean_act": tgt.mean_act,
                            "activation_spearman": activation_spearman,
                            **parts,
                        }
                    )
            if np.all(matrix < 0):
                warn(f"match no candidates {source}->{target} L{layer}/T{t_bin}")
                continue
            row_ind, col_ind = linear_sum_assignment(-matrix)
            kept = 0
            for i, j in zip(row_ind, col_ind, strict=True):
                score = float(matrix[i, j])
                if score < args.min_hungarian_score:
                    continue
                src = src_feats[int(i)]
                tgt = tgt_feats[int(j)]
                matching = [
                    row
                    for row in candidate_rows
                    if row["source"] == source
                    and row["target"] == target
                    and int(row["layer"]) == layer
                    and int(row["t_bin"]) == t_bin
                    and int(row["source_feature_id"]) == src.feature_id
                    and int(row["target_feature_id"]) == tgt.feature_id
                ]
                out_row = dict(matching[-1]) if matching else {
                    "source": source,
                    "target": target,
                    "layer": layer,
                    "t_bin": t_bin,
                    "t_center": T_CENTERS[int(t_bin)],
                    "source_feature_id": src.feature_id,
                    "target_feature_id": tgt.feature_id,
                    "source_top_label": src.top_label,
                    "target_top_label": tgt.top_label,
                    "source_top_class_idx": src.top_class_idx,
                    "target_top_class_idx": tgt.top_class_idx,
                    "source_entropy": src.entropy,
                    "target_entropy": tgt.entropy,
                    "source_density": src.density,
                    "target_density": tgt.density,
                    "source_decoder_norm": "" if src.decoder_norm is None else src.decoder_norm,
                    "target_decoder_norm": "" if tgt.decoder_norm is None else tgt.decoder_norm,
                    "source_mean_act": src.mean_act,
                    "target_mean_act": tgt.mean_act,
                    **feature_match_score(src, tgt),
                }
                out_row["match_type"] = _classify_precausal_match(out_row, args)
                out_row["passes_same_individual_precausal_gate"] = (
                    out_row["match_type"] == "one_to_one_shared_precausal"
                )
                hungarian_rows.append(out_row)
                kept += 1
            info(f"match {source}->{target} L{layer}/T{t_bin}: kept {kept}")
    write_csv(out_dir / "metrics" / "feature_match_candidates.csv", candidate_rows)
    write_csv(out_dir / "metrics" / "feature_match_hungarian.csv", hungarian_rows)
    _write_match_summary(hungarian_rows, out_dir / "metrics" / "feature_match_summary.csv")
    _plot_match_counts(hungarian_rows, out_dir / "plots" / "feature_match_counts.png")
    _plot_match_score_grid(hungarian_rows, out_dir / "plots" / "match_score_grid.png")
    _plot_match_type_counts(hungarian_rows, out_dir / "plots" / "match_type_counts.png")
    summary = {
        "analysis": "feature-match",
        "feature_bank": str(args.feature_bank),
        "n_candidate_rows": len(candidate_rows),
        "n_hungarian_rows": len(hungarian_rows),
        "same_top_class_only": bool(args.same_top_class_only),
        "metadata_only": bool(args.metadata_only),
        "precausal_same_individual_rows": sum(
            1 for row in hungarian_rows if row.get("match_type") == "one_to_one_shared_precausal"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_md(
        out_dir,
        "Cross-Tokenizer Feature Matching",
        [
            f"Metadata matching produced =={len(hungarian_rows)}== one-to-one feature pairs.",
            "Pairs are same-cell and directed; matching is based on class-top examples and dashboard statistics.",
            "These pairs are candidates for activation-level causal patching, not proof of shared features.",
        ],
    )
    ok(f"feature matching complete: {out_dir}")
    return 0


def _plot_match_counts(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        key = f"{row['source']}->{row['target']} L{row['layer']}/T{row['t_bin']}"
        counts[key] += 1
    labels = list(counts)
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.35 * len(labels)), 4.0))
    ax.bar(range(len(labels)), vals, color=PB["blue"])
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_ylabel("Hungarian pairs")
    ax.set_title("Feature match counts")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_match_summary(rows: list[dict], out_path: Path) -> None:
    summary_rows = []
    grouped: dict[tuple[str, str, int, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["source"]),
                str(row["target"]),
                int(row["layer"]),
                int(row["t_bin"]),
                str(row.get("match_type", "unknown")),
            )
        ].append(row)
    for (source, target, layer, t_bin, match_type), vals in sorted(grouped.items()):
        scores = [float(v.get("match_score", float("nan"))) for v in vals]
        activation = [float(v.get("activation_corr", float("nan"))) for v in vals]
        decoder = [float(v.get("mapped_decoder_cosine", float("nan"))) for v in vals]
        summary_rows.append(
            {
                "source": source,
                "target": target,
                "layer": layer,
                "t_bin": t_bin,
                "match_type": match_type,
                "n_pairs": len(vals),
                "mean_match_score": float(np.nanmean(scores)) if scores else float("nan"),
                "mean_activation_corr": float(np.nanmean(activation)) if activation else float("nan"),
                "mean_mapped_decoder_cosine": float(np.nanmean(decoder)) if decoder else float("nan"),
            }
        )
    write_csv(out_path, summary_rows)


def _plot_match_score_grid(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    keys = sorted({(str(r["source"]), str(r["target"])) for r in rows})
    cells = sorted({(int(r["layer"]), int(r["t_bin"])) for r in rows})
    arr = np.full((len(keys), len(cells)), np.nan, dtype=np.float32)
    for i, key in enumerate(keys):
        for j, cell in enumerate(cells):
            vals = [
                float(r["match_score"])
                for r in rows
                if (str(r["source"]), str(r["target"])) == key
                and (int(r["layer"]), int(r["t_bin"])) == cell
            ]
            if vals:
                arr[i, j] = float(np.nanmean(vals))
    fig, ax = plt.subplots(figsize=(max(6.5, 0.75 * len(cells)), max(3.5, 0.45 * len(keys))))
    im = ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(
        range(len(cells)),
        [f"L{layer}/T{t_bin}" for layer, t_bin in cells],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(keys)), [f"{s}->{t}" for s, t in keys])
    ax.set_title("Mean feature match score")
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_match_type_counts(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    counts = Counter(str(row.get("match_type", "unknown")) for row in rows)
    labels = list(counts)
    vals = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(labels)), 4.0))
    ax.bar(range(len(labels)), vals, color=PB["gold"])
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("Hungarian rows")
    ax.set_title("Pre-causal match classification")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _resolve_sae_ckpt(sae_root: Path, condition: str, layer: int, t_bin: int, dit_step: int) -> Path:
    base = sae_root / condition / f"L{layer}_T{t_bin}" / f"step_{dit_step:06d}"
    finals = sorted(p for p in base.glob("final_*") if p.is_dir())
    if not finals:
        raise FileNotFoundError(f"no final_* SAE dir under {base}")
    return finals[-1]


def _load_matryoshka_sae(ckpt_dir: Path, device: torch.device) -> torch.nn.Module:
    from sae_lens import MatryoshkaBatchTopKTrainingSAE

    last_error: OSError | None = None
    for attempt in range(1, 4):
        try:
            sae = MatryoshkaBatchTopKTrainingSAE.load_from_disk(str(ckpt_dir), device=str(device))
            break
        except OSError as exc:
            last_error = exc
            if attempt == 3:
                raise
            wait_s = 5 * attempt
            warn(f"SAE load failed for {ckpt_dir} with {exc!r}; retrying in {wait_s}s")
            time.sleep(wait_s)
    else:
        raise last_error if last_error is not None else RuntimeError(f"failed to load SAE from {ckpt_dir}")
    sae.eval()
    return sae


def _torch_affine(weight: np.ndarray, bias: np.ndarray, device: torch.device, dtype: torch.dtype):
    w = torch.from_numpy(weight).to(device=device, dtype=dtype)
    b = torch.from_numpy(bias).to(device=device, dtype=dtype)

    def apply(x: torch.Tensor) -> torch.Tensor:
        return x @ w + b

    return apply


def _target_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    x64 = x.astype(np.float64)
    return x64.sum(axis=0), (x64 * x64).sum(axis=0), int(x64.shape[0])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _linear_calibration(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(source) & np.isfinite(target)
    source = source[valid]
    target = target[valid]
    if source.size < 2 or float(np.var(source)) <= 1e-12:
        intercept = float(np.mean(target)) if target.size else 0.0
        return {"slope": 0.0, "intercept": intercept, "corr": float("nan"), "r2": float("nan")}
    slope, intercept = np.polyfit(source, target, deg=1)
    pred = slope * source + intercept
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "corr": _corr(source, target),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def _sample_image_block_np(acts: np.ndarray, max_tokens: int, seed: int) -> np.ndarray:
    if acts.shape[0] * acts.shape[1] <= max_tokens:
        return acts.astype(np.float32)
    n_images = max(1, min(acts.shape[0], math.ceil(max_tokens / acts.shape[1])))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(acts.shape[0], size=n_images, replace=False))
    return acts[idx].astype(np.float32)


def _active_mask_from_scores(
    scores: torch.Tensor,
    *,
    threshold: float,
    n_images: int,
    n_tokens: int,
    min_active_tokens: int,
) -> torch.Tensor:
    mask = scores > float(threshold)
    if int(mask.sum().item()) >= min_active_tokens:
        return mask
    if n_images <= 0 or n_tokens <= 0 or scores.numel() != n_images * n_tokens:
        take = min(int(scores.numel()), int(min_active_tokens))
        idx = torch.topk(scores, k=take).indices
        mask[idx] = True
        return mask
    by_image = scores.reshape(n_images, n_tokens)
    fallback = torch.zeros_like(by_image, dtype=torch.bool)
    top_pos = by_image.argmax(dim=1)
    fallback[torch.arange(n_images, device=scores.device), top_pos] = True
    mask = mask | fallback.reshape(-1)
    if int(mask.sum().item()) >= min_active_tokens:
        return mask
    take = min(int(scores.numel()), int(min_active_tokens))
    idx = torch.topk(scores, k=take).indices
    mask[idx] = True
    return mask


def _r2_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _load_match_rows(args: argparse.Namespace) -> list[dict]:
    rows = read_csv(args.match_csv)
    selected = []
    cell_set = set(selected_cells(args))
    for row in rows:
        if row["source"] not in args.conditions or row["target"] not in args.conditions:
            continue
        if (int(row["layer"]), int(row["t_bin"])) not in cell_set:
            continue
        if args.directed_pairs:
            pair = f"{row['source']}->{row['target']}"
            if pair not in args.directed_pairs:
                continue
        selected.append(row)
    selected.sort(key=lambda r: float(r.get("match_score", 0.0)), reverse=True)
    if args.max_pairs:
        selected = selected[: args.max_pairs]
    return selected


def _load_bank_features(path: Path | None) -> dict[tuple[str, int, int], list[FeatureRow]]:
    if path is None:
        return {}
    return _group_bank(read_csv(path))


def _select_density_norm_control(
    bank_index: dict[tuple[str, int, int], list[FeatureRow]],
    *,
    condition: str,
    layer: int,
    t_bin: int,
    target_feature_id: int,
    target_density: float,
    target_decoder_norm: float | None,
    rng: np.random.Generator,
    fallback_d_sae: int,
) -> int:
    candidates = [
        feat
        for feat in bank_index.get((condition, layer, t_bin), [])
        if feat.feature_id != target_feature_id
    ]
    if not candidates:
        draw = int(rng.integers(0, fallback_d_sae))
        return (draw + 1) % fallback_d_sae if draw == target_feature_id else draw
    scored = []
    for feat in candidates:
        density_penalty = abs(math.log10(max(feat.density, 1e-12)) - math.log10(max(target_density, 1e-12)))
        norm_penalty = 0.0
        if target_decoder_norm is not None and feat.decoder_norm is not None:
            norm_penalty = abs(
                math.log10(max(feat.decoder_norm, 1e-12))
                - math.log10(max(target_decoder_norm, 1e-12))
            )
        scored.append((density_penalty + norm_penalty, feat.feature_id))
    scored.sort(key=lambda row: row[0])
    top_k = scored[: min(16, len(scored))]
    return int(top_k[int(rng.integers(0, len(top_k)))][1])


def _bank_lookup(bank_index: dict[tuple[str, int, int], list[FeatureRow]]) -> dict[tuple[str, int, int, int], FeatureRow]:
    return {
        (feat.condition, feat.layer, feat.t_bin, feat.feature_id): feat
        for feats in bank_index.values()
        for feat in feats
    }


def _select_group_control_features(
    bank_index: dict[tuple[str, int, int], list[FeatureRow]],
    bank_by_feature: dict[tuple[str, int, int, int], FeatureRow],
    *,
    condition: str,
    layer: int,
    t_bin: int,
    target_feature_ids: list[int],
    rng: np.random.Generator,
    fallback_d_sae: int,
) -> list[int]:
    target_set = set(int(fid) for fid in target_feature_ids)
    out: list[int] = []
    used = set(target_set)
    for target_fid in target_feature_ids:
        target_feat = bank_by_feature.get((condition, layer, t_bin, int(target_fid)))
        if target_feat is None:
            candidates = [
                fid for fid in range(fallback_d_sae)
                if fid not in used
            ]
            if not candidates:
                out.append(int(target_fid))
                continue
            choice = int(candidates[int(rng.integers(0, len(candidates)))])
            used.add(choice)
            out.append(choice)
            continue
        choice = _select_density_norm_control(
            bank_index,
            condition=condition,
            layer=layer,
            t_bin=t_bin,
            target_feature_id=int(target_fid),
            target_density=target_feat.density,
            target_decoder_norm=target_feat.decoder_norm,
            rng=rng,
            fallback_d_sae=fallback_d_sae,
        )
        if choice in used:
            candidates = [
                feat.feature_id
                for feat in bank_index.get((condition, layer, t_bin), [])
                if feat.feature_id not in used
            ]
            if candidates:
                choice = int(candidates[int(rng.integers(0, len(candidates)))])
        used.add(int(choice))
        out.append(int(choice))
    return out


def _fit_ridge_group_calibration(source: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, object]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("group calibration expects 2-D source and target matrices")
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if source.shape[0] < 2:
        weight = np.zeros((source.shape[1], target.shape[1]), dtype=np.float64)
        bias = np.zeros(target.shape[1], dtype=np.float64)
        return {"weight": weight, "bias": bias, "mean_corr": float("nan"), "mean_r2": float("nan")}
    x_mean = source.mean(axis=0, keepdims=True)
    y_mean = target.mean(axis=0, keepdims=True)
    xc = source - x_mean
    yc = target - y_mean
    reg = float(alpha) * np.eye(xc.shape[1], dtype=np.float64)
    weight = np.linalg.solve(xc.T @ xc + reg, xc.T @ yc)
    bias = (y_mean - x_mean @ weight).reshape(-1)
    pred = source @ weight + bias
    corrs = [_corr(pred[:, j], target[:, j]) for j in range(target.shape[1])]
    r2s = [_r2_np(pred[:, j], target[:, j]) for j in range(target.shape[1])]
    return {
        "weight": weight,
        "bias": bias,
        "mean_corr": float(np.nanmean(corrs)) if any(np.isfinite(c) for c in corrs) else float("nan"),
        "mean_r2": float(np.nanmean(r2s)) if any(np.isfinite(r) for r in r2s) else float("nan"),
    }


def _load_group_tasks(args: argparse.Namespace) -> list[dict]:
    rows = read_csv(args.group_members_csv)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        layer = int(row["layer"])
        t_bin = int(row["t_bin"])
        if (layer, t_bin) not in set(selected_cells(args)):
            continue
        by_group[str(row["group_id"])].append(row)
    tasks: list[dict] = []
    requested = set(args.group_ids or [])
    for group_id, members in sorted(by_group.items()):
        if requested and group_id not in requested:
            continue
        conditions = sorted({str(row["condition"]) for row in members if str(row["condition"]) in args.conditions})
        if len(conditions) < args.min_group_conditions:
            continue
        layers = {int(row["layer"]) for row in members}
        t_bins = {int(row["t_bin"]) for row in members}
        if len(layers) != 1 or len(t_bins) != 1:
            warn(f"skipping multi-cell group {group_id}: layers={sorted(layers)} t_bins={sorted(t_bins)}")
            continue
        layer = next(iter(layers))
        t_bin = next(iter(t_bins))
        members_by_condition: dict[str, list[dict]] = defaultdict(list)
        for row in members:
            if str(row["condition"]) in conditions:
                members_by_condition[str(row["condition"])].append(row)
        for source, target in permutations(conditions, 2):
            pair = f"{source}->{target}"
            if args.directed_pairs and pair not in args.directed_pairs:
                continue
            source_features = sorted({int(row["feature_id"]) for row in members_by_condition[source]})
            target_features = sorted({int(row["feature_id"]) for row in members_by_condition[target]})
            if not source_features or not target_features:
                continue
            if args.max_source_features:
                source_features = source_features[: args.max_source_features]
            if args.max_target_features:
                target_features = target_features[: args.max_target_features]
            labels = [str(row.get("top_label", "")) for row in members if row.get("top_label", "")]
            label = Counter(labels).most_common(1)[0][0] if labels else ""
            tasks.append(
                {
                    "group_id": group_id,
                    "family_label": label,
                    "source": source,
                    "target": target,
                    "layer": layer,
                    "t_bin": t_bin,
                    "source_feature_ids": source_features,
                    "target_feature_ids": target_features,
                }
            )
    tasks.sort(key=lambda row: (row["group_id"], row["source"], row["target"]))
    if args.max_groups:
        tasks = tasks[: args.max_groups]
    return tasks


def run_group_activation_patch(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id, resume=args.resume)
    tasks = _load_group_tasks(args)
    if not tasks:
        raise ValueError("no feature-family group tasks selected")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)
    bank_index = _load_bank_features(args.feature_bank)
    bank_by_feature = _bank_lookup(bank_index)
    result_rows: list[dict] = []
    modes = (
        "native_reconstruction",
        "reconstruction_only",
        "group_native_ablate",
        "group_native_clamp",
        "group_transfer_replace",
        "random_group_control",
        "shuffled_group_control",
    )
    for task in tasks:
        source = str(task["source"])
        target = str(task["target"])
        layer = int(task["layer"])
        t_bin = int(task["t_bin"])
        source_ids = [int(fid) for fid in task["source_feature_ids"]]
        target_ids = [int(fid) for fid in task["target_feature_ids"]]
        info(
            f"group activation patch {task['group_id']} {source}->{target} "
            f"L{layer}/T{t_bin}: {len(source_ids)} source -> {len(target_ids)} target features"
        )
        src_pos, tgt_pos, _ = _aligned_positions(
            args.activations_root, source, target, args.dit_step, args.max_images, args.seed
        )
        src_acts = _read_rows(_cell_path(args.activations_root, source, args.dit_step, layer, t_bin), src_pos)
        tgt_acts = _read_rows(_cell_path(args.activations_root, target, args.dit_step, layer, t_bin), tgt_pos)
        src_fit, _ = _sample_tokens_np(src_acts, args.max_fit_tokens, args.seed + 101)
        tgt_fit, _ = _sample_tokens_np(tgt_acts, args.max_fit_tokens, args.seed + 101)
        tgt_eval_img = _sample_image_block_np(tgt_acts, args.max_eval_tokens, args.seed + 103)
        tgt_eval = tgt_eval_img.reshape(-1, tgt_eval_img.shape[-1])
        tgt_to_src = fit_ridge_affine(tgt_fit, src_fit, alpha=args.ridge_alpha)

        target_sae = _load_matryoshka_sae(_resolve_sae_ckpt(args.sae_root, target, layer, t_bin, args.dit_step), device)
        source_sae = _load_matryoshka_sae(_resolve_sae_ckpt(args.sae_root, source, layer, t_bin, args.dit_step), device)
        sae_dtype = next(target_sae.parameters()).dtype
        tgt_to_src_t = _torch_affine(tgt_to_src.weight, tgt_to_src.bias, device, sae_dtype)

        fit_tensor = torch.from_numpy(tgt_fit).to(device=device, dtype=sae_dtype)
        fit_src_tensor = tgt_to_src_t(fit_tensor)
        z_tgt_fit_parts: list[torch.Tensor] = []
        z_src_fit_parts: list[torch.Tensor] = []
        for start in range(0, fit_tensor.shape[0], args.sae_batch_size):
            xb = fit_tensor[start : start + args.sae_batch_size]
            sb = fit_src_tensor[start : start + args.sae_batch_size]
            z_tgt_fit_parts.append(target_sae.encode(xb).detach().float().cpu())
            z_src_fit_parts.append(source_sae.encode(sb).detach().float().cpu())
        z_tgt_fit = torch.cat(z_tgt_fit_parts, dim=0).numpy()
        z_src_fit = torch.cat(z_src_fit_parts, dim=0).numpy()
        source_fit_group = z_src_fit[:, source_ids]
        target_fit_group = z_tgt_fit[:, target_ids]
        calib = _fit_ridge_group_calibration(source_fit_group, target_fit_group, args.group_ridge_alpha)
        active_scores_fit = np.max(source_fit_group, axis=1)
        active_threshold = float(np.quantile(active_scores_fit, args.active_quantile))
        clamp_values = np.quantile(target_fit_group, args.clamp_quantile, axis=0).astype(np.float64)
        control_ids = _select_group_control_features(
            bank_index,
            bank_by_feature,
            condition=target,
            layer=layer,
            t_bin=t_bin,
            target_feature_ids=target_ids,
            rng=rng,
            fallback_d_sae=int(target_sae.cfg.d_sae),
        )

        accums = {mode: MetricAccumulator() for mode in modes}
        target_tensor = torch.from_numpy(tgt_eval).to(device=device, dtype=sae_dtype)
        n_tokens = int(tgt_eval_img.shape[1])
        coeff_source: list[np.ndarray] = []
        coeff_target: list[np.ndarray] = []
        coeff_calibrated: list[np.ndarray] = []
        weight_t = torch.from_numpy(np.asarray(calib["weight"])).to(device=device, dtype=sae_dtype)
        bias_t = torch.from_numpy(np.asarray(calib["bias"])).to(device=device, dtype=sae_dtype)
        clamp_t = torch.from_numpy(clamp_values).to(device=device, dtype=sae_dtype)
        target_idx = torch.tensor(target_ids, device=device, dtype=torch.long)
        control_idx = torch.tensor(control_ids, device=device, dtype=torch.long)
        for start in range(0, target_tensor.shape[0], args.sae_batch_size):
            xb = target_tensor[start : start + args.sae_batch_size]
            src_xb = tgt_to_src_t(xb)
            z_tgt = target_sae.encode(xb)
            z_src = source_sae.encode(src_xb)
            native_recon = target_sae.decode(z_tgt)
            recon_err = xb - native_recon
            source_group = z_src[:, source_ids]
            target_group = z_tgt[:, target_ids]
            calibrated = source_group @ weight_t + bias_t
            coeff_source.append(source_group.detach().float().cpu().numpy())
            coeff_target.append(target_group.detach().float().cpu().numpy())
            coeff_calibrated.append(calibrated.detach().float().cpu().numpy())
            source_scores = source_group.max(dim=1).values
            batch_n_images = int(xb.shape[0] // n_tokens) if n_tokens > 0 else 0
            if batch_n_images * n_tokens != int(xb.shape[0]):
                batch_n_images = 0
            mask = _active_mask_from_scores(
                source_scores,
                threshold=active_threshold,
                n_images=batch_n_images,
                n_tokens=n_tokens,
                min_active_tokens=args.min_active_tokens,
            )
            if not bool(mask.any()):
                continue
            target_masked = xb[mask]
            native_masked = native_recon[mask]
            err_masked = recon_err[mask]
            z_masked = z_tgt[mask]
            calibrated_masked = calibrated[mask]

            accums["native_reconstruction"].update(native_masked, target_masked)
            accums["reconstruction_only"].update(native_masked + err_masked, target_masked)

            ablated = z_masked.clone()
            ablated.index_fill_(1, target_idx, 0.0)
            accums["group_native_ablate"].update(target_sae.decode(ablated) + err_masked, target_masked)

            clamped = z_masked.clone()
            clamped[:, target_ids] = clamp_t
            accums["group_native_clamp"].update(target_sae.decode(clamped) + err_masked, target_masked)

            patched = z_masked.clone()
            patched[:, target_ids] = calibrated_masked
            accums["group_transfer_replace"].update(target_sae.decode(patched) + err_masked, target_masked)

            order = torch.randperm(calibrated_masked.shape[0], device=device)
            shuffled = z_masked.clone()
            shuffled[:, target_ids] = calibrated_masked[order]
            accums["shuffled_group_control"].update(target_sae.decode(shuffled) + err_masked, target_masked)

            random_control = z_masked.clone()
            random_control[:, control_idx] = calibrated_masked
            accums["random_group_control"].update(target_sae.decode(random_control) + err_masked, target_masked)

        native_metrics = accums["reconstruction_only"].finalize()
        source_all = np.concatenate(coeff_source, axis=0)
        target_all = np.concatenate(coeff_target, axis=0)
        calibrated_all = np.concatenate(coeff_calibrated, axis=0)
        group_corrs = [_corr(calibrated_all[:, j], target_all[:, j]) for j in range(target_all.shape[1])]
        group_r2s = [_r2_np(calibrated_all[:, j], target_all[:, j]) for j in range(target_all.shape[1])]
        active_frac = float(np.mean(np.max(source_all, axis=1) > active_threshold))
        for mode in modes:
            metrics = accums[mode].finalize()
            result_rows.append(
                {
                    "group_id": task["group_id"],
                    "family_label": task["family_label"],
                    "source": source,
                    "target": target,
                    "layer": layer,
                    "t_bin": t_bin,
                    "t_center": T_CENTERS[int(t_bin)],
                    "source_feature_ids": source_ids,
                    "target_feature_ids": target_ids,
                    "control_target_feature_ids": control_ids if mode == "random_group_control" else "",
                    "n_source_features": len(source_ids),
                    "n_target_features": len(target_ids),
                    "mode": mode,
                    "group_calibrated_corr": float(np.nanmean(group_corrs)) if any(np.isfinite(c) for c in group_corrs) else float("nan"),
                    "group_calibrated_r2": float(np.nanmean(group_r2s)) if any(np.isfinite(r) for r in group_r2s) else float("nan"),
                    "calibration_corr": calib["mean_corr"],
                    "calibration_r2": calib["mean_r2"],
                    "patch_active_frac": active_frac,
                    "active_threshold": active_threshold,
                    "n_fit_tokens": int(src_fit.shape[0]),
                    "n_eval_tokens": int(tgt_eval.shape[0]),
                    "mse": metrics["mse"],
                    "cosine": metrics["cosine"],
                    "ev": metrics["ev"],
                    "delta_mse_vs_reconstruction_only": metrics["mse"] - native_metrics["mse"],
                    "delta_cosine_vs_reconstruction_only": metrics["cosine"] - native_metrics["cosine"],
                    "delta_ev_vs_reconstruction_only": metrics["ev"] - native_metrics["ev"],
                    "delta_mse_vs_native": metrics["mse"] - native_metrics["mse"],
                    "delta_cosine_vs_native": metrics["cosine"] - native_metrics["cosine"],
                    "delta_ev_vs_native": metrics["ev"] - native_metrics["ev"],
                }
            )
        del target_sae, source_sae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(out_dir / "metrics" / "group_activation_patching.csv", result_rows)
    control_rows = [
        row
        for row in result_rows
        if str(row["mode"]).endswith("_control")
    ]
    write_csv(out_dir / "metrics" / "group_activation_controls.csv", control_rows)
    _plot_patch_summary(result_rows, out_dir / "plots" / "group_activation_delta_ev.png")
    summary = {
        "analysis": "group-activation-patching",
        "group_members_csv": str(args.group_members_csv),
        "n_rows": len(result_rows),
        "n_group_tasks": len({(r["group_id"], r["source"], r["target"], r["layer"], r["t_bin"]) for r in result_rows}),
        "active_quantile": args.active_quantile,
        "min_active_tokens": args.min_active_tokens,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "metrics" / "group_activation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_summary_md(
        out_dir,
        "Group-Level Activation Patching",
        [
            f"Patched =={summary['n_group_tasks']}== directed feature-family group transfers in activation space.",
            "Group transfer uses multivariate ridge calibration from source SAE coefficients to target SAE coefficients.",
            "Controls include group native ablation/clamp, shuffled pairing, and random matched target-feature groups.",
        ],
    )
    ok(f"group activation patching complete: {out_dir}")
    return 0


def run_activation_patch(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id, resume=args.resume)
    rows = _load_match_rows(args)
    if not rows:
        raise ValueError("no match rows selected for activation patching")
    groups: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["target"], int(row["layer"]), int(row["t_bin"]))].append(row)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    result_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    bank_index = _load_bank_features(args.feature_bank)
    for (source, target, layer, t_bin), group in groups.items():
        group = group[: args.max_pairs_per_group] if args.max_pairs_per_group else group
        info(f"activation patch {source}->{target} L{layer}/T{t_bin}: {len(group)} pairs")
        src_pos, tgt_pos, _ = _aligned_positions(
            args.activations_root, source, target, args.dit_step, args.max_images, args.seed
        )
        src_acts = _read_rows(_cell_path(args.activations_root, source, args.dit_step, layer, t_bin), src_pos)
        tgt_acts = _read_rows(_cell_path(args.activations_root, target, args.dit_step, layer, t_bin), tgt_pos)
        src_fit, _ = _sample_tokens_np(src_acts, args.max_fit_tokens, args.seed + 41)
        tgt_fit, _ = _sample_tokens_np(tgt_acts, args.max_fit_tokens, args.seed + 41)
        tgt_eval_img = _sample_image_block_np(tgt_acts, args.max_eval_tokens, args.seed + 43)
        tgt_eval = tgt_eval_img.reshape(-1, tgt_eval_img.shape[-1])
        tgt_to_src = fit_ridge_affine(tgt_fit, src_fit, alpha=args.ridge_alpha)
        src_to_tgt = fit_ridge_affine(src_fit, tgt_fit, alpha=args.ridge_alpha)
        native_probe = alignment_metrics(tgt_to_src.apply(tgt_fit[: min(2048, tgt_fit.shape[0])]), src_fit[: min(2048, src_fit.shape[0])])

        target_sae = _load_matryoshka_sae(_resolve_sae_ckpt(args.sae_root, target, layer, t_bin, args.dit_step), device)
        source_sae = _load_matryoshka_sae(_resolve_sae_ckpt(args.sae_root, source, layer, t_bin, args.dit_step), device)
        sae_dtype = next(target_sae.parameters()).dtype
        tgt_to_src_t = _torch_affine(tgt_to_src.weight, tgt_to_src.bias, device, sae_dtype)
        source_to_target_w = torch.from_numpy(src_to_tgt.weight).to(device=device, dtype=sae_dtype)
        src_decoder_rows = _decoder_weight(source_sae).to(device=device, dtype=sae_dtype)

        accums: dict[tuple[int, str], MetricAccumulator] = {}
        coeff_values: dict[int, dict[str, list[np.ndarray]]] = {}
        pair_specs = []
        fit_tensor = torch.from_numpy(tgt_fit).to(device=device, dtype=sae_dtype)
        fit_src_tensor = tgt_to_src_t(fit_tensor)
        z_tgt_fit_parts: list[torch.Tensor] = []
        z_src_fit_parts: list[torch.Tensor] = []
        for start in range(0, fit_tensor.shape[0], args.sae_batch_size):
            xb = fit_tensor[start : start + args.sae_batch_size]
            sb = fit_src_tensor[start : start + args.sae_batch_size]
            z_tgt_fit_parts.append(target_sae.encode(xb).detach().float().cpu())
            z_src_fit_parts.append(source_sae.encode(sb).detach().float().cpu())
        z_tgt_fit = torch.cat(z_tgt_fit_parts, dim=0).numpy()
        z_src_fit = torch.cat(z_src_fit_parts, dim=0).numpy()
        del fit_tensor, fit_src_tensor, z_tgt_fit_parts, z_src_fit_parts
        for idx, row in enumerate(group):
            src_fid = int(row["source_feature_id"])
            tgt_fid = int(row["target_feature_id"])
            source_fit_vals = z_src_fit[:, src_fid]
            target_fit_vals = z_tgt_fit[:, tgt_fid]
            calib = _linear_calibration(source_fit_vals, target_fit_vals)
            active_threshold = float(np.quantile(source_fit_vals, args.active_quantile))
            target_density = float(row.get("target_density", 0.0))
            target_decoder_norm = _parse_optional_float(row.get("target_decoder_norm", ""))
            control_target_fid = _select_density_norm_control(
                bank_index,
                condition=target,
                layer=layer,
                t_bin=t_bin,
                target_feature_id=tgt_fid,
                target_density=target_density,
                target_decoder_norm=target_decoder_norm,
                rng=rng,
                fallback_d_sae=int(target_sae.cfg.d_sae),
            )
            source_direction = src_decoder_rows[src_fid] @ source_to_target_w
            pair_specs.append(
                (
                    idx,
                    row,
                    src_fid,
                    tgt_fid,
                    control_target_fid,
                    calib,
                    active_threshold,
                    source_direction.detach().clone(),
                )
            )
            coeff_values[idx] = {"source": [], "target": [], "calibrated_source": []}
            for mode in (
                "native_reconstruction",
                "reconstruction_only",
                "native_ablate",
                "native_clamp",
                "transfer_replace",
                "source_direction_inject",
                "random_matched_control",
                "shuffled_pairing_control",
            ):
                accums[(idx, mode)] = MetricAccumulator()

        target_tensor = torch.from_numpy(tgt_eval).to(device=device, dtype=sae_dtype)
        n_tokens = int(tgt_eval_img.shape[1])
        for start in range(0, target_tensor.shape[0], args.sae_batch_size):
            xb = target_tensor[start : start + args.sae_batch_size]
            src_xb = tgt_to_src_t(xb)
            z_tgt = target_sae.encode(xb)
            z_src = source_sae.encode(src_xb)
            native_recon = target_sae.decode(z_tgt)
            recon_err = xb - native_recon
            global_start = start
            global_stop = start + xb.shape[0]
            for idx, _row, src_fid, tgt_fid, control_target_fid, calib, active_threshold, source_direction in pair_specs:
                target_vals = z_tgt[:, tgt_fid].detach().float().cpu().numpy()
                source_raw = z_src[:, src_fid]
                source_vals = source_raw.detach().float().cpu().numpy()
                calibrated = source_raw * float(calib["slope"]) + float(calib["intercept"])
                coeff_values[idx]["target"].append(target_vals)
                coeff_values[idx]["source"].append(source_vals)
                coeff_values[idx]["calibrated_source"].append(calibrated.detach().float().cpu().numpy())

                full_source_scores = torch.from_numpy(
                    np.concatenate(coeff_values[idx]["source"], axis=0)
                ).to(device=device, dtype=sae_dtype)
                current_scores = full_source_scores[global_start:global_stop]
                batch_n_images = int(xb.shape[0] // n_tokens) if n_tokens > 0 else 0
                if batch_n_images * n_tokens != int(xb.shape[0]):
                    batch_n_images = 0
                mask = _active_mask_from_scores(
                    current_scores,
                    threshold=active_threshold,
                    n_images=batch_n_images,
                    n_tokens=n_tokens,
                    min_active_tokens=args.min_active_tokens,
                )
                if not bool(mask.any()):
                    continue
                target_masked = xb[mask]
                native_masked = native_recon[mask]
                err_masked = recon_err[mask]
                z_masked = z_tgt[mask]
                calibrated_masked = calibrated[mask]

                accums[(idx, "native_reconstruction")].update(native_masked, target_masked)
                accums[(idx, "reconstruction_only")].update(native_masked + err_masked, target_masked)

                ablated = z_masked.clone()
                ablated[:, tgt_fid] = 0
                accums[(idx, "native_ablate")].update(target_sae.decode(ablated) + err_masked, target_masked)

                clamped = z_masked.clone()
                clamp_value = float(np.quantile(z_tgt_fit[:, tgt_fid], args.clamp_quantile))
                clamped[:, tgt_fid] = clamp_value
                accums[(idx, "native_clamp")].update(target_sae.decode(clamped) + err_masked, target_masked)

                patched = z_masked.clone()
                patched[:, tgt_fid] = calibrated_masked
                accums[(idx, "transfer_replace")].update(target_sae.decode(patched) + err_masked, target_masked)

                order = torch.randperm(calibrated_masked.shape[0], device=device)
                shuffled = z_masked.clone()
                shuffled[:, tgt_fid] = calibrated_masked[order]
                accums[(idx, "shuffled_pairing_control")].update(
                    target_sae.decode(shuffled) + err_masked, target_masked
                )

                random_control = z_masked.clone()
                random_control[:, control_target_fid] = calibrated_masked
                accums[(idx, "random_matched_control")].update(
                    target_sae.decode(random_control) + err_masked, target_masked
                )

                delta = (calibrated_masked - z_masked[:, tgt_fid]).unsqueeze(1)
                direction = source_direction.to(device=device, dtype=sae_dtype).unsqueeze(0)
                accums[(idx, "source_direction_inject")].update(
                    native_masked + err_masked + delta * direction,
                    target_masked,
                )

        for idx, row, src_fid, tgt_fid, control_target_fid, calib, active_threshold, _source_direction in pair_specs:
            native_metrics = accums[(idx, "reconstruction_only")].finalize()
            source_all = np.concatenate(coeff_values[idx]["source"])
            target_all = np.concatenate(coeff_values[idx]["target"])
            calibrated_all = np.concatenate(coeff_values[idx]["calibrated_source"])
            feature_corr = _corr(source_all, target_all)
            calibrated_corr = _corr(calibrated_all, target_all)
            calibrated_r2 = _r2_np(calibrated_all, target_all)
            source_active_frac = float(np.mean(source_all > 0))
            target_active_frac = float(np.mean(target_all > 0))
            active_frac = float(np.mean(source_all > active_threshold))
            for mode in (
                "native_reconstruction",
                "reconstruction_only",
                "native_ablate",
                "native_clamp",
                "transfer_replace",
                "source_direction_inject",
                "random_matched_control",
                "shuffled_pairing_control",
            ):
                metrics = accums[(idx, mode)].finalize()
                result_rows.append(
                    {
                        "source": source,
                        "target": target,
                        "layer": layer,
                        "t_bin": t_bin,
                        "t_center": T_CENTERS[int(t_bin)],
                        "source_feature_id": src_fid,
                        "target_feature_id": tgt_fid,
                        "control_target_feature_id": control_target_fid if mode == "random_matched_control" else "",
                        "mode": mode,
                        "match_score": float(row.get("match_score", 0.0)),
                        "match_type": row.get("match_type", ""),
                        "source_top_label": row.get("source_top_label", ""),
                        "target_top_label": row.get("target_top_label", ""),
                        "feature_activation_corr": feature_corr,
                        "calibrated_feature_corr": calibrated_corr,
                        "calibrated_feature_r2": calibrated_r2,
                        "source_feature_active_frac": source_active_frac,
                        "target_feature_active_frac": target_active_frac,
                        "patch_active_frac": active_frac,
                        "active_threshold": active_threshold,
                        "calibration_slope": calib["slope"],
                        "calibration_intercept": calib["intercept"],
                        "calibration_corr": calib["corr"],
                        "calibration_r2": calib["r2"],
                        "alignment_fit_cosine_probe": native_probe["cosine"],
                        "n_fit_tokens": int(src_fit.shape[0]),
                        "n_eval_tokens": int(tgt_eval.shape[0]),
                        "mse": metrics["mse"],
                        "cosine": metrics["cosine"],
                        "ev": metrics["ev"],
                        "delta_mse_vs_reconstruction_only": metrics["mse"] - native_metrics["mse"],
                        "delta_cosine_vs_reconstruction_only": metrics["cosine"] - native_metrics["cosine"],
                        "delta_ev_vs_reconstruction_only": metrics["ev"] - native_metrics["ev"],
                        "delta_mse_vs_native": metrics["mse"] - native_metrics["mse"],
                        "delta_cosine_vs_native": metrics["cosine"] - native_metrics["cosine"],
                        "delta_ev_vs_native": metrics["ev"] - native_metrics["ev"],
                    }
                )
        del target_sae, source_sae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(out_dir / "metrics" / "activation_feature_patching.csv", result_rows)
    control_rows = [
        row
        for row in result_rows
        if str(row["mode"]).endswith("_control") or str(row["mode"]) == "shuffled_pairing_control"
    ]
    write_csv(out_dir / "metrics" / "activation_feature_controls.csv", control_rows)
    _plot_patch_summary(result_rows, out_dir / "plots" / "activation_feature_patching_delta_ev.png")
    _plot_patch_summary(result_rows, out_dir / "plots" / "feature_patch_effects.png")
    _plot_restoration_by_match_type(result_rows, out_dir / "plots" / "restoration_by_match_type.png")
    summary = {
        "analysis": "activation-feature-patching",
        "match_csv": str(args.match_csv),
        "n_rows": len(result_rows),
        "n_feature_pairs": len({(r["source"], r["target"], r["layer"], r["t_bin"], r["source_feature_id"], r["target_feature_id"]) for r in result_rows}),
        "active_quantile": args.active_quantile,
        "min_active_tokens": args.min_active_tokens,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "metrics" / "activation_feature_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_summary_md(
        out_dir,
        "Feature-Level Activation Patching",
        [
            f"Patched =={summary['n_feature_pairs']}== directed feature pairs in activation space.",
            "Transfer patches use a calibrated source coefficient and preserve the target SAE reconstruction error.",
            "Controls include native ablation/clamp, shuffled pairing, random matched target feature, and source-direction injection.",
        ],
    )
    ok(f"activation feature patching complete: {out_dir}")
    return 0


def _sample_tokens_np(acts: np.ndarray, max_tokens: int, seed: int) -> tuple[np.ndarray, None]:
    flat = acts.reshape(-1, acts.shape[-1]).astype(np.float32)
    if flat.shape[0] <= max_tokens:
        return flat, None
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], size=max_tokens, replace=False)
    return flat[idx], None


def _plot_patch_summary(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    by_mode: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["mode"] in {"native", "native_reconstruction", "reconstruction_only"}:
            continue
        by_mode[str(row["mode"])].append(float(row["delta_ev_vs_native"]))
    labels = list(by_mode)
    vals = [float(np.nanmean(by_mode[label])) for label in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axhline(0.0, color=PB["dark"], linewidth=1)
    ax.bar(range(len(labels)), vals, color=[PB["blue"], PB["gold"], PB["red"], PB["dark"]][: len(labels)])
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("mean delta EV vs reconstruction-only")
    ax.set_title("Feature patching reconstruction effect")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_restoration_by_match_type(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["mode"] != "transfer_replace":
            continue
        match_type = str(row.get("match_type", "unknown") or "unknown")
        grouped[match_type].append(float(row["delta_ev_vs_native"]))
    if not grouped:
        return
    labels = list(grouped)
    vals = [float(np.nanmean(grouped[label])) for label in labels]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(labels)), 4.0))
    ax.axhline(0.0, color=PB["dark"], linewidth=1)
    ax.bar(range(len(labels)), vals, color=PB["blue"])
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("mean transfer delta EV vs reconstruction-only")
    ax.set_title("Transfer effect by match type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _clean_optional(raw: object) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    return "" if text in {"", "__NA__", "None", "nan"} else text


def _optional_int(raw: object) -> int | None:
    text = _clean_optional(raw)
    if not text:
        return None
    return int(float(text))


def _optional_float(raw: object) -> float | None:
    text = _clean_optional(raw)
    if not text:
        return None
    value = float(text)
    return None if math.isnan(value) else value


def _optional_bool(raw: object) -> bool | None:
    text = _clean_optional(raw)
    if not text:
        return None
    return text.lower() in {"1", "true", "yes"}


def _json_dict(raw: object) -> dict:
    text = _clean_optional(raw)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sampling_key_from_task(row: dict) -> tuple:
    mode = _clean_optional(row.get("mode"))
    target = _clean_optional(row.get("target"))
    source = _clean_optional(row.get("source"))
    layer = _optional_int(row.get("layer"))
    t_bin = _optional_int(row.get("t_bin"))
    source_feature_id = _optional_int(row.get("source_feature_id"))
    target_feature_id = _optional_int(row.get("target_feature_id"))
    random_source_feature_id = _optional_int(row.get("random_source_feature_id"))
    wrong_t_bin = _optional_int(row.get("wrong_t_bin"))
    if mode == "baseline":
        return ("baseline", target)
    if mode in {"native_ablate", "native_clamp"}:
        return ("native", mode, target, layer, t_bin, target_feature_id)
    if mode == "random_matched_control":
        return ("random", mode, source, target, layer, t_bin, target_feature_id, random_source_feature_id)
    if mode == "wrong_window_control":
        return ("wrong", mode, source, target, layer, t_bin, source_feature_id, target_feature_id, wrong_t_bin)
    return ("transfer", mode, source, target, layer, t_bin, source_feature_id, target_feature_id)


def _sampling_key_from_result(row: dict) -> tuple:
    mode = _clean_optional(row.get("mode"))
    target = _clean_optional(row.get("target_condition"))
    source = _clean_optional(row.get("source_condition"))
    layer = _optional_int(row.get("layer"))
    t_bin = _optional_int(row.get("t_bin"))
    source_feature_id = _optional_int(row.get("source_feature_id"))
    target_feature_id = _optional_int(row.get("target_feature_id"))
    random_source_feature_id = _optional_int(row.get("random_source_feature_id"))
    wrong_t_bin = _optional_int(row.get("wrong_t_bin"))
    if mode == "baseline":
        return ("baseline", target)
    if mode in {"native_ablate", "native_clamp"}:
        return ("native", mode, target, layer, t_bin, target_feature_id)
    if mode == "random_matched_control":
        return ("random", mode, source, target, layer, t_bin, target_feature_id, random_source_feature_id)
    if mode == "wrong_window_control":
        return ("wrong", mode, source, target, layer, t_bin, source_feature_id, target_feature_id, wrong_t_bin)
    return ("transfer", mode, source, target, layer, t_bin, source_feature_id, target_feature_id)


def _candidate_key(row: dict) -> tuple[str, str, int | None, int | None, int | None, int | None]:
    return (
        _clean_optional(row.get("source")),
        _clean_optional(row.get("target")),
        _optional_int(row.get("layer")),
        _optional_int(row.get("t_bin")),
        _optional_int(row.get("source_feature_id")),
        _optional_int(row.get("target_feature_id")),
    )


def _index_sampling_targets(target_rows: list[dict]) -> dict[tuple, dict]:
    return {_candidate_key(row): row for row in target_rows}


def _expand_sampling_rows(
    result_rows: list[dict],
    task_rows: list[dict],
    target_rows: list[dict],
) -> list[dict]:
    result_by_key = {_sampling_key_from_result(row): row for row in result_rows}
    target_by_key = _index_sampling_targets(target_rows)
    baselines: dict[str, float] = {}
    for row in result_rows:
        if _clean_optional(row.get("mode")) == "baseline":
            target = _clean_optional(row.get("target_condition"))
            fid = _optional_float(row.get("fid"))
            if fid is not None:
                baselines[target] = fid

    native_first_task: dict[tuple, str] = {}
    for task in task_rows:
        key = _sampling_key_from_task(task)
        if key and key[0] == "native":
            native_first_task.setdefault(key, _clean_optional(task.get("task_id")))

    rows = []
    for task in task_rows:
        key = _sampling_key_from_task(task)
        result = result_by_key.get(key)
        mode = _clean_optional(task.get("mode"))
        source = _clean_optional(task.get("source"))
        target = _clean_optional(task.get("target"))
        target_meta = target_by_key.get(_candidate_key(task), {})
        fid = _optional_float(result.get("fid") if result else None)
        baseline_fid = fid if mode == "baseline" else baselines.get(target)
        hook = _json_dict(result.get("hook_stats") if result else None)
        reused_native = bool(
            key
            and key[0] == "native"
            and _clean_optional(task.get("task_id")) != native_first_task.get(key, "")
        )
        delta_fid = (
            float(fid - baseline_fid)
            if fid is not None and baseline_fid is not None
            else float("nan")
        )
        rows.append(
            {
                "task_id": _clean_optional(task.get("task_id")),
                "candidate_id": _clean_optional(task.get("candidate_id")),
                "selection_role": _clean_optional(target_meta.get("selection_role"))
                or ("baseline" if mode == "baseline" else ""),
                "mode": mode,
                "source": source,
                "target": target,
                "layer": _optional_int(task.get("layer")) or "",
                "t_bin": _optional_int(task.get("t_bin")) or "",
                "source_feature_id": _optional_int(task.get("source_feature_id")) or "",
                "target_feature_id": _optional_int(task.get("target_feature_id")) or "",
                "random_source_feature_id": _optional_int(task.get("random_source_feature_id")) or "",
                "wrong_t_bin": _optional_int(task.get("wrong_t_bin")) or "",
                "source_top_label": _clean_optional(target_meta.get("source_top_label")),
                "target_top_label": _clean_optional(target_meta.get("target_top_label")),
                "match_score": _optional_float(target_meta.get("match_score")),
                "specificity_delta_ev": _optional_float(target_meta.get("specificity_delta_ev")),
                "activation_transfer_delta_ev": _optional_float(target_meta.get("transfer_delta_ev")),
                "activation_random_control_delta_ev": _optional_float(target_meta.get("random_control_delta_ev")),
                "activation_shuffled_control_delta_ev": _optional_float(target_meta.get("shuffled_control_delta_ev")),
                "calibrated_feature_corr": _optional_float(target_meta.get("calibrated_feature_corr")),
                "calibration_slope": _optional_float(task.get("scale")),
                "calibration_intercept": _optional_float(task.get("bias")),
                "patch_active_frac": _optional_float(target_meta.get("patch_active_frac")),
                "target_baseline_fid": baseline_fid if baseline_fid is not None else "",
                "fid": fid if fid is not None else "",
                "delta_fid": delta_fid,
                "hook_active": int(hook.get("active", 0) or 0),
                "hook_skipped": int(hook.get("skipped", 0) or 0),
                "hook_no_t": int(hook.get("no_t", 0) or 0),
                "n_samples": _optional_int(result.get("n_samples") if result else None) or "",
                "cfg_scale": _optional_float(result.get("cfg_scale") if result else None),
                "sampler": _clean_optional(result.get("sampler") if result else ""),
                "sample_steps": _optional_int(result.get("sample_steps") if result else None) or "",
                "seed": _optional_int(result.get("seed") if result else None) or "",
                "no_normalize": _optional_bool(result.get("no_normalize") if result else None),
                "raw_run_tag": _clean_optional(result.get("run_tag") if result else ""),
                "reused_native_control": reused_native,
                "coverage_status": "ok" if result is not None else "missing",
                "gen_minutes": _optional_float(result.get("gen_minutes") if result else None),
                "fid_minutes": _optional_float(result.get("fid_minutes") if result else None),
            }
        )
    return rows


def _mode_value(mode_rows: dict[str, dict], mode: str, field: str) -> float:
    row = mode_rows.get(mode, {})
    value = _optional_float(row.get(field))
    return float(value) if value is not None else float("nan")


def _summarize_sampling_candidates(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["mode"] != "baseline":
            grouped[str(row["candidate_id"])].append(row)

    summary_rows = []
    expected_modes = {
        "transfer_replace",
        "native_ablate",
        "native_clamp",
        "random_matched_control",
        "wrong_window_control",
    }
    for candidate_id, vals in sorted(grouped.items()):
        mode_rows = {str(row["mode"]): row for row in vals}
        first = vals[0]
        transfer = mode_rows.get("transfer_replace", {})
        transfer_delta = _mode_value(mode_rows, "transfer_replace", "delta_fid")
        random_delta = _mode_value(mode_rows, "random_matched_control", "delta_fid")
        wrong_delta = _mode_value(mode_rows, "wrong_window_control", "delta_fid")
        ablate_delta = _mode_value(mode_rows, "native_ablate", "delta_fid")
        clamp_delta = _mode_value(mode_rows, "native_clamp", "delta_fid")
        transfer_minus_random = transfer_delta - random_delta
        transfer_minus_wrong = transfer_delta - wrong_delta
        status = "missing_transfer"
        if "transfer_replace" in mode_rows:
            active = int(transfer.get("hook_active", 0) or 0)
            if active <= 0:
                status = "transfer_hook_inactive"
            elif transfer_minus_random < 0 and transfer_minus_wrong < 0:
                status = "transfer_lower_fid_than_random_and_wrong_window"
            elif transfer_minus_random < 0:
                status = "transfer_lower_fid_than_random_only"
            elif transfer_minus_wrong < 0:
                status = "transfer_lower_fid_than_wrong_window_only"
            else:
                status = "no_transfer_specific_fid_advantage"
        present_modes = sorted(mode_rows)
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "selection_role": first.get("selection_role", ""),
                "source": first.get("source", ""),
                "target": first.get("target", ""),
                "layer": first.get("layer", ""),
                "t_bin": first.get("t_bin", ""),
                "source_feature_id": first.get("source_feature_id", ""),
                "target_feature_id": first.get("target_feature_id", ""),
                "source_top_label": first.get("source_top_label", ""),
                "target_top_label": first.get("target_top_label", ""),
                "match_score": first.get("match_score", ""),
                "specificity_delta_ev": first.get("specificity_delta_ev", ""),
                "calibrated_feature_corr": first.get("calibrated_feature_corr", ""),
                "calibration_slope": first.get("calibration_slope", ""),
                "patch_active_frac": first.get("patch_active_frac", ""),
                "target_baseline_fid": first.get("target_baseline_fid", ""),
                "transfer_fid": _mode_value(mode_rows, "transfer_replace", "fid"),
                "transfer_delta_fid": transfer_delta,
                "native_ablate_delta_fid": ablate_delta,
                "native_clamp_delta_fid": clamp_delta,
                "random_control_delta_fid": random_delta,
                "wrong_window_delta_fid": wrong_delta,
                "transfer_minus_random_control_delta_fid": transfer_minus_random,
                "transfer_minus_wrong_window_delta_fid": transfer_minus_wrong,
                "wrong_window_minus_transfer_delta_fid": wrong_delta - transfer_delta,
                "best_native_delta_fid": float(np.nanmin([ablate_delta, clamp_delta])),
                "transfer_hook_active": int(transfer.get("hook_active", 0) or 0),
                "transfer_hook_skipped": int(transfer.get("hook_skipped", 0) or 0),
                "transfer_hook_no_t": int(transfer.get("hook_no_t", 0) or 0),
                "n_modes_present": len(present_modes),
                "present_modes": present_modes,
                "complete_expected_modes": expected_modes.issubset(set(present_modes)),
                "screen_status": status,
            }
        )
    return summary_rows


def _plot_sampling_delta_by_candidate(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    modes = [
        ("transfer_delta_fid", "transfer", PB["blue"]),
        ("random_control_delta_fid", "random", PB["gold"]),
        ("wrong_window_delta_fid", "wrong-window", PB["red"]),
        ("native_ablate_delta_fid", "native ablate", "#8a9868"),
        ("native_clamp_delta_fid", "native clamp", "#7890a0"),
    ]
    labels = [
        f"{row['source']}->{row['target']}\nL{row['layer']}/T{row['t_bin']} F{row['source_feature_id']}->{row['target_feature_id']}"
        for row in rows
    ]
    x = np.arange(len(rows), dtype=np.float32)
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(10.0, 0.78 * len(rows)), 5.2))
    ax.axhline(0.0, color=PB["dark"], linewidth=1.0)
    for i, (field, label, color) in enumerate(modes):
        vals = [float(row.get(field, float("nan"))) for row in rows]
        ax.bar(x + (i - 2) * width, vals, width=width, label=label, color=color)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("FID - target baseline (lower is better)")
    ax.set_title("Sampling-time feature patching screen")
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_sampling_transfer_vs_controls(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    labels = [
        f"{row['source']}->{row['target']} L{row['layer']}/T{row['t_bin']}\n{row['target_top_label']}"
        for row in rows
    ]
    x = np.arange(len(rows), dtype=np.float32)
    width = 0.34
    random_vals = [float(row["transfer_minus_random_control_delta_fid"]) for row in rows]
    wrong_vals = [float(row["transfer_minus_wrong_window_delta_fid"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(9.5, 0.72 * len(rows)), 4.8))
    ax.axhline(0.0, color=PB["dark"], linewidth=1.0)
    ax.bar(x - width / 2, random_vals, width=width, color=PB["blue"], label="transfer - random")
    ax.bar(x + width / 2, wrong_vals, width=width, color=PB["red"], label="transfer - wrong-window")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Delta FID difference (negative favors transfer)")
    ax.set_title("Transfer specificity against sampling controls")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _sampling_summary_payload(
    rows: list[dict],
    candidate_rows: list[dict],
    *,
    sampling_csv: Path,
    task_tsv: Path,
    target_csv: Path | None,
) -> dict:
    baselines = {
        str(row["target"]): float(row["fid"])
        for row in rows
        if row["mode"] == "baseline" and _optional_float(row.get("fid")) is not None
    }
    n_samples_values = [
        int(row["n_samples"])
        for row in rows
        if _optional_int(row.get("n_samples")) is not None
    ]
    n_samples = max(n_samples_values) if n_samples_values else 0
    run_stage = "final" if n_samples >= 5000 else "screen"
    transfer_rows = [row for row in candidate_rows if not math.isnan(float(row["transfer_delta_fid"]))]
    recommended = sorted(
        [
            row
            for row in transfer_rows
            if int(row["transfer_hook_active"]) > 0
            and float(row["transfer_minus_random_control_delta_fid"]) < 0
        ],
        key=lambda row: (
            float(row["transfer_minus_random_control_delta_fid"]),
            float(row["transfer_minus_wrong_window_delta_fid"]),
        ),
    )[:6]
    if not recommended:
        recommended = sorted(
            [row for row in transfer_rows if int(row["transfer_hook_active"]) > 0],
            key=lambda row: float(row["transfer_minus_random_control_delta_fid"]),
        )[:6]
    return {
        "analysis": f"sampling-time-feature-patching-{run_stage}",
        "run_stage": run_stage,
        "n_samples": n_samples,
        "sampling_csv": str(sampling_csv),
        "task_tsv": str(task_tsv),
        "target_csv": "" if target_csv is None else str(target_csv),
        "n_task_rows": len(rows),
        "n_candidate_rows": len(candidate_rows),
        "n_missing_logical_tasks": sum(1 for row in rows if row["coverage_status"] != "ok"),
        "n_reused_native_control_rows": sum(1 for row in rows if bool(row["reused_native_control"])),
        "baseline_fid": baselines,
        "n_transfer_candidates": len(transfer_rows),
        "n_transfer_hook_inactive": sum(1 for row in candidate_rows if int(row["transfer_hook_active"]) <= 0),
        "n_transfer_lower_fid_than_baseline": sum(
            1 for row in candidate_rows if float(row["transfer_delta_fid"]) < 0
        ),
        "n_transfer_lower_fid_than_random": sum(
            1 for row in candidate_rows if float(row["transfer_minus_random_control_delta_fid"]) < 0
        ),
        "n_transfer_lower_fid_than_wrong_window": sum(
            1 for row in candidate_rows if float(row["transfer_minus_wrong_window_delta_fid"]) < 0
        ),
        "n_transfer_lower_fid_than_random_and_wrong_window": sum(
            1
            for row in candidate_rows
            if float(row["transfer_minus_random_control_delta_fid"]) < 0
            and float(row["transfer_minus_wrong_window_delta_fid"]) < 0
        ),
        "screen_status_counts": dict(Counter(str(row["screen_status"]) for row in candidate_rows)),
        "selection_role_counts": dict(Counter(str(row["selection_role"]) for row in candidate_rows)),
        "recommended_final_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "source": row["source"],
                "target": row["target"],
                "layer": row["layer"],
                "t_bin": row["t_bin"],
                "source_feature_id": row["source_feature_id"],
                "target_feature_id": row["target_feature_id"],
                "target_top_label": row["target_top_label"],
                "transfer_delta_fid": row["transfer_delta_fid"],
                "transfer_minus_random_control_delta_fid": row["transfer_minus_random_control_delta_fid"],
                "transfer_minus_wrong_window_delta_fid": row["transfer_minus_wrong_window_delta_fid"],
                "transfer_hook_active": row["transfer_hook_active"],
                "screen_status": row["screen_status"],
            }
            for row in recommended
        ],
    }


def _write_sampling_summary_md(out_dir: Path, payload: dict, candidate_rows: list[dict]) -> None:
    candidates = sorted(
        candidate_rows,
        key=lambda row: float(row["transfer_minus_random_control_delta_fid"]),
    )
    run_stage = str(payload.get("run_stage", "screen"))
    n_samples = int(payload.get("n_samples", 0) or 0)
    title = "Sampling-Time Feature Patching Final" if run_stage == "final" else "Sampling-Time Feature Patching Screen"
    caveat = (
        "This is the final N=5000 sampling validation for selected candidates."
        if run_stage == "final"
        else "This is a screening result for selecting final N=5000 interventions, not a final FID claim."
    )
    reuse_line = (
        f"- {payload['n_reused_native_control_rows']} native control rows were reused because native target-feature interventions are independent of the source tokenizer."
        if int(payload["n_reused_native_control_rows"]) > 0
        else "- No native-control rows were reused."
    )
    lines = [
        f"# {title}",
        "",
        "> [!summary] TL;DR",
        (
            f"> **N={n_samples} sampling-time feature patching** covered "
            f"=={payload['n_task_rows']} logical task slots== with "
            f"{payload['n_reused_native_control_rows']} reused native controls."
        ),
        (
            f"> Transfer hooks were inactive for "
            f"=={payload['n_transfer_hook_inactive']}/{payload['n_transfer_candidates']}== candidates; "
            f"{payload['n_transfer_lower_fid_than_random_and_wrong_window']} beat both random and wrong-window controls."
        ),
        f"> {caveat}",
        "",
        "## Method",
        "",
        f"- Same-noise ImageNet-class sampling was run with `N={n_samples}`, CFG `1.5`, ODE dopri5, 250 sampler steps, seed `0`.",
        "- Modes were `baseline`, `transfer_replace`, `native_ablate`, `native_clamp`, `random_matched_control`, and `wrong_window_control`.",
        "- `eq_vae` sampling used the non-normalized/noz convention.",
        reuse_line,
        "",
        "## Baselines",
        "",
    ]
    for target, fid in sorted(payload["baseline_fid"].items()):
        lines.append(f"- `{target}`: FID `{fid:.3f}`")
    lines.extend(["", "## Candidate Ranking", ""])
    for row in candidates:
        lines.append(
            "- "
            f"`{row['candidate_id']}`: transfer delta FID `{float(row['transfer_delta_fid']):+.3f}`, "
            f"transfer-random `{float(row['transfer_minus_random_control_delta_fid']):+.3f}`, "
            f"transfer-wrong-window `{float(row['transfer_minus_wrong_window_delta_fid']):+.3f}`, "
            f"hook active `{row['transfer_hook_active']}`, status `{row['screen_status']}`."
        )
    lines.extend(["", "## Recommended Final Candidates", ""])
    for row in payload["recommended_final_candidates"]:
        lines.append(
            "- "
            f"`{row['candidate_id']}`: `{row['source']}->{row['target']}` "
            f"L{row['layer']}/T{row['t_bin']} F{row['source_feature_id']}->{row['target_feature_id']} "
            f"({row['target_top_label']}), transfer-random "
            f"`{float(row['transfer_minus_random_control_delta_fid']):+.3f}`."
        )
    lines.append("")
    summary_name = "sampling_final_summary.md" if run_stage == "final" else "sampling_screen_summary.md"
    (out_dir / summary_name).write_text("\n".join(lines), encoding="utf-8")


def run_summarize_sampling(args: argparse.Namespace) -> int:
    out_dir = args.out_dir or args.sampling_csv.parent
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    result_rows = read_csv(args.sampling_csv)
    task_rows = read_tsv(args.task_tsv)
    target_rows = read_csv(args.target_csv) if args.target_csv is not None else []
    analysis_rows = _expand_sampling_rows(result_rows, task_rows, target_rows)
    candidate_rows = _summarize_sampling_candidates(analysis_rows)
    write_csv(out_dir / "metrics" / "sampling_feature_patching_analysis.csv", analysis_rows)
    write_csv(out_dir / "metrics" / "sampling_candidate_summary.csv", candidate_rows)
    _plot_sampling_delta_by_candidate(candidate_rows, out_dir / "plots" / "sampling_delta_fid_by_candidate.png")
    _plot_sampling_transfer_vs_controls(candidate_rows, out_dir / "plots" / "sampling_transfer_vs_controls.png")
    payload = _sampling_summary_payload(
        analysis_rows,
        candidate_rows,
        sampling_csv=args.sampling_csv,
        task_tsv=args.task_tsv,
        target_csv=args.target_csv,
    )
    (out_dir / "metrics" / "sampling_feature_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_sampling_summary_md(out_dir, payload, candidate_rows)
    ok(f"sampling summary complete: {out_dir}")
    return 0


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        lroot = self.find(left)
        rroot = self.find(right)
        if lroot != rroot:
            self.parent[rroot] = lroot


def _feature_key(condition: str, layer: int, t_bin: int, feature_id: int) -> str:
    return f"{condition}:L{layer}:T{t_bin}:F{feature_id}"


def run_group_features(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id, resume=args.resume)
    rows = read_csv(args.match_csv)
    uf = _UnionFind()
    metadata: dict[str, dict] = {}
    for row in rows:
        score = float(row.get("match_score", 0.0))
        activation = float(row.get("activation_corr", float("nan")))
        class_jaccard = float(row.get("top_class_jaccard", row.get("top9_class_jaccard", 0.0)))
        if score < args.min_group_score and not (
            class_jaccard >= args.min_group_class_jaccard
            and np.isfinite(activation)
            and activation >= args.min_group_activation_corr
        ):
            continue
        layer = int(row["layer"])
        t_bin = int(row["t_bin"])
        left = _feature_key(str(row["source"]), layer, t_bin, int(row["source_feature_id"]))
        right = _feature_key(str(row["target"]), layer, t_bin, int(row["target_feature_id"]))
        uf.union(left, right)
        metadata[left] = {
            "condition": row["source"],
            "layer": layer,
            "t_bin": t_bin,
            "feature_id": int(row["source_feature_id"]),
            "top_label": row.get("source_top_label", ""),
            "top_class_idx": row.get("source_top_class_idx", ""),
        }
        metadata[right] = {
            "condition": row["target"],
            "layer": layer,
            "t_bin": t_bin,
            "feature_id": int(row["target_feature_id"]),
            "top_label": row.get("target_top_label", ""),
            "top_class_idx": row.get("target_top_class_idx", ""),
        }
    components: dict[str, list[str]] = defaultdict(list)
    for key in metadata:
        components[uf.find(key)].append(key)

    group_rows = []
    member_rows = []
    for group_idx, members in enumerate(sorted(components.values(), key=lambda vals: (-len(vals), sorted(vals)[0]))):
        conditions = sorted({str(metadata[m]["condition"]) for m in members})
        labels = [str(metadata[m].get("top_label", "")) for m in members if metadata[m].get("top_label", "")]
        label_counts = Counter(labels)
        family_label = label_counts.most_common(1)[0][0] if label_counts else ""
        layers = sorted({int(metadata[m]["layer"]) for m in members})
        t_bins = sorted({int(metadata[m]["t_bin"]) for m in members})
        group_id = f"fg_{group_idx:05d}"
        group_rows.append(
            {
                "group_id": group_id,
                "family_label": family_label,
                "n_members": len(members),
                "n_conditions": len(conditions),
                "conditions": conditions,
                "layers": layers,
                "t_bins": t_bins,
                "classification": "shared_feature_family" if len(conditions) > 1 else "single_condition_family",
            }
        )
        for member in sorted(members):
            member_rows.append({"group_id": group_id, **metadata[member]})

    write_csv(out_dir / "metrics" / "feature_groups.csv", group_rows)
    write_csv(out_dir / "metrics" / "feature_group_members.csv", member_rows)
    summary = {
        "analysis": "feature-family-grouping",
        "match_csv": str(args.match_csv),
        "n_groups": len(group_rows),
        "n_members": len(member_rows),
        "n_cross_condition_groups": sum(1 for row in group_rows if int(row["n_conditions"]) > 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_md(
        out_dir,
        "Feature Families",
        [
            f"Grouped matched features into =={len(group_rows)}== candidate families.",
            "Groups are connected components over high-scoring cross-tokenizer feature matches.",
            "A group is a hypothesis for shared concept family, not yet proof of same individual feature.",
        ],
    )
    ok(f"feature family grouping complete: {out_dir}")
    return 0


def _write_summary_md(out_dir: Path, title: str, lines: list[str]) -> None:
    payload = [f"# {title}", "", "> [!summary] TL;DR"]
    payload.extend(f"> {line}" for line in lines[:4])
    payload.extend(["", *lines, ""])
    (out_dir / "summary.md").write_text("\n".join(payload), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    bank = sub.add_parser("build-bank")
    _add_common_args(bank)
    bank.add_argument("--dashboard_root", type=Path, default=DEFAULT_DASHBOARD_ROOT)
    bank.add_argument("--atlas_csv", type=Path, default=None)
    bank.add_argument("--hydrate_dashboard_json", action="store_true")
    bank.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    bank.add_argument("--with_decoder_norms", action="store_true")
    bank.add_argument("--device", type=str, default=None)
    bank.add_argument("--monosemantic_only", action="store_true", default=True)
    bank.add_argument("--include_all_live", dest="monosemantic_only", action="store_false")
    bank.add_argument("--min_density", type=float, default=1e-4)
    bank.add_argument("--max_density", type=float, default=0.10)
    bank.add_argument("--max_entropy", type=float, default=2.5)
    bank.add_argument("--max_features_per_cell", type=int, default=None)
    bank.set_defaults(func=run_build_bank)

    match = sub.add_parser("match")
    _add_common_args(match)
    match.add_argument("--feature_bank", type=Path, required=True)
    match.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    match.add_argument("--metadata_only", action="store_true")
    match.add_argument("--device", type=str, default=None)
    match.add_argument("--profile_batch_images", type=int, default=2)
    match.add_argument("--max_fit_tokens", type=int, default=8192)
    match.add_argument("--ridge_alpha", type=float, default=10.0)
    match.add_argument("--cache_profiles", action="store_true")
    match.add_argument("--same_top_class_only", action="store_true", default=True)
    match.add_argument("--allow_cross_class", dest="same_top_class_only", action="store_false")
    match.add_argument("--min_candidate_score", type=float, default=0.25)
    match.add_argument("--min_hungarian_score", type=float, default=0.35)
    match.add_argument("--shared_score_threshold", type=float, default=0.65)
    match.add_argument("--shared_activation_corr_threshold", type=float, default=0.40)
    match.add_argument("--shared_class_jaccard_threshold", type=float, default=0.30)
    match.set_defaults(func=run_match)

    patch = sub.add_parser("activation-patch")
    _add_common_args(patch)
    patch.add_argument("--match_csv", type=Path, required=True)
    patch.add_argument("--feature_bank", type=Path, default=None)
    patch.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    patch.add_argument("--max_fit_tokens", type=int, default=50_000)
    patch.add_argument("--max_eval_tokens", type=int, default=8192)
    patch.add_argument("--ridge_alpha", type=float, default=10.0)
    patch.add_argument("--sae_batch_size", type=int, default=512)
    patch.add_argument("--device", type=str, default=None)
    patch.add_argument("--max_pairs", type=int, default=None)
    patch.add_argument("--max_pairs_per_group", type=int, default=16)
    patch.add_argument("--directed_pairs", nargs="+", default=None, help="Optional filters like sd_vae->eq_vae.")
    patch.add_argument("--active_quantile", type=float, default=0.95)
    patch.add_argument("--clamp_quantile", type=float, default=0.99)
    patch.add_argument("--min_active_tokens", type=int, default=32)
    patch.set_defaults(func=run_activation_patch)

    group = sub.add_parser("group-features")
    _add_common_args(group)
    group.add_argument("--match_csv", type=Path, required=True)
    group.add_argument("--min_group_score", type=float, default=0.65)
    group.add_argument("--min_group_activation_corr", type=float, default=0.40)
    group.add_argument("--min_group_class_jaccard", type=float, default=0.30)
    group.set_defaults(func=run_group_features)

    group_patch = sub.add_parser("group-activation-patch")
    _add_common_args(group_patch)
    group_patch.add_argument("--group_members_csv", type=Path, required=True)
    group_patch.add_argument("--feature_bank", type=Path, default=None)
    group_patch.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    group_patch.add_argument("--max_fit_tokens", type=int, default=50_000)
    group_patch.add_argument("--max_eval_tokens", type=int, default=8192)
    group_patch.add_argument("--ridge_alpha", type=float, default=10.0)
    group_patch.add_argument("--group_ridge_alpha", type=float, default=10.0)
    group_patch.add_argument("--sae_batch_size", type=int, default=512)
    group_patch.add_argument("--device", type=str, default=None)
    group_patch.add_argument("--group_ids", nargs="+", default=None)
    group_patch.add_argument("--directed_pairs", nargs="+", default=None, help="Optional filters like sd_vae->eq_vae.")
    group_patch.add_argument("--max_groups", type=int, default=None)
    group_patch.add_argument("--max_source_features", type=int, default=8)
    group_patch.add_argument("--max_target_features", type=int, default=8)
    group_patch.add_argument("--min_group_conditions", type=int, default=2)
    group_patch.add_argument("--active_quantile", type=float, default=0.95)
    group_patch.add_argument("--clamp_quantile", type=float, default=0.99)
    group_patch.add_argument("--min_active_tokens", type=int, default=32)
    group_patch.set_defaults(func=run_group_activation_patch)

    sampling = sub.add_parser("summarize-sampling")
    sampling.add_argument("--sampling_csv", type=Path, required=True)
    sampling.add_argument("--task_tsv", type=Path, required=True)
    sampling.add_argument("--target_csv", type=Path, default=None)
    sampling.add_argument("--out_dir", type=Path, default=None)
    sampling.set_defaults(func=run_summarize_sampling)
    return p


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    p.add_argument("--t_bins", nargs="+", type=int, default=list(T_BINS))
    p.add_argument("--cells", nargs="+", default=None)
    p.add_argument("--dit_step", type=int, default=200_000)
    p.add_argument("--activations_root", type=Path, default=ACTIVATIONS_YNULL)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_images", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "t_bins") and any(t not in T_CENTERS for t in args.t_bins):
        parser.error(f"--t_bins must be drawn from {sorted(T_CENTERS)}")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
