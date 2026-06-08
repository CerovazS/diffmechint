"""Shared constants, row schemas, and parse/decoder helpers for feature patching."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from diffmechint.sae import load_matryoshka_sae, resolve_sae_ckpt
from diffmechint.utils import warn

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


def decoder_weight(sae: torch.nn.Module) -> torch.Tensor:
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
    sae = load_matryoshka_sae(resolve_sae_ckpt(sae_root, condition, layer, t_bin, dit_step), device)
    dec = decoder_weight(sae).float()
    ids = torch.as_tensor(feature_ids, device=dec.device, dtype=torch.long)
    norms = dec.index_select(0, ids).norm(dim=1).detach().cpu().numpy()
    del sae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {int(fid): float(norm) for fid, norm in zip(feature_ids, norms, strict=True)}


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


def _sample_tokens_np(acts: np.ndarray, max_tokens: int, seed: int) -> tuple[np.ndarray, None]:
    flat = acts.reshape(-1, acts.shape[-1]).astype(np.float32)
    if flat.shape[0] <= max_tokens:
        return flat, None
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], size=max_tokens, replace=False)
    return flat[idx], None


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


def _nan_float(raw: object) -> float:
    value = _optional_float(raw)
    return float(value) if value is not None else float("nan")


def _finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _finite_median(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.median(finite)) if finite else float("nan")


def _feature_key(condition: str, layer: int, t_bin: int, feature_id: int) -> str:
    return f"{condition}:L{layer}:T{t_bin}:F{feature_id}"
