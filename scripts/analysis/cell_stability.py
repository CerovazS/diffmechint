"""Cell-level cross-seed SAE stability probes and subspace metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from diffmechint.probing.concepts import CONCEPTS, get_concept
from diffmechint.utils import info

CELL_RE = re.compile(r"^(?P<condition>[a-z_0-9]+)_L(?P<layer>\d+)_T(?P<t_bin>\d+)$")
LAYERS = (3, 6, 9)
T_BINS = (0, 1, 2)
PB = {
    "ink": "#1c1b19",
    "teal": "#335C67",
    "amber": "#E09F3E",
    "red": "#9E2A2B",
    "cream": "#FFF3B0",
    "wine": "#540B0E",
}
DEFAULT_SAE0 = Path("/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k")
DEFAULT_SAE1 = Path("/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k_seed3407")
DEFAULT_ACTS = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_ynull")
DEFAULT_LATENTS = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents_val")
DEFAULT_CONCEPTS = ("animal_binary", "broad_8", "vehicle_binary", "food_binary", "instrument_binary")


def _parse_cell_name(name: str) -> tuple[str, int, int] | None:
    m = CELL_RE.match(name)
    if m is None:
        return None
    return m["condition"], int(m["layer"]), int(m["t_bin"])


def _cell_name(condition: str, layer: int, t_bin: int) -> str:
    return f"{condition}_L{layer}_T{t_bin}"


def _resolve_sae_ckpt(sae_root: Path, condition: str, layer: int, t_bin: int, dit_step: int) -> Path:
    base = sae_root / condition / f"L{layer}_T{t_bin}" / f"step_{dit_step:06d}"
    finals = []
    last_exc: OSError | None = None
    for attempt in range(5):
        try:
            finals = sorted(
                p for p in base.glob("final_*")
                if (p / "sae_weights.safetensors").exists()
            )
            break
        except OSError as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise FileNotFoundError(f"could not inspect SAE checkpoint dir {base}: {last_exc}") from last_exc
    if not finals:
        raise FileNotFoundError(f"missing final SAE checkpoint under {base}")
    return finals[-1]


def _available_cells(sae0_root: Path, sae1_root: Path, activations_root: Path, dit_step: int) -> list[tuple[str, int, int]]:
    cells = []
    for condition in ("sd_vae", "repa_e", "eq_vae"):
        for layer in LAYERS:
            for t_bin in T_BINS:
                shard = activations_root / condition / f"step_{dit_step:06d}" / f"{layer}_{t_bin}.h5"
                try:
                    _resolve_sae_ckpt(sae0_root, condition, layer, t_bin, dit_step)
                    _resolve_sae_ckpt(sae1_root, condition, layer, t_bin, dit_step)
                except FileNotFoundError:
                    continue
                if shard.exists():
                    cells.append((condition, layer, t_bin))
    return cells


def _load_weight_tensors(ckpt_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    last_exc: OSError | None = None
    for attempt in range(5):
        try:
            state = load_file(str(ckpt_dir / "sae_weights.safetensors"), device=str(device))
            break
        except OSError as exc:
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    else:
        raise OSError(f"failed to load SAE weights after retries from {ckpt_dir}: {last_exc}") from last_exc
    return {
        "W_enc": state["W_enc"].float(),
        "W_dec": state["W_dec"].float(),
        "b_enc": state["b_enc"].float(),
        "b_dec": state["b_dec"].float(),
        "W_dec_norm": state["W_dec"].float().norm(dim=-1).clamp_min(1e-12),
    }


def _direct_encode(x: torch.Tensor, weights: dict[str, torch.Tensor], *, k: int) -> torch.Tensor:
    hidden = (x.float() - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"]
    hidden = hidden * weights["W_dec_norm"]
    values, indices = torch.topk(hidden, k=k, dim=-1, sorted=False)
    out = torch.zeros_like(hidden)
    out.scatter_(-1, indices, values.relu())
    return out


def _direct_decode(z: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return (z / weights["W_dec_norm"]) @ weights["W_dec"] + weights["b_dec"]


def _resolve_labels(activation_shard_dir: Path, latents_root: Path, condition: str) -> np.ndarray:
    manifest = json.loads((activation_shard_dir / "manifest.json").read_text())
    sample_idx = np.asarray(manifest["sample_idx"], dtype=np.int64)

    labels = []
    for shard in sorted((latents_root / condition).glob("*.h5")):
        with h5py.File(shard, "r") as f:
            labels.append(np.asarray(f["labels"][()], dtype=np.int64))
    if not labels:
        raise FileNotFoundError(f"no latent label shards under {latents_root / condition}")
    all_labels = np.concatenate(labels)
    if int(sample_idx.max()) >= len(all_labels):
        raise IndexError("activation manifest sample_idx exceeds available latent labels")
    return all_labels[sample_idx]


def _class_balanced_indices(labels: np.ndarray, max_images: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    by_class: dict[int, np.ndarray] = {}
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        by_class[int(cls)] = idx

    selected: list[int] = []
    quota = max(1, max_images // max(len(by_class), 1))
    for cls in sorted(by_class):
        selected.extend(by_class[cls][:quota].tolist())
    if len(selected) < max_images:
        used = set(selected)
        rest = np.asarray([i for i in range(len(labels)) if i not in used], dtype=np.int64)
        rng.shuffle(rest)
        selected.extend(rest[: max_images - len(selected)].tolist())
    return np.asarray(sorted(selected[:max_images]), dtype=np.int64)


def _remap_labels(labels: np.ndarray, concept_name: str) -> np.ndarray:
    concept = get_concept(concept_name)
    mapping = {int(v): int(concept.label_fn(int(v))) for v in np.unique(labels).tolist()}
    out = np.empty_like(labels, dtype=np.int64)
    for raw, mapped in mapping.items():
        out[labels == raw] = mapped
    return out


@torch.no_grad()
def _extract_image_features(
    weights: dict[str, torch.Tensor],
    shard_path: Path,
    image_indices: np.ndarray,
    *,
    batch_images: int,
    device: torch.device,
    pool: str,
    k: int,
) -> np.ndarray:
    d_sae = int(weights["W_dec"].shape[0])
    rows = np.zeros((len(image_indices), d_sae), dtype=np.float16)
    with h5py.File(shard_path, "r") as f:
        acts = f["activations"]
        t = int(acts.shape[1])
        d = int(acts.shape[2])
        for out_lo in range(0, len(image_indices), batch_images):
            out_hi = min(out_lo + batch_images, len(image_indices))
            idx = image_indices[out_lo:out_hi]
            chunk = acts[idx]
            x = torch.from_numpy(chunk).to(device=device, dtype=torch.float32, non_blocking=True)
            z = _direct_encode(x.reshape(-1, d), weights, k=k).reshape(len(idx), t, d_sae)
            if pool == "max":
                pooled = z.amax(dim=1)
            elif pool == "mean":
                pooled = z.mean(dim=1)
            else:
                raise ValueError(f"unknown feature pool {pool!r}")
            rows[out_lo:out_hi] = pooled.to(torch.float16).cpu().numpy()
            if out_lo == 0 or out_hi == len(image_indices):
                info(f"    encoded images {out_hi}/{len(image_indices)}")
    return rows


def _train_probe(features: np.ndarray, labels: np.ndarray, *, seed: int, max_iter: int) -> tuple[float, int, int, int]:
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = int(np.unique(labels).size)
    if n_classes < 2:
        return math.nan, len(labels), 0, n_classes
    counts = Counter(labels.tolist())
    stratify = labels if min(counts.values()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features.astype(np.float32, copy=False),
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )
    scaler = StandardScaler(with_mean=False)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=max_iter,
        tol=1e-3,
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return float(accuracy_score(y_test, pred)), len(y_train), len(y_test), n_classes


def _select_probe_features(features: np.ndarray, max_probe_features: int | None) -> tuple[np.ndarray, int]:
    if max_probe_features is None or max_probe_features <= 0 or features.shape[1] <= max_probe_features:
        return features, int(features.shape[1])
    variances = features.astype(np.float32, copy=False).var(axis=0)
    cols = np.argpartition(variances, -max_probe_features)[-max_probe_features:]
    cols = np.sort(cols)
    return features[:, cols], len(cols)


def _plot_probe_summary(paired: pd.DataFrame, out_dir: Path) -> None:
    if paired.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    grouped = paired.groupby("concept")["delta_accuracy"].mean().sort_values()
    colors = [PB["teal"] if v >= 0 else PB["red"] for v in grouped.to_numpy()]
    ax.bar(np.arange(len(grouped)), grouped.to_numpy(), color=colors, edgecolor=PB["ink"], linewidth=0.5)
    ax.axhline(0, color=PB["ink"], linewidth=0.8)
    ax.set_xticks(np.arange(len(grouped)), grouped.index, rotation=30, ha="right")
    ax.set_ylabel("seed3407 - seed0 accuracy")
    ax.set_title("Cell-level pooled SAE-feature probe deltas")
    fig.tight_layout()
    fig.savefig(out_dir / "pooled_probe_delta_by_concept.png", dpi=160)
    plt.close(fig)


def _finalize_probe_rows(rows: list[dict], out_dir: Path, args: argparse.Namespace, cells: list[tuple[str, int, int]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pooled_probe_scores.csv", index=False)
    keys = ["cell", "condition", "layer", "t_bin", "dit_step", "concept"]
    paired = df[df["seed_label"] == args.seed0_label].merge(
        df[df["seed_label"] == args.seed1_label],
        on=keys,
        suffixes=("_seed0", "_seed1"),
    )
    if not paired.empty:
        paired["delta_accuracy"] = paired["accuracy_seed1"] - paired["accuracy_seed0"]
    paired.to_csv(out_dir / "pooled_probe_paired.csv", index=False)
    correlations = {}
    for concept_name in args.concepts:
        sub = paired[paired["concept"] == concept_name][["accuracy_seed0", "accuracy_seed1"]].dropna()
        correlations[concept_name] = float(sub["accuracy_seed0"].corr(sub["accuracy_seed1"])) if len(sub) >= 2 else math.nan
    summary = {
        "n_cells": len(cells),
        "cells": [_cell_name(*c) for c in cells],
        "concepts": list(args.concepts),
        "pool": args.pool,
        "max_images": args.max_images,
        "max_probe_features": args.max_probe_features,
        "mean_delta_accuracy": float(paired["delta_accuracy"].mean()) if not paired.empty else math.nan,
        "mean_abs_delta_accuracy": float(paired["delta_accuracy"].abs().mean()) if not paired.empty else math.nan,
        "correlations_by_concept": correlations,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot_probe_summary(paired, out_dir)
    print(json.dumps(summary, indent=2))


def cmd_pooled_probes(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    cells = _available_cells(args.seed0_sae_root, args.seed1_sae_root, args.activations_root, args.dit_step)
    if args.cells:
        wanted = set(args.cells)
        cells = [c for c in cells if _cell_name(*c) in wanted]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition, layer, t_bin in cells:
        cell = _cell_name(condition, layer, t_bin)
        shard_dir = args.activations_root / condition / f"step_{args.dit_step:06d}"
        shard_path = shard_dir / f"{layer}_{t_bin}.h5"
        labels_all = _resolve_labels(shard_dir, args.latents_root, condition)
        indices = _class_balanced_indices(labels_all, args.max_images, args.sample_seed)
        labels_raw = labels_all[indices]
        info(f"{cell}: selected {len(indices)} class-balanced images")
        for seed_label, sae_root in ((args.seed0_label, args.seed0_sae_root), (args.seed1_label, args.seed1_sae_root)):
            ckpt = _resolve_sae_ckpt(sae_root, condition, layer, t_bin, args.dit_step)
            info(f"  {seed_label}: loading {ckpt}")
            weights = _load_weight_tensors(ckpt, device)
            features = _extract_image_features(
                weights,
                shard_path,
                indices,
                batch_images=args.batch_images,
                device=device,
                pool=args.pool,
                k=args.k,
            )
            del weights
            if device.type == "cuda":
                torch.cuda.empty_cache()
            features, n_probe_features = _select_probe_features(features, args.max_probe_features)
            info(f"    probe feature columns: {n_probe_features}")
            for concept_name in args.concepts:
                labels = _remap_labels(labels_raw, concept_name)
                acc, n_train, n_test, n_classes = _train_probe(
                    features,
                    labels,
                    seed=args.probe_seed,
                    max_iter=args.max_iter,
                )
                chance = 1.0 / max(n_classes, 1)
                rows.append({
                    "cell": cell,
                    "condition": condition,
                    "layer": layer,
                    "t_bin": t_bin,
                    "dit_step": args.dit_step,
                    "seed_label": seed_label,
                    "concept": concept_name,
                    "accuracy": acc,
                    "chance_accuracy": chance,
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_classes": n_classes,
                    "n_images": len(indices),
                    "n_probe_features": n_probe_features,
                    "feature_pool": args.pool,
                    "sae_ckpt": str(ckpt),
                })
                info(f"    {concept_name}: acc={acc:.4f} chance={chance:.4f}")
            _finalize_probe_rows(rows, args.out_dir, args, cells)
            del features
    _finalize_probe_rows(rows, args.out_dir, args, cells)
    return 0


def _decoder_cov_metrics(ckpt0: Path, ckpt1: Path, top_dims: int) -> dict[str, float]:
    w0 = load_file(str(ckpt0 / "sae_weights.safetensors"), device="cpu")["W_dec"].float()
    w1 = load_file(str(ckpt1 / "sae_weights.safetensors"), device="cpu")["W_dec"].float()
    c0 = w0.T @ w0
    c1 = w1.T @ w1
    cov_cos = float(torch.sum(c0 * c1) / (torch.linalg.norm(c0) * torch.linalg.norm(c1)).clamp_min(1e-12))
    eval0, evec0 = torch.linalg.eigh(c0)
    eval1, evec1 = torch.linalg.eigh(c1)
    u0 = evec0[:, torch.argsort(eval0, descending=True)[:top_dims]]
    u1 = evec1[:, torch.argsort(eval1, descending=True)[:top_dims]]
    s = torch.linalg.svdvals(u0.T @ u1)
    return {
        "decoder_cov_cosine": cov_cos,
        "decoder_top_subspace_mean_cos": float(s.mean()),
        "decoder_top_subspace_min_cos": float(s.min()),
        "decoder_top_subspace_projection_similarity": float(s.pow(2).sum() / top_dims),
    }


def _sample_tokens(shard_path: Path, image_indices: np.ndarray, max_tokens: int, seed: int) -> torch.Tensor:
    with h5py.File(shard_path, "r") as f:
        acts = f["activations"][image_indices]
    tokens = torch.from_numpy(acts.reshape(-1, acts.shape[-1]).astype(np.float32))
    if tokens.shape[0] > max_tokens:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(tokens.shape[0], size=max_tokens, replace=False))
        tokens = tokens[idx]
    return tokens


@torch.no_grad()
def _reconstruct_tokens(
    weights: dict[str, torch.Tensor],
    tokens: torch.Tensor,
    *,
    batch_tokens: int,
    device: torch.device,
    k: int,
) -> torch.Tensor:
    chunks = []
    for lo in range(0, tokens.shape[0], batch_tokens):
        hi = min(lo + batch_tokens, tokens.shape[0])
        x = tokens[lo:hi].to(device=device, dtype=torch.float32)
        chunks.append(_direct_decode(_direct_encode(x, weights, k=k), weights).float().cpu())
    return torch.cat(chunks, dim=0)


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    num = torch.linalg.norm(xy, ord="fro").pow(2)
    den = torch.linalg.norm(xx, ord="fro") * torch.linalg.norm(yy, ord="fro")
    return float(num / den.clamp_min(1e-12))


def _procrustes_similarity(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x.T @ y)
    norm_x = torch.linalg.norm(x)
    norm_y = torch.linalg.norm(y)
    similarity_t = s.sum() / (norm_x * norm_y).clamp_min(1e-12)
    similarity = float(torch.clamp(similarity_t, min=0.0, max=1.0))
    residual = float(torch.sqrt(torch.clamp(norm_x.pow(2) + norm_y.pow(2) - 2 * s.sum(), min=0.0)) / (norm_x + norm_y).clamp_min(1e-12))
    cosine = float((torch.nn.functional.normalize(x, dim=1) * torch.nn.functional.normalize(y, dim=1)).sum(dim=1).mean())
    return similarity, residual, cosine


def _plot_subspace(rows: pd.DataFrame, out_dir: Path) -> None:
    if rows.empty:
        return
    cols = ["decoder_cov_cosine", "reconstruction_cka", "reconstruction_procrustes_similarity"]
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [rows[c].dropna().mean() for c in cols]
    ax.bar(np.arange(len(cols)), vals, color=[PB["teal"], PB["amber"], PB["wine"]], edgecolor=PB["ink"], linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(cols)), ["decoder cov", "recon CKA", "recon Procrustes"], rotation=20, ha="right")
    ax.set_ylabel("similarity")
    ax.set_title("Feature-order-invariant cross-seed subspace similarity")
    fig.tight_layout()
    fig.savefig(out_dir / "subspace_similarity_means.png", dpi=160)
    plt.close(fig)


def cmd_subspace(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    cells = _available_cells(args.seed0_sae_root, args.seed1_sae_root, args.activations_root, args.dit_step)
    if args.cells:
        wanted = set(args.cells)
        cells = [c for c in cells if _cell_name(*c) in wanted]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition, layer, t_bin in cells:
        cell = _cell_name(condition, layer, t_bin)
        shard_dir = args.activations_root / condition / f"step_{args.dit_step:06d}"
        labels = _resolve_labels(shard_dir, args.latents_root, condition)
        indices = _class_balanced_indices(labels, args.max_images, args.sample_seed)
        shard_path = shard_dir / f"{layer}_{t_bin}.h5"
        ckpt0 = _resolve_sae_ckpt(args.seed0_sae_root, condition, layer, t_bin, args.dit_step)
        ckpt1 = _resolve_sae_ckpt(args.seed1_sae_root, condition, layer, t_bin, args.dit_step)
        info(f"{cell}: decoder metrics")
        metrics = _decoder_cov_metrics(ckpt0, ckpt1, args.top_dims)
        info(f"{cell}: reconstruction CKA/Procrustes on sampled y-null tokens")
        tokens = _sample_tokens(shard_path, indices, args.max_tokens, args.sample_seed)
        weights0 = _load_weight_tensors(ckpt0, device)
        weights1 = _load_weight_tensors(ckpt1, device)
        recon0 = _reconstruct_tokens(weights0, tokens, batch_tokens=args.batch_tokens, device=device, k=args.k)
        recon1 = _reconstruct_tokens(weights1, tokens, batch_tokens=args.batch_tokens, device=device, k=args.k)
        del weights0, weights1
        if device.type == "cuda":
            torch.cuda.empty_cache()
        proc_sim, proc_resid, recon_cos = _procrustes_similarity(recon0, recon1)
        metrics.update({
            "cell": cell,
            "condition": condition,
            "layer": layer,
            "t_bin": t_bin,
            "dit_step": args.dit_step,
            "n_images": len(indices),
            "n_tokens": int(tokens.shape[0]),
            "reconstruction_cka": _linear_cka(recon0, recon1),
            "reconstruction_procrustes_similarity": proc_sim,
            "reconstruction_procrustes_residual": proc_resid,
            "reconstruction_mean_cosine": recon_cos,
            "reconstruction_relative_fro_delta": float(
                torch.linalg.norm(recon1 - recon0)
                / (((torch.linalg.norm(recon0) + torch.linalg.norm(recon1)) / 2).clamp_min(1e-12))
            ),
            "seed0_ckpt": str(ckpt0),
            "seed1_ckpt": str(ckpt1),
        })
        rows.append(metrics)
        info(f"  cov={metrics['decoder_cov_cosine']:.4f} cka={metrics['reconstruction_cka']:.4f} procrustes={proc_sim:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "subspace_similarity.csv", index=False)
    summary = {
        "n_cells": len(cells),
        "cells": [_cell_name(*c) for c in cells],
        "max_images": args.max_images,
        "max_tokens": args.max_tokens,
        "top_dims": args.top_dims,
        "means": {c: float(df[c].dropna().mean()) for c in df.columns if c.endswith("similarity") or c.endswith("cosine") or c == "reconstruction_cka"},
        "mins": {c: float(df[c].dropna().min()) for c in df.columns if c.endswith("similarity") or c.endswith("cosine") or c == "reconstruction_cka"},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot_subspace(df, args.out_dir)
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed0_sae_root", type=Path, default=DEFAULT_SAE0)
    common.add_argument("--seed1_sae_root", type=Path, default=DEFAULT_SAE1)
    common.add_argument("--activations_root", type=Path, default=DEFAULT_ACTS)
    common.add_argument("--latents_root", type=Path, default=DEFAULT_LATENTS)
    common.add_argument("--dit_step", type=int, default=200_000)
    common.add_argument("--out_dir", type=Path, required=True)
    common.add_argument("--cells", nargs="*", default=None)
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--sample_seed", type=int, default=20260522)
    common.add_argument("--max_images", type=int, default=1000)

    probes = sub.add_parser("pooled-probes", parents=[common])
    probes.add_argument("--concepts", nargs="+", default=list(DEFAULT_CONCEPTS))
    probes.add_argument("--pool", choices=["max", "mean"], default="max")
    probes.add_argument("--k", type=int, default=256)
    probes.add_argument("--batch_images", type=int, default=8)
    probes.add_argument("--max_probe_features", type=int, default=4096)
    probes.add_argument("--probe_seed", type=int, default=0)
    probes.add_argument("--max_iter", type=int, default=1000)
    probes.add_argument("--seed0_label", default="seed0")
    probes.add_argument("--seed1_label", default="seed3407")
    probes.set_defaults(func=cmd_pooled_probes)

    subspace = sub.add_parser("subspace", parents=[common])
    subspace.add_argument("--top_dims", type=int, default=128)
    subspace.add_argument("--max_tokens", type=int, default=65_536)
    subspace.add_argument("--batch_tokens", type=int, default=4096)
    subspace.add_argument("--k", type=int, default=256)
    subspace.set_defaults(func=cmd_subspace)
    return p


def main() -> int:
    args = build_parser().parse_args()
    for concept_name in getattr(args, "concepts", []):
        if concept_name not in CONCEPTS:
            raise KeyError(f"unknown concept {concept_name!r}; available: {sorted(CONCEPTS)}")
        if not get_concept(concept_name).available:
            raise ValueError(f"concept {concept_name!r} is not available without external labels")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
