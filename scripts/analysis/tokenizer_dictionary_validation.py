"""Phase 4.10 cross-tokenizer dictionary validation analyses."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from diffmechint.probing.concepts import get_concept
from diffmechint.utils import info, model_subdir, model_variant_spec, ok, parse_layers

SCRATCH_PROJECT_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint")
ACTIVATIONS_YNULL = SCRATCH_PROJECT_ROOT / "activations_ynull"
LATENTS_VAL = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents_val")
SAE_ROOT = Path(
    "/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k"
)
OUTPUT_ROOT = Path("outputs")
DEFAULT_OUT_ROOT = Path("outputs/phase4_10_tokenizer_dictionary_validation")
DEFAULT_CONDITIONS = ("sd_vae", "repa_e", "eq_vae")
DEFAULT_CONCEPTS = (
    "object",
    "animal_binary",
    "broad_8",
    "vehicle_binary",
    "food_binary",
    "instrument_binary",
)
LAYERS = (3, 6, 9)
T_BINS = (0, 1, 2)
T_CENTERS = {0: 0.025, 1: 0.20, 2: 0.50}
PB = {
    "blue": "#335C67",
    "cream": "#FFF3B0",
    "gold": "#E09F3E",
    "red": "#9E2A2B",
    "dark": "#540B0E",
}


@dataclass(frozen=True)
class AffineMap:
    """Dense affine map with row-vector convention: `x @ weight + bias`."""

    weight: np.ndarray
    bias: np.ndarray
    kind: str

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight + self.bias


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA for rows as observations."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must have the same number of observations.")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    denom = np.linalg.norm(xtx, ord="fro") * np.linalg.norm(yty, ord="fro")
    if denom == 0:
        return float("nan")
    return float(np.linalg.norm(xty, ord="fro") ** 2 / denom)


def rsa_spearman(x: np.ndarray, y: np.ndarray, metric: str = "cosine") -> float:
    """Spearman correlation between upper triangles of pairwise distance matrices."""
    x_dist = pdist(np.asarray(x, dtype=np.float32), metric=metric)
    y_dist = pdist(np.asarray(y, dtype=np.float32), metric=metric)
    if x_dist.shape[0] == 0:
        return float("nan")
    corr = spearmanr(x_dist, y_dist).correlation
    return float(corr)


def fit_ridge_affine(x_from: np.ndarray, x_to: np.ndarray, alpha: float = 10.0) -> AffineMap:
    """Fit `x_from -> x_to` as a multi-output ridge affine map."""
    x = np.asarray(x_from, dtype=np.float64)
    y = np.asarray(x_to, dtype=np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    n_obs, n_dim = x_centered.shape
    if n_obs <= n_dim:
        gram = x_centered @ x_centered.T
        gram.flat[:: n_obs + 1] += float(alpha)
        dual = np.linalg.solve(gram, y_centered)
        weight = x_centered.T @ dual
    else:
        gram = x_centered.T @ x_centered
        gram.flat[:: n_dim + 1] += float(alpha)
        weight = np.linalg.solve(gram, x_centered.T @ y_centered)
    bias = (y_mean.reshape(-1) - x_mean.reshape(-1) @ weight).astype(np.float32)
    return AffineMap(weight=weight.astype(np.float32), bias=bias, kind="ridge")


def fit_procrustes_affine(x_from: np.ndarray, x_to: np.ndarray) -> AffineMap:
    """Fit the centred orthogonal Procrustes map `x_from -> x_to`."""
    src_mean = x_from.mean(axis=0, keepdims=True)
    tgt_mean = x_to.mean(axis=0, keepdims=True)
    rotation, _ = orthogonal_procrustes(
        (x_from - src_mean).astype(np.float64),
        (x_to - tgt_mean).astype(np.float64),
    )
    weight = rotation.astype(np.float32)
    bias = (tgt_mean.reshape(-1) - src_mean.reshape(-1) @ weight).astype(np.float32)
    return AffineMap(weight=weight, bias=bias, kind="procrustes")


def alignment_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return MSE, mean cosine, and explained variance for a residual map."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    err = target - pred
    mse = float(np.mean(err**2))
    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    denom = np.maximum(pred_norm * target_norm, 1e-12)
    cosine = float(np.mean(np.sum(pred * target, axis=1) / denom))
    var = float(np.var(target, axis=0).sum())
    ev = float(1.0 - np.var(err, axis=0).sum() / var) if var > 0 else float("nan")
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "cosine": cosine, "explained_variance": ev, "r2": r2}


def make_run_dir(out_root: Path, run_id: str | None, *, resume: bool = False) -> Path:
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / run_id
    if out_dir.exists():
        if not resume:
            raise FileExistsError(f"run directory already exists: {out_dir}")
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)
    (out_dir / "artifacts").mkdir(exist_ok=True)
    return out_dir


def _cell_path(root: Path, condition: str, dit_step: int, layer: int, t_bin: int) -> Path:
    return root / condition / f"step_{dit_step}" / f"{layer}_{t_bin}.h5"


def _manifest(root: Path, condition: str, dit_step: int) -> dict:
    path = root / condition / f"step_{dit_step}" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest missing: {path}")
    return json.loads(path.read_text())


def _resolve_labels(latents_root: Path, condition: str, sample_idx: np.ndarray) -> np.ndarray:
    shards = sorted((latents_root / condition).glob("*.h5"))
    if not shards:
        raise FileNotFoundError(f"no latent val shards under {latents_root / condition}")
    labels = []
    for shard in shards:
        with h5py.File(shard, "r") as f:
            labels.append(np.asarray(f["labels"][()], dtype=np.int64))
    all_labels = np.concatenate(labels)
    if int(sample_idx.max()) >= all_labels.shape[0]:
        raise IndexError("manifest sample_idx exceeds available latent labels")
    return all_labels[sample_idx]


def _aligned_positions(
    root: Path,
    left_condition: str,
    right_condition: str,
    dit_step: int,
    max_images: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_idx = np.asarray(_manifest(root, left_condition, dit_step)["sample_idx"], dtype=np.int64)
    right_idx = np.asarray(_manifest(root, right_condition, dit_step)["sample_idx"], dtype=np.int64)
    if np.array_equal(left_idx, right_idx):
        n = min(left_idx.shape[0], max_images)
        rng = np.random.default_rng(seed)
        pos = np.sort(rng.choice(left_idx.shape[0], size=n, replace=False))
        return pos, pos, left_idx[pos]

    right_pos = {int(v): i for i, v in enumerate(right_idx.tolist())}
    pairs = [(i, right_pos[int(v)], int(v)) for i, v in enumerate(left_idx) if int(v) in right_pos]
    if not pairs:
        raise ValueError(f"no shared sample_idx between {left_condition} and {right_condition}")
    rng = np.random.default_rng(seed)
    take = rng.choice(len(pairs), size=min(len(pairs), max_images), replace=False)
    chosen = [pairs[int(i)] for i in take]
    chosen.sort(key=lambda row: row[0])
    return (
        np.asarray([p[0] for p in chosen], dtype=np.int64),
        np.asarray([p[1] for p in chosen], dtype=np.int64),
        np.asarray([p[2] for p in chosen], dtype=np.int64),
    )


def _read_rows(path: Path, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    with h5py.File(path, "r") as f:
        arr = np.asarray(f["activations"][sorted_rows], dtype=np.float32)
    inv = np.argsort(order)
    return arr[inv]


def _sample_tokens(
    acts: np.ndarray,
    max_tokens: int,
    seed: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    flat = acts.reshape(-1, acts.shape[-1])
    if labels is None:
        flat_labels = None
    else:
        flat_labels = np.repeat(labels.astype(np.int64), acts.shape[1])
    if flat.shape[0] <= max_tokens:
        return flat, flat_labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], size=max_tokens, replace=False)
    return flat[idx], None if flat_labels is None else flat_labels[idx]


def _pool_for_probe(acts: np.ndarray, labels: np.ndarray, pool: str) -> tuple[np.ndarray, np.ndarray]:
    if pool == "mean":
        return acts.mean(axis=1), labels
    if pool == "tokens":
        return acts.reshape(-1, acts.shape[-1]), np.repeat(labels, acts.shape[1])
    raise ValueError(f"unknown pool mode {pool!r}")


def _remap_concept(labels: np.ndarray, concept_name: str) -> np.ndarray:
    concept = get_concept(concept_name)
    if not concept.available:
        raise ValueError(f"concept {concept_name!r} is unavailable")
    unique = np.unique(labels)
    mapping = {int(v): int(concept.label_fn(int(v))) for v in unique.tolist()}
    out = np.empty_like(labels, dtype=np.int64)
    for raw, mapped in mapping.items():
        out[labels == raw] = mapped
    return out


def _fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    max_iter: int,
    c_value: float,
    ridge_alpha: float,
) -> LogisticRegression | RidgeClassifier:
    if model == "ridge":
        clf = RidgeClassifier(alpha=ridge_alpha)
        clf.fit(x.astype(np.float32), y.astype(np.int64))
        return clf
    clf = LogisticRegression(C=c_value, max_iter=max_iter, solver="lbfgs")
    clf.fit(x.astype(np.float32), y.astype(np.int64))
    return clf


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_summary_md(out_dir: Path, title: str, lines: list[str]) -> None:
    payload = [f"# {title}", "", "> [!summary] TL;DR", *[f"> {line}" for line in lines[:4]], ""]
    payload.extend(lines)
    payload.append("")
    (out_dir / "summary.md").write_text("\n".join(payload), encoding="utf-8")


def _selected_cells(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.cells:
        cells = []
        for raw in args.cells:
            cleaned = raw.upper().replace("L", "").replace("T", "").replace(",", "_")
            parts = cleaned.split("_")
            if len(parts) != 2:
                raise ValueError(f"invalid cell spec {raw!r}; expected L3_T1")
            cells.append((int(parts[0]), int(parts[1])))
        return cells
    return [(int(layer), int(t_bin)) for layer in args.layers for t_bin in args.t_bins]


def _plot_pair_heatmap(
    rows: list[dict],
    metric: str,
    out_path: Path,
    *,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    labels = sorted({f"{r['source']}->{r['target']}" for r in rows})
    cells = sorted({(int(r["layer"]), int(r["t_bin"])) for r in rows})
    arr = np.full((len(labels), len(cells)), np.nan)
    label_idx = {v: i for i, v in enumerate(labels)}
    cell_idx = {v: i for i, v in enumerate(cells)}
    for row in rows:
        key = f"{row['source']}->{row['target']}"
        cell = (int(row["layer"]), int(row["t_bin"]))
        if key in label_idx and cell in cell_idx:
            arr[label_idx[key], cell_idx[cell]] = float(row[metric])
    fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(labels) + 2.4))
    im = ax.imshow(arr, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(
        range(len(cells)), [f"L{layer}/T{t_bin}" for layer, t_bin in cells], rotation=45, ha="right"
    )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_square_heatmap(rows: list[dict], metric: str, out_path: Path, title: str) -> None:
    labels = sorted({str(r["pair"]) for r in rows})
    arr = np.asarray([[float(r[metric])] for r in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5.5, 0.32 * max(4, len(labels)) + 2.0))
    im = ax.imshow(arr, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks([0], [metric])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.08, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_probe_transfer(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id, resume=args.resume)
    probe_path = out_dir / "metrics" / "probe_transfer.csv"
    align_path = out_dir / "metrics" / "alignment_quality.csv"
    probe_rows = _read_csv_rows(probe_path) if args.resume else []
    align_rows = _read_csv_rows(align_path) if args.resume else []
    completed_pairs = {
        (r["source"], r["target"], int(r["layer"]), int(r["t_bin"]))
        for r in align_rows
        if r.get("alignment_kind") == "procrustes"
    }
    for layer, t_bin in _selected_cells(args):
        info(f"probe-transfer cell L{layer}/T{t_bin}")
        for source, target in permutations(args.conditions, 2):
            pair_key = (source, target, int(layer), int(t_bin))
            if pair_key in completed_pairs:
                info(f"probe-transfer skip completed {source}->{target} L{layer}/T{t_bin}")
                continue
            info(f"probe-transfer pair {source}->{target} L{layer}/T{t_bin}")
            src_pos, tgt_pos, sample_idx = _aligned_positions(
                args.activations_root, source, target, args.dit_step, args.max_images, args.seed
            )
            src_path = _cell_path(args.activations_root, source, args.dit_step, layer, t_bin)
            tgt_path = _cell_path(args.activations_root, target, args.dit_step, layer, t_bin)
            src_acts = _read_rows(src_path, src_pos)
            tgt_acts = _read_rows(tgt_path, tgt_pos)
            labels = _resolve_labels(args.latents_root, source, sample_idx)

            src_fit, _ = _sample_tokens(src_acts, args.max_fit_tokens, args.seed + 11)
            tgt_fit, _ = _sample_tokens(tgt_acts, args.max_fit_tokens, args.seed + 11)
            src_eval, _ = _sample_tokens(src_acts, args.max_eval_tokens, args.seed + 13)
            tgt_eval, _ = _sample_tokens(tgt_acts, args.max_eval_tokens, args.seed + 13)
            maps = [
                fit_ridge_affine(tgt_fit, src_fit, alpha=args.ridge_alpha),
                fit_procrustes_affine(tgt_fit, src_fit),
            ]
            for amap in maps:
                metrics = alignment_metrics(amap.apply(tgt_eval), src_eval)
                align_rows.append(
                    {
                        "source": source,
                        "target": target,
                        "layer": layer,
                        "t_bin": t_bin,
                        "t_center": T_CENTERS[int(t_bin)],
                        "alignment_kind": amap.kind,
                        "n_fit_tokens": int(src_fit.shape[0]),
                        "n_eval_tokens": int(src_eval.shape[0]),
                        **metrics,
                    }
                )

            for concept_name in args.concepts:
                concept_labels = _remap_concept(labels, concept_name)
                src_x, y = _pool_for_probe(src_acts, concept_labels, args.pool)
                tgt_x, y_tgt = _pool_for_probe(tgt_acts, concept_labels, args.pool)
                idx = np.arange(src_x.shape[0])
                train_idx, test_idx = train_test_split(
                    idx, test_size=args.test_size, random_state=args.seed, stratify=None
                )
                if args.max_probe_train and train_idx.shape[0] > args.max_probe_train:
                    rng = np.random.default_rng(args.seed)
                    train_idx = rng.choice(train_idx, size=args.max_probe_train, replace=False)
                clf_src = _fit_probe(
                    src_x[train_idx],
                    y[train_idx],
                    model=args.probe_model,
                    max_iter=args.probe_max_iter,
                    c_value=args.probe_c,
                    ridge_alpha=args.ridge_probe_alpha,
                )
                clf_tgt = _fit_probe(
                    tgt_x[train_idx],
                    y_tgt[train_idx],
                    model=args.probe_model,
                    max_iter=args.probe_max_iter,
                    c_value=args.probe_c,
                    ridge_alpha=args.ridge_probe_alpha,
                )
                source_native = accuracy_score(y[test_idx], clf_src.predict(src_x[test_idx]))
                target_native = accuracy_score(y_tgt[test_idx], clf_tgt.predict(tgt_x[test_idx]))
                raw = accuracy_score(y_tgt[test_idx], clf_src.predict(tgt_x[test_idx]))
                base = {
                    "source": source,
                    "target": target,
                    "layer": layer,
                    "t_bin": t_bin,
                    "t_center": T_CENTERS[int(t_bin)],
                    "concept": concept_name,
                    "pool": args.pool,
                    "n_train": int(train_idx.shape[0]),
                    "n_test": int(test_idx.shape[0]),
                    "source_native_acc": float(source_native),
                    "target_native_acc": float(target_native),
                }
                probe_rows.append({**base, "alignment_kind": "source_native", "accuracy": float(source_native)})
                probe_rows.append({**base, "alignment_kind": "target_native", "accuracy": float(target_native)})
                probe_rows.append({**base, "alignment_kind": "target_raw", "accuracy": float(raw)})
                for amap in maps:
                    aligned = amap.apply(tgt_x[test_idx])
                    acc = accuracy_score(y_tgt[test_idx], clf_src.predict(aligned))
                    probe_rows.append({**base, "alignment_kind": amap.kind, "accuracy": float(acc)})
                del concept_labels, src_x, y, tgt_x, y_tgt, idx, train_idx, test_idx, clf_src, clf_tgt
            _write_csv(probe_path, probe_rows)
            _write_csv(align_path, align_rows)
            del src_acts, tgt_acts, src_fit, tgt_fit, src_eval, tgt_eval, labels, maps
            gc.collect()

    retained = [
        {
            **r,
            "target_retention": (
                float(r["accuracy"]) / float(r["target_native_acc"])
                if float(r["target_native_acc"]) > 0
                else float("nan")
            ),
        }
        for r in probe_rows
        if r["alignment_kind"] == "ridge"
    ]
    _plot_pair_heatmap(
        retained,
        "target_retention",
        out_dir / "plots" / "probe_transfer_retention_ridge.png",
        title="Probe transfer retention after ridge alignment",
        vmin=0.0,
        vmax=1.1,
    )
    summary = {
        "analysis": "probe-transfer",
        "conditions": list(args.conditions),
        "layers": list(args.layers),
        "t_bins": list(args.t_bins),
        "concepts": list(args.concepts),
        "n_probe_rows": len(probe_rows),
        "n_alignment_rows": len(align_rows),
        "criterion": "Alignment should recover target-native probe performance if dictionaries differ only by a simple residual-space transform.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_md(
        out_dir,
        "Probe Transfer + Linear Alignment",
        [
            "Ridge and Procrustes alignment are evaluated against target-native probe accuracy.",
            f"Wrote {len(probe_rows)} probe rows and {len(align_rows)} residual alignment rows.",
            "Interpret low aligned retention as evidence against a simple rotation/renaming account.",
        ],
    )
    ok(f"probe-transfer complete: {out_dir}")
    return 0


def _knn_stats(x: np.ndarray, y: np.ndarray, labels: np.ndarray, k: int) -> dict[str, float]:
    k_eff = min(k + 1, x.shape[0])
    x_nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine").fit(x).kneighbors(return_distance=False)
    y_nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine").fit(y).kneighbors(return_distance=False)
    x_nn = x_nn[:, 1:]
    y_nn = y_nn[:, 1:]
    x_purity = np.mean(labels[x_nn] == labels[:, None])
    y_purity = np.mean(labels[y_nn] == labels[:, None])
    overlaps = []
    for a, b in zip(x_nn, y_nn, strict=True):
        overlaps.append(len(set(a.tolist()) & set(b.tolist())) / max(1, a.shape[0]))
    return {
        "left_purity": float(x_purity),
        "right_purity": float(y_purity),
        "knn_overlap": float(np.mean(overlaps)),
        "k": int(x_nn.shape[1]),
    }


def run_residual_similarity(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id, resume=args.resume)
    sim_path = out_dir / "metrics" / "residual_similarity.csv"
    knn_path = out_dir / "metrics" / "rsa_knn.csv"
    sim_rows = _read_csv_rows(sim_path) if args.resume else []
    knn_rows = _read_csv_rows(knn_path) if args.resume else []
    completed_pairs = {str(r["pair"]) for r in sim_rows}

    def add_pair(
        left_cond: str,
        right_cond: str,
        left_layer: int,
        right_layer: int,
        left_t: int,
        right_t: int,
        pair_kind: str,
    ) -> None:
        pair = f"{left_cond}:L{left_layer}/T{left_t}__{right_cond}:L{right_layer}/T{right_t}"
        if pair in completed_pairs:
            info(f"residual-similarity skip completed {pair}")
            return
        info(f"residual-similarity pair {pair} ({pair_kind})")
        left_pos, right_pos, sample_idx = _aligned_positions(
            args.activations_root,
            left_cond,
            right_cond,
            args.dit_step,
            args.max_images,
            args.seed,
        )
        left = _read_rows(
            _cell_path(args.activations_root, left_cond, args.dit_step, left_layer, left_t),
            left_pos,
        )
        right = _read_rows(
            _cell_path(args.activations_root, right_cond, args.dit_step, right_layer, right_t),
            right_pos,
        )
        labels = _resolve_labels(args.latents_root, left_cond, sample_idx)
        left_tok, _ = _sample_tokens(left, args.max_tokens, args.seed + 21)
        right_tok, _ = _sample_tokens(right, args.max_tokens, args.seed + 21)
        left_img = left.mean(axis=1)
        right_img = right.mean(axis=1)
        sim_rows.append(
            {
                "pair": pair,
                "pair_kind": pair_kind,
                "left_condition": left_cond,
                "right_condition": right_cond,
                "left_layer": left_layer,
                "right_layer": right_layer,
                "left_t_bin": left_t,
                "right_t_bin": right_t,
                "linear_cka": linear_cka(left_tok, right_tok),
                "rsa_spearman": rsa_spearman(left_img, right_img),
                "n_images": int(left.shape[0]),
                "n_tokens": int(left_tok.shape[0]),
            }
        )
        knn_rows.append(
            {
                "pair": pair,
                "pair_kind": pair_kind,
                "left_condition": left_cond,
                "right_condition": right_cond,
                "left_layer": left_layer,
                "right_layer": right_layer,
                "left_t_bin": left_t,
                "right_t_bin": right_t,
                **_knn_stats(left_img, right_img, labels, args.knn_k),
            }
        )
        completed_pairs.add(pair)
        _write_csv(sim_path, sim_rows)
        _write_csv(knn_path, knn_rows)
        del left, right, left_tok, right_tok, left_img, right_img, labels
        gc.collect()

    requested = set(_selected_cells(args))
    if "cross_tokenizer_same_cell" in args.pair_kinds:
        for layer, t_bin in sorted(requested):
            for left, right in permutations(args.conditions, 2):
                add_pair(left, right, layer, layer, t_bin, t_bin, "cross_tokenizer_same_cell")
    if "within_tokenizer_cross_layer" in args.pair_kinds:
        for condition in args.conditions:
            for t_bin in args.t_bins:
                for left_layer, right_layer in permutations(args.layers, 2):
                    add_pair(
                        condition,
                        condition,
                        left_layer,
                        right_layer,
                        t_bin,
                        t_bin,
                        "within_tokenizer_cross_layer",
                    )
    if "within_tokenizer_cross_timestep" in args.pair_kinds:
        for condition in args.conditions:
            for layer in args.layers:
                for left_t, right_t in permutations(args.t_bins, 2):
                    add_pair(
                        condition,
                        condition,
                        layer,
                        layer,
                        left_t,
                        right_t,
                        "within_tokenizer_cross_timestep",
                    )

    _write_csv(sim_path, sim_rows)
    _write_csv(knn_path, knn_rows)
    _plot_square_heatmap(
        sim_rows,
        "linear_cka",
        out_dir / "plots" / "residual_similarity_cka.png",
        "Residual linear CKA",
    )
    _plot_square_heatmap(
        sim_rows,
        "rsa_spearman",
        out_dir / "plots" / "residual_similarity_rsa.png",
        "Image-level RSA Spearman",
    )
    summary = {
        "analysis": "residual-similarity",
        "conditions": list(args.conditions),
        "layers": list(args.layers),
        "t_bins": list(args.t_bins),
        "n_similarity_rows": len(sim_rows),
        "n_knn_rows": len(knn_rows),
        "criterion": "Tokenizer-blocked CKA/RSA structure means residual geometry diverges before SAE dictionary learning.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_md(
        out_dir,
        "Residual CKA/RSA Feature-Free Similarity",
        [
            "Residual streams are compared without SAE features.",
            f"Wrote {len(sim_rows)} CKA/RSA rows and {len(knn_rows)} kNN rows.",
            "Tokenizer-blocked structure supports residual geometric divergence before SAE learning.",
        ],
    )
    ok(f"residual-similarity complete: {out_dir}")
    return 0


def _resolve_sae_ckpt(sae_root: Path, condition: str, layer: int, t_bin: int, dit_step: int) -> Path:
    base = sae_root / condition / f"L{layer}_T{t_bin}" / f"step_{dit_step:06d}"
    finals = sorted(p for p in base.glob("final_*") if p.is_dir())
    if not finals:
        raise FileNotFoundError(f"no final_* SAE dir under {base}")
    return finals[-1]


def _load_matryoshka_sae(ckpt_dir: Path, device: torch.device) -> torch.nn.Module:
    from sae_lens import MatryoshkaBatchTopKTrainingSAE

    sae = MatryoshkaBatchTopKTrainingSAE.load_from_disk(str(ckpt_dir), device=str(device))
    sae.eval()
    return sae


@torch.no_grad()
def _sae_reconstruct(sae: torch.nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    device = next(sae.parameters()).device
    dtype = next(sae.parameters()).dtype
    out = []
    for start in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device=device, dtype=dtype)
        out.append(sae(xb).detach().cpu().float().numpy())
    return np.concatenate(out, axis=0)


def run_activation_proxy(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id)
    rows: list[dict] = []
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    for layer, t_bin in _selected_cells(args):
        for source, target in permutations(args.conditions, 2):
            info(f"activation-proxy {source}->{target} L{layer}/T{t_bin}")
            src_pos, tgt_pos, _ = _aligned_positions(
                args.activations_root, source, target, args.dit_step, args.max_images, args.seed
            )
            src_acts = _read_rows(
                _cell_path(args.activations_root, source, args.dit_step, layer, t_bin),
                src_pos,
            )
            tgt_acts = _read_rows(
                _cell_path(args.activations_root, target, args.dit_step, layer, t_bin),
                tgt_pos,
            )
            src_fit, _ = _sample_tokens(src_acts, args.max_fit_tokens, args.seed + 31)
            tgt_fit, _ = _sample_tokens(tgt_acts, args.max_fit_tokens, args.seed + 31)
            tgt_eval, _ = _sample_tokens(tgt_acts, args.max_eval_tokens, args.seed + 33)
            tgt_to_src = fit_ridge_affine(tgt_fit, src_fit, alpha=args.ridge_alpha)
            src_to_tgt = fit_ridge_affine(src_fit, tgt_fit, alpha=args.ridge_alpha)

            target_sae = _load_matryoshka_sae(
                _resolve_sae_ckpt(args.sae_root, target, layer, t_bin, args.dit_step), device
            )
            source_sae = _load_matryoshka_sae(
                _resolve_sae_ckpt(args.sae_root, source, layer, t_bin, args.dit_step), device
            )
            native = _sae_reconstruct(target_sae, tgt_eval, args.sae_batch_size)
            transferred_source_space = _sae_reconstruct(
                source_sae, tgt_to_src.apply(tgt_eval), args.sae_batch_size
            )
            transferred = src_to_tgt.apply(transferred_source_space)
            native_metrics = alignment_metrics(native, tgt_eval)
            transfer_metrics = alignment_metrics(transferred, tgt_eval)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "layer": layer,
                    "t_bin": t_bin,
                    "t_center": T_CENTERS[int(t_bin)],
                    "alignment_kind": "ridge_bidirectional",
                    "n_fit_tokens": int(src_fit.shape[0]),
                    "n_eval_tokens": int(tgt_eval.shape[0]),
                    "native_mse": native_metrics["mse"],
                    "native_cosine": native_metrics["cosine"],
                    "native_ev": native_metrics["explained_variance"],
                    "transferred_mse": transfer_metrics["mse"],
                    "transferred_cosine": transfer_metrics["cosine"],
                    "transferred_ev": transfer_metrics["explained_variance"],
                    "delta_ev": transfer_metrics["explained_variance"]
                    - native_metrics["explained_variance"],
                    "delta_cosine": transfer_metrics["cosine"] - native_metrics["cosine"],
                }
            )
            del target_sae, source_sae
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _write_csv(out_dir / "metrics" / "activation_proxy.csv", rows)
    _plot_pair_heatmap(
        rows,
        "delta_ev",
        out_dir / "plots" / "activation_proxy_delta_ev.png",
        title="Transferred source SAE EV minus native target SAE EV",
    )
    summary = {
        "analysis": "activation-proxy",
        "conditions": list(args.conditions),
        "layers": list(args.layers),
        "t_bins": list(args.t_bins),
        "n_rows": len(rows),
        "criterion": "Transferred source SAE should match native target SAE reconstruction if dictionaries are causally interchangeable.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_md(
        out_dir,
        "Cross-Tokenizer SAE Activation Proxy",
        [
            "Target residuals are mapped through source SAE dictionaries and mapped back.",
            f"Wrote {len(rows)} directed activation-proxy rows.",
            "Native-vs-transferred EV/cosine gaps indicate whether dictionaries are interchangeable.",
        ],
    )
    ok(f"activation-proxy complete: {out_dir}")
    return 0


def run_aggregate_runs(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id, resume=args.resume)
    matched: list[Path] = []
    for pattern in args.run_globs:
        matched.extend(sorted(args.out_root.glob(pattern)))
    matched = sorted({p for p in matched if p.is_dir() and p != out_dir})
    if not matched:
        raise FileNotFoundError(f"no run directories matched: {args.run_globs}")

    summary: dict[str, object] = {
        "analysis": f"{args.analysis}-aggregate",
        "source_runs": [str(p) for p in matched],
    }
    if args.analysis == "probe-transfer":
        probe_rows: list[dict] = []
        align_rows: list[dict] = []
        for run_dir in matched:
            probe_rows.extend(_read_csv_rows(run_dir / "metrics" / "probe_transfer.csv"))
            align_rows.extend(_read_csv_rows(run_dir / "metrics" / "alignment_quality.csv"))
        _write_csv(out_dir / "metrics" / "probe_transfer.csv", probe_rows)
        _write_csv(out_dir / "metrics" / "alignment_quality.csv", align_rows)
        retained = [
            {
                **r,
                "target_retention": (
                    float(r["accuracy"]) / float(r["target_native_acc"])
                    if float(r["target_native_acc"]) > 0
                    else float("nan")
                ),
            }
            for r in probe_rows
            if r["alignment_kind"] == "ridge"
        ]
        if retained:
            _plot_pair_heatmap(
                retained,
                "target_retention",
                out_dir / "plots" / "probe_transfer_retention_ridge.png",
                title="Probe transfer retention after ridge alignment",
                vmin=0.0,
                vmax=1.1,
            )
        summary.update({"n_probe_rows": len(probe_rows), "n_alignment_rows": len(align_rows)})
    elif args.analysis == "residual-similarity":
        sim_rows: list[dict] = []
        knn_rows: list[dict] = []
        for run_dir in matched:
            sim_rows.extend(_read_csv_rows(run_dir / "metrics" / "residual_similarity.csv"))
            knn_rows.extend(_read_csv_rows(run_dir / "metrics" / "rsa_knn.csv"))
        _write_csv(out_dir / "metrics" / "residual_similarity.csv", sim_rows)
        _write_csv(out_dir / "metrics" / "rsa_knn.csv", knn_rows)
        if sim_rows:
            _plot_square_heatmap(
                sim_rows,
                "linear_cka",
                out_dir / "plots" / "residual_similarity_cka.png",
                "Residual linear CKA",
            )
            _plot_square_heatmap(
                sim_rows,
                "rsa_spearman",
                out_dir / "plots" / "residual_similarity_rsa.png",
                "Image-level RSA Spearman",
            )
        summary.update({"n_similarity_rows": len(sim_rows), "n_knn_rows": len(knn_rows)})
    else:
        raise ValueError(f"unknown aggregate analysis {args.analysis!r}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_md(
        out_dir,
        f"{args.analysis} Aggregate",
        [
            f"Aggregated {len(matched)} chunked run directories.",
            "Merged metrics are written under this aggregate run directory.",
            "Use the source_runs field in summary.json to audit provenance.",
        ],
    )
    ok(f"aggregate complete: {out_dir}")
    return 0


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--model_variant", type=str, default="sit_b_2",
                   help="SiT variant id or model name; used for auto layers and namespaced roots.")
    p.add_argument("--layers", nargs="+", default=["auto"])
    p.add_argument("--t_bins", nargs="+", type=int, default=list(T_BINS))
    p.add_argument("--dit_step", type=int, default=200_000)
    p.add_argument("--activations_root", type=Path, default=None)
    p.add_argument("--latents_root", type=Path, default=LATENTS_VAL)
    p.add_argument("--out_root", type=Path, default=None)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--resume", action="store_true", help="Resume into an existing run directory.")
    p.add_argument("--max_images", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cells", nargs="+", default=None, help="Optional exact cells, e.g. L3_T1 L6_T2.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe-transfer")
    _add_common_args(probe)
    probe.add_argument("--concepts", nargs="+", default=list(DEFAULT_CONCEPTS))
    probe.add_argument("--pool", choices=["mean", "tokens"], default="mean")
    probe.add_argument("--probe_model", choices=["ridge", "logistic"], default="ridge")
    probe.add_argument("--test_size", type=float, default=0.25)
    probe.add_argument("--max_fit_tokens", type=int, default=50_000)
    probe.add_argument("--max_eval_tokens", type=int, default=50_000)
    probe.add_argument("--max_probe_train", type=int, default=50_000)
    probe.add_argument("--ridge_alpha", type=float, default=10.0)
    probe.add_argument("--ridge_probe_alpha", type=float, default=1.0)
    probe.add_argument("--probe_max_iter", type=int, default=1000)
    probe.add_argument("--probe_c", type=float, default=1.0)
    probe.set_defaults(func=run_probe_transfer)

    sim = sub.add_parser("residual-similarity")
    _add_common_args(sim)
    sim.add_argument(
        "--pair_kinds",
        nargs="+",
        default=[
            "cross_tokenizer_same_cell",
            "within_tokenizer_cross_layer",
            "within_tokenizer_cross_timestep",
        ],
        choices=[
            "cross_tokenizer_same_cell",
            "within_tokenizer_cross_layer",
            "within_tokenizer_cross_timestep",
        ],
    )
    sim.add_argument("--max_tokens", type=int, default=50_000)
    sim.add_argument("--knn_k", type=int, default=10)
    sim.set_defaults(func=run_residual_similarity)

    proxy = sub.add_parser("activation-proxy")
    _add_common_args(proxy)
    proxy.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    proxy.add_argument("--max_fit_tokens", type=int, default=50_000)
    proxy.add_argument("--max_eval_tokens", type=int, default=50_000)
    proxy.add_argument("--ridge_alpha", type=float, default=10.0)
    proxy.add_argument("--sae_batch_size", type=int, default=8192)
    proxy.add_argument("--device", type=str, default=None)
    proxy.set_defaults(func=run_activation_proxy)

    agg = sub.add_parser("aggregate-runs")
    agg.add_argument("--analysis", choices=["probe-transfer", "residual-similarity"], required=True)
    agg.add_argument("--run_globs", nargs="+", required=True)
    agg.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    agg.add_argument("--run_id", type=str, default=None)
    agg.add_argument("--resume", action="store_true")
    agg.set_defaults(func=run_aggregate_runs)
    return p


def _normalize_common_args(args: argparse.Namespace) -> None:
    if not hasattr(args, "model_variant"):
        return
    spec = model_variant_spec(args.model_variant)
    args.model_variant = spec.variant_id
    args.layers = parse_layers(args.layers, spec)
    if args.activations_root is None:
        namespaced = model_subdir(SCRATCH_PROJECT_ROOT, spec.variant_id, "activations_ynull")
        args.activations_root = (
            namespaced if namespaced.exists() or spec.variant_id != "sit_b_2" else ACTIVATIONS_YNULL
        )
    if args.out_root is None:
        args.out_root = model_subdir(
            OUTPUT_ROOT, spec.variant_id, "analysis", "tokenizer_dictionary_validation"
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _normalize_common_args(args)
    if hasattr(args, "t_bins") and any(t not in T_CENTERS for t in args.t_bins):
        parser.error(f"--t_bins must be drawn from {sorted(T_CENTERS)}")
    if hasattr(args, "max_images") and args.max_images < 2:
        parser.error("--max_images must be at least 2")
    return int(args.func(args))


if __name__ == "__main__":
    if not math.isfinite(linear_cka(np.eye(2), np.eye(2))):
        raise RuntimeError("linear_cka sanity check failed")
    sys.exit(main())
