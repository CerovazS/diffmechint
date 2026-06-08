"""Concept-margin SAE feature attribution (EAP) for semantic probes (E22, E29)."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from diffmechint.analysis.alignment import (
    ACTIVATIONS_YNULL,
    LATENTS_VAL,
    SAE_ROOT,
    T_CENTERS,
    cell_path,
    load_manifest,
    read_rows,
    resolve_labels,
)
from diffmechint.analysis.patching import (
    DEFAULT_CONDITIONS,
    DEFAULT_OUT_ROOT,
    LAYERS,
    PB,
    T_BINS,
    decoder_weight,
    read_csv,
    selected_cells,
    write_csv,
)
from diffmechint.probing.concepts import get_concept
from diffmechint.sae import load_matryoshka_sae, resolve_sae_ckpt
from diffmechint.utils import info, make_run_dir, ok, warn

DEFAULT_CONCEPTS = ("animal_binary", "vehicle_binary", "food_binary", "instrument_binary")


def _parse_float(raw: object, default: float = math.nan) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) else value


def _feature_ids_from_bank(
    feature_bank: Path | None,
    condition: str,
    layer: int,
    t_bin: int,
    max_features: int | None,
    required_feature_ids: set[int] | None = None,
) -> tuple[list[int], dict[int, dict]]:
    if feature_bank is None:
        return [], {}
    rows = [
        row
        for row in read_csv(feature_bank)
        if str(row["condition"]) == condition and int(row["layer"]) == layer and int(row["t_bin"]) == t_bin
    ]
    rows.sort(
        key=lambda row: (
            _parse_float(row.get("entropy"), default=999.0),
            -_parse_float(row.get("top_activation"), default=0.0),
        )
    )
    required_feature_ids = required_feature_ids or set()
    rows_by_id = {int(row["feature_id"]): row for row in rows}
    if max_features:
        selected = rows[:max_features]
        selected_ids = {int(row["feature_id"]) for row in selected}
        for feature_id in sorted(required_feature_ids):
            if feature_id in rows_by_id and feature_id not in selected_ids:
                selected.append(rows_by_id[feature_id])
        rows = selected
    ids = [int(row["feature_id"]) for row in rows]
    return ids, {int(row["feature_id"]): row for row in rows}


def _candidate_requirements(candidate_manifest: Path | None) -> tuple[dict[tuple[str, int, int], set[int]], list[dict]]:
    if candidate_manifest is None:
        return {}, []
    rows = read_csv(candidate_manifest)
    required: dict[tuple[str, int, int], set[int]] = {}
    for row in rows:
        layer = int(row["layer"])
        t_bin = int(row["t_bin"])
        for role in ("source", "target"):
            key = (str(row[role]), layer, t_bin)
            required.setdefault(key, set()).add(int(row[f"{role}_feature_id"]))
    return required, rows


def _balanced_indices(labels: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape[0] <= max_count:
        return np.arange(labels.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, max_count // max(len(classes), 1))
    chosen: list[np.ndarray] = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls)
        take = min(idx.shape[0], per_class)
        chosen.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    if out.shape[0] < max_count:
        remaining = np.setdiff1d(np.arange(labels.shape[0], dtype=np.int64), out, assume_unique=False)
        if remaining.shape[0] > 0:
            fill = min(max_count - out.shape[0], remaining.shape[0])
            out = np.concatenate([out, rng.choice(remaining, size=fill, replace=False)])
    out = np.sort(out[:max_count])
    return out.astype(np.int64)


def _concept_image_positions(labels_all: np.ndarray, concept_name: str, max_count: int, seed: int) -> np.ndarray:
    if labels_all.shape[0] <= max_count:
        return np.arange(labels_all.shape[0], dtype=np.int64)
    try:
        concept_labels = _binary_labels(labels_all, concept_name)
    except (KeyError, ValueError, NotImplementedError):
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(labels_all.shape[0], size=max_count, replace=False)).astype(np.int64)
    if len(np.unique(concept_labels)) < 2:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(labels_all.shape[0], size=max_count, replace=False)).astype(np.int64)
    return _balanced_indices(concept_labels, max_count, seed)


def _binary_labels(raw_labels: np.ndarray, concept_name: str) -> np.ndarray:
    concept = get_concept(concept_name)
    if concept.num_classes != 2:
        raise ValueError(f"concept-eap currently expects binary concepts, got {concept_name}")
    if not concept.available:
        raise ValueError(f"concept {concept_name!r} is unavailable")
    mapping = {int(v): int(concept.label_fn(int(v))) for v in np.unique(raw_labels).tolist()}
    out = np.empty_like(raw_labels, dtype=np.int64)
    for src, dst in mapping.items():
        out[raw_labels == src] = dst
    return out


def _margin_np(x: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return x.astype(np.float32) @ coef.astype(np.float32) + float(intercept)


def _margin_gap(margin: np.ndarray, labels: np.ndarray) -> float:
    pos = margin[labels == 1]
    neg = margin[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(pos.mean() - neg.mean())


def _margin_stats(margin: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = (margin > 0).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "margin_gap": _margin_gap(margin, labels),
        "mean_margin_pos": float(np.mean(margin[labels == 1])) if np.any(labels == 1) else float("nan"),
        "mean_margin_neg": float(np.mean(margin[labels == 0])) if np.any(labels == 0) else float("nan"),
    }


def _append_margin(
    store: dict[str, list[float]],
    name: str,
    values: torch.Tensor,
) -> None:
    store.setdefault(name, []).append(values.detach().float().cpu().numpy())


def _concat_margin(store: dict[str, list[float]], name: str) -> np.ndarray:
    vals = store.get(name, [])
    if not vals:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(vals).astype(np.float32)


def _evaluate_faithfulness(
    sae: torch.nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    top_feature_ids: list[int],
    *,
    device: torch.device,
    batch_size: int,
    topk_values: list[int],
) -> tuple[list[dict], dict[str, float]]:
    sae_dtype = next(sae.parameters()).dtype
    coef_t = torch.from_numpy(coef.astype(np.float32)).to(device=device)
    x_t = torch.from_numpy(x_eval.astype(np.float32)).to(device=device, dtype=sae_dtype)
    topk_values = sorted({min(k, len(top_feature_ids)) for k in topk_values if k > 0 and top_feature_ids})
    top_tensors = {
        k: torch.tensor(top_feature_ids[:k], device=device, dtype=torch.long)
        for k in topk_values
    }
    margins: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, x_t.shape[0], batch_size):
            xb = x_t[start : start + batch_size]
            z = sae.encode(xb)
            rec = sae.decode(z)
            err = xb - rec
            clean_margin = xb.float() @ coef_t + float(intercept)
            recon_margin = rec.float() @ coef_t + float(intercept)
            err_margin = err.float() @ coef_t
            _append_margin(margins, "clean", clean_margin)
            _append_margin(margins, "sae_reconstruction", recon_margin)
            _append_margin(margins, "reconstruction_error", err_margin)
            for k, idx in top_tensors.items():
                z_keep = torch.zeros_like(z)
                z_keep[:, idx] = z[:, idx]
                z_drop = z.clone()
                z_drop[:, idx] = 0
                kept = sae.decode(z_keep)
                dropped = sae.decode(z_drop) + err
                _append_margin(margins, f"top{k}_features_only", kept.float() @ coef_t + float(intercept))
                _append_margin(margins, f"top{k}_with_error", (kept + err).float() @ coef_t + float(intercept))
                _append_margin(margins, f"top{k}_ablated", dropped.float() @ coef_t + float(intercept))
    clean = _margin_stats(_concat_margin(margins, "clean"), y_eval)
    clean_gap = clean["margin_gap"]
    recon = _margin_stats(_concat_margin(margins, "sae_reconstruction"), y_eval)
    err_margin = _concat_margin(margins, "reconstruction_error")
    error_gap = _margin_gap(err_margin, y_eval)
    rows = []
    for k in topk_values:
        with_error = _margin_stats(_concat_margin(margins, f"top{k}_with_error"), y_eval)
        features_only = _margin_stats(_concat_margin(margins, f"top{k}_features_only"), y_eval)
        ablated = _margin_stats(_concat_margin(margins, f"top{k}_ablated"), y_eval)
        denom = clean_gap if abs(clean_gap) > 1e-12 else float("nan")
        rows.append(
            {
                "top_k": int(k),
                "sufficiency_with_error_margin_gap": with_error["margin_gap"],
                "sufficiency_features_only_margin_gap": features_only["margin_gap"],
                "necessity_ablated_margin_gap": ablated["margin_gap"],
                "sufficiency_with_error_retention": with_error["margin_gap"] / denom,
                "sufficiency_features_only_retention": features_only["margin_gap"] / denom,
                "necessity_drop_fraction": (clean_gap - ablated["margin_gap"]) / denom,
                "sufficiency_with_error_accuracy": with_error["accuracy"],
                "sufficiency_features_only_accuracy": features_only["accuracy"],
                "necessity_ablated_accuracy": ablated["accuracy"],
            }
        )
    base = {
        "clean_accuracy": clean["accuracy"],
        "clean_margin_gap": clean_gap,
        "sae_reconstruction_accuracy": recon["accuracy"],
        "sae_reconstruction_margin_gap": recon["margin_gap"],
        "error_node_margin_gap": error_gap,
        "error_node_abs_gap_fraction": abs(error_gap) / abs(clean_gap) if abs(clean_gap) > 1e-12 else float("nan"),
    }
    return rows, base


def _plot_top_features(rows: list[dict], out_path: Path, title: str) -> None:
    if not rows:
        return
    top = rows[: min(20, len(rows))]
    labels = [f"F{row['feature_id']}" for row in top][::-1]
    vals = [float(row["signed_attribution"]) for row in top][::-1]
    colors = [PB["blue"] if v >= 0 else PB["red"] for v in vals]
    fig, ax = plt.subplots(figsize=(7.0, max(4.0, 0.28 * len(labels) + 1.4)))
    ax.axvline(0.0, color=PB["dark"], linewidth=1.0)
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("signed attribution to concept margin")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_faithfulness(rows: list[dict], out_path: Path, title: str) -> None:
    if not rows:
        return
    x = [int(row["top_k"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.axhline(1.0, color=PB["dark"], linewidth=1.0, linestyle=":")
    ax.plot(x, [float(row["sufficiency_with_error_retention"]) for row in rows], marker="o", color=PB["blue"], label="suff + error")
    ax.plot(x, [float(row["sufficiency_features_only_retention"]) for row in rows], marker="o", color=PB["gold"], label="features only")
    ax.plot(x, [float(row["necessity_drop_fraction"]) for row in rows], marker="o", color=PB["red"], label="necessity drop")
    ax.set_xscale("log")
    ax.set_xlabel("top-k SAE features")
    ax.set_ylabel("fraction of clean concept-margin gap")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_concept_eap(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id, resume=args.resume)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    required_by_cell, candidate_rows = _candidate_requirements(args.candidate_manifest)
    all_node_rows: list[dict] = []
    all_edge_rows: list[dict] = []
    all_faithfulness_rows: list[dict] = []
    circuit_payloads: list[dict] = []
    summary_rows: list[dict] = []

    for condition in args.conditions:
        manifest = load_manifest(args.activations_root, condition, args.dit_step)
        sample_idx_all = np.asarray(manifest["sample_idx"], dtype=np.int64)
        labels_all = resolve_labels(args.latents_root, condition, sample_idx_all)
        for layer, t_bin in selected_cells(args):
            feature_ids, bank_by_id = _feature_ids_from_bank(
                args.feature_bank,
                condition,
                layer,
                t_bin,
                args.max_features_per_cell,
                required_by_cell.get((condition, layer, t_bin), set()),
            )
            sae = load_matryoshka_sae(resolve_sae_ckpt(args.sae_root, condition, layer, t_bin, args.dit_step), device)
            if not feature_ids:
                d_sae = int(sae.cfg.d_sae)
                take = min(args.max_features_per_cell or d_sae, d_sae)
                feature_ids = list(range(take))
                bank_by_id = {}
                warn(f"no feature bank rows for {condition} L{layer}/T{t_bin}; using first {take} SAE features")
            decoder = decoder_weight(sae).float().detach().cpu().numpy()
            decoder_rows = decoder[np.asarray(feature_ids, dtype=np.int64)]
            for concept_name in args.concepts:
                info(f"concept-eap {condition} L{layer}/T{t_bin} {concept_name}")
                img_pos = _concept_image_positions(
                    labels_all,
                    concept_name,
                    min(args.max_images, sample_idx_all.shape[0]),
                    args.seed + 17 * layer + t_bin,
                )
                acts = read_rows(
                    cell_path(args.activations_root, condition, args.dit_step, layer, t_bin),
                    img_pos,
                ).astype(np.float32)
                labels = labels_all[img_pos]
                flat = acts.reshape(-1, acts.shape[-1])
                concept_labels_img = _binary_labels(labels, concept_name)
                y_tokens = np.repeat(concept_labels_img, acts.shape[1])
                if len(np.unique(y_tokens)) < 2:
                    warn(f"skip {condition} L{layer}/T{t_bin} {concept_name}: one-class sample")
                    continue
                probe_idx = _balanced_indices(y_tokens, args.max_probe_tokens, args.seed + 31)
                train_idx, test_idx = train_test_split(
                    probe_idx,
                    test_size=args.test_size,
                    random_state=args.seed,
                    stratify=y_tokens[probe_idx],
                )
                clf = RidgeClassifier(alpha=args.ridge_probe_alpha)
                clf.fit(flat[train_idx].astype(np.float32), y_tokens[train_idx].astype(np.int64))
                coef = clf.coef_.reshape(-1).astype(np.float32)
                intercept = float(np.ravel(clf.intercept_)[0])
                probe_margin = _margin_np(flat[test_idx], coef, intercept)
                probe_stats = _margin_stats(probe_margin, y_tokens[test_idx])

                eval_idx = _balanced_indices(y_tokens, args.max_eval_tokens, args.seed + 43)
                x_eval = flat[eval_idx].astype(np.float32)
                y_eval = y_tokens[eval_idx].astype(np.int64)
                sae_dtype = next(sae.parameters()).dtype
                feat_t = torch.tensor(feature_ids, device=device, dtype=torch.long)
                pos_sum = np.zeros(len(feature_ids), dtype=np.float64)
                neg_sum = np.zeros(len(feature_ids), dtype=np.float64)
                pos_count = 0
                neg_count = 0
                density_count = np.zeros(len(feature_ids), dtype=np.int64)
                x_eval_t = torch.from_numpy(x_eval).to(device=device, dtype=sae_dtype)
                with torch.no_grad():
                    for start in range(0, x_eval_t.shape[0], args.sae_batch_size):
                        xb = x_eval_t[start : start + args.sae_batch_size]
                        yb = y_eval[start : start + xb.shape[0]]
                        z = sae.encode(xb).index_select(1, feat_t).detach().float().cpu().numpy()
                        pos_mask = yb == 1
                        neg_mask = yb == 0
                        if np.any(pos_mask):
                            pos_sum += z[pos_mask].sum(axis=0)
                            pos_count += int(pos_mask.sum())
                        if np.any(neg_mask):
                            neg_sum += z[neg_mask].sum(axis=0)
                            neg_count += int(neg_mask.sum())
                        density_count += (z > 0).sum(axis=0)
                mean_pos = pos_sum / max(pos_count, 1)
                mean_neg = neg_sum / max(neg_count, 1)
                activation_delta = mean_pos - mean_neg
                direction_score = decoder_rows @ coef
                signed_attr = activation_delta * direction_score
                abs_attr = np.abs(signed_attr)
                order = np.argsort(-abs_attr)
                top_feature_ids = [feature_ids[int(i)] for i in order[: args.max_circuit_features]]
                concept_node = f"{condition}:L{layer}:T{t_bin}:{concept_name}:margin"
                node_rows = []
                edge_rows = []
                for rank, idx in enumerate(order, start=1):
                    fid = int(feature_ids[int(idx)])
                    bank = bank_by_id.get(fid, {})
                    node_id = f"{condition}:L{layer}:T{t_bin}:F{fid}"
                    row = {
                        "node_id": node_id,
                        "node_type": "sae_feature",
                        "condition": condition,
                        "dit_step": args.dit_step,
                        "layer": layer,
                        "t_bin": t_bin,
                        "t_center": T_CENTERS[int(t_bin)],
                        "concept": concept_name,
                        "feature_id": fid,
                        "rank_abs": rank,
                        "signed_attribution": float(signed_attr[int(idx)]),
                        "absolute_attribution": float(abs_attr[int(idx)]),
                        "decoder_probe_direction_score": float(direction_score[int(idx)]),
                        "mean_activation_pos": float(mean_pos[int(idx)]),
                        "mean_activation_neg": float(mean_neg[int(idx)]),
                        "activation_delta_pos_minus_neg": float(activation_delta[int(idx)]),
                        "activation_density_eval": float(density_count[int(idx)] / max(x_eval.shape[0], 1)),
                        "top_label": bank.get("top_label", ""),
                        "top_class_idx": bank.get("top_class_idx", ""),
                        "vlm_interpretation": bank.get("vlm_interpretation", ""),
                    }
                    node_rows.append(row)
                    edge_rows.append(
                        {
                            "source_node": node_id,
                            "target_node": concept_node,
                            "edge_type": "decoder_direction_to_probe_margin",
                            "condition": condition,
                            "dit_step": args.dit_step,
                            "layer": layer,
                            "t_bin": t_bin,
                            "concept": concept_name,
                            "feature_id": fid,
                            "signed_attribution": float(signed_attr[int(idx)]),
                            "absolute_attribution": float(abs_attr[int(idx)]),
                            "rank_abs": rank,
                        }
                    )
                faith_rows, base_faith = _evaluate_faithfulness(
                    sae,
                    x_eval,
                    y_eval,
                    coef,
                    intercept,
                    top_feature_ids,
                    device=device,
                    batch_size=args.sae_batch_size,
                    topk_values=args.topk_values,
                )
                error_node = f"{condition}:L{layer}:T{t_bin}:SAE_ERROR"
                node_rows.append(
                    {
                        "node_id": error_node,
                        "node_type": "reconstruction_error",
                        "condition": condition,
                        "dit_step": args.dit_step,
                        "layer": layer,
                        "t_bin": t_bin,
                        "t_center": T_CENTERS[int(t_bin)],
                        "concept": concept_name,
                        "feature_id": "error",
                        "rank_abs": "",
                        "signed_attribution": base_faith["error_node_margin_gap"],
                        "absolute_attribution": abs(base_faith["error_node_margin_gap"]),
                        "decoder_probe_direction_score": "",
                        "mean_activation_pos": "",
                        "mean_activation_neg": "",
                        "activation_delta_pos_minus_neg": "",
                        "activation_density_eval": "",
                        "top_label": "",
                        "top_class_idx": "",
                        "vlm_interpretation": "SAE reconstruction error node",
                    }
                )
                edge_rows.append(
                    {
                        "source_node": error_node,
                        "target_node": concept_node,
                        "edge_type": "error_projection_to_probe_margin",
                        "condition": condition,
                        "dit_step": args.dit_step,
                        "layer": layer,
                        "t_bin": t_bin,
                        "concept": concept_name,
                        "feature_id": "error",
                        "signed_attribution": base_faith["error_node_margin_gap"],
                        "absolute_attribution": abs(base_faith["error_node_margin_gap"]),
                        "rank_abs": "",
                    }
                )
                for row in faith_rows:
                    row.update(
                        {
                            "condition": condition,
                            "dit_step": args.dit_step,
                            "layer": layer,
                            "t_bin": t_bin,
                            "concept": concept_name,
                        }
                    )
                all_node_rows.extend(node_rows)
                all_edge_rows.extend(edge_rows)
                all_faithfulness_rows.extend(faith_rows)
                circuit_payloads.append(
                    {
                        "condition": condition,
                        "dit_step": args.dit_step,
                        "layer": layer,
                        "t_bin": t_bin,
                        "concept": concept_name,
                        "target_node": concept_node,
                        "top_feature_ids": top_feature_ids,
                        "error_node": error_node,
                        "probe": {
                            "model": "RidgeClassifier",
                            "ridge_alpha": args.ridge_probe_alpha,
                            "test_accuracy": probe_stats["accuracy"],
                            "test_margin_gap": probe_stats["margin_gap"],
                            "n_probe_train": int(train_idx.shape[0]),
                            "n_probe_test": int(test_idx.shape[0]),
                        },
                        "faithfulness": base_faith,
                    }
                )
                summary_rows.append(
                    {
                        "condition": condition,
                        "layer": layer,
                        "t_bin": t_bin,
                        "concept": concept_name,
                        "n_features_scored": len(feature_ids),
                        "probe_test_accuracy": probe_stats["accuracy"],
                        "probe_margin_gap": probe_stats["margin_gap"],
                        "clean_margin_gap": base_faith["clean_margin_gap"],
                        "sae_reconstruction_margin_gap": base_faith["sae_reconstruction_margin_gap"],
                        "error_node_abs_gap_fraction": base_faith["error_node_abs_gap_fraction"],
                        "top1_feature_id": top_feature_ids[0] if top_feature_ids else "",
                        "top1_abs_attribution": float(abs_attr[order[0]]) if len(order) else float("nan"),
                    }
                )
                scoped = f"{condition}_L{layer}_T{t_bin}_{concept_name}"
                _plot_top_features(node_rows[:-1], out_dir / "plots" / f"{scoped}_top_features.png", scoped)
                _plot_faithfulness(faith_rows, out_dir / "plots" / f"{scoped}_topn_faithfulness_curve.png", scoped)
                del clf, x_eval, y_eval, x_eval_t, acts
                gc.collect()
            del sae
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(out_dir / "metrics" / "eap_feature_nodes.csv", all_node_rows)
    write_csv(out_dir / "metrics" / "eap_feature_edges.csv", all_edge_rows)
    write_csv(out_dir / "metrics" / "topn_faithfulness.csv", all_faithfulness_rows)
    write_csv(out_dir / "metrics" / "concept_eap_summary.csv", summary_rows)
    error_share_rows = [
        {
            "condition": row["condition"],
            "dit_step": args.dit_step,
            "layer": row["layer"],
            "t_bin": row["t_bin"],
            "concept": row["concept"],
            "clean_margin_gap": row["clean_margin_gap"],
            "sae_reconstruction_margin_gap": row["sae_reconstruction_margin_gap"],
            "error_node_abs_gap_fraction": row["error_node_abs_gap_fraction"],
        }
        for row in summary_rows
    ]
    if error_share_rows:
        write_csv(out_dir / "metrics" / "eap_error_node_share.csv", error_share_rows)
    if candidate_rows:
        node_by_key = {
            (
                str(row["condition"]),
                int(row["layer"]),
                int(row["t_bin"]),
                str(row["concept"]),
                str(row["feature_id"]),
            ): row
            for row in all_node_rows
        }
        candidate_rank_rows = []
        for candidate in candidate_rows:
            concept = f"class:{candidate['target_top_class_idx']}"
            layer = int(candidate["layer"])
            t_bin = int(candidate["t_bin"])
            for role in ("source", "target"):
                condition = str(candidate[role])
                feature_id = str(candidate[f"{role}_feature_id"])
                node = node_by_key.get((condition, layer, t_bin, concept, feature_id), {})
                rank = int(node["rank_abs"]) if node.get("rank_abs") not in {None, ""} else None
                candidate_rank_rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "selection_role": candidate.get("selection_role", ""),
                        "candidate_side": role,
                        "condition": condition,
                        "layer": layer,
                        "t_bin": t_bin,
                        "concept": concept,
                        "feature_id": feature_id,
                        "found_in_eap": bool(node),
                        "rank_abs": "" if rank is None else rank,
                        "in_top1": bool(rank is not None and rank <= 1),
                        "in_top5": bool(rank is not None and rank <= 5),
                        "in_top10": bool(rank is not None and rank <= 10),
                        "in_top25": bool(rank is not None and rank <= 25),
                        "in_top50": bool(rank is not None and rank <= 50),
                        "in_top100": bool(rank is not None and rank <= 100),
                        "signed_attribution": node.get("signed_attribution", ""),
                        "absolute_attribution": node.get("absolute_attribution", ""),
                        "top_label": node.get("top_label", ""),
                        "vlm_interpretation": node.get("vlm_interpretation", ""),
                    }
                )
        write_csv(out_dir / "metrics" / "eap_candidate_ranks.csv", candidate_rank_rows)
    payload = {
        "analysis": "sae-concept-eap",
        "conditions": list(args.conditions),
        "cells": [f"L{layer}_T{t_bin}" for layer, t_bin in selected_cells(args)],
        "concepts": list(args.concepts),
        "feature_bank": "" if args.feature_bank is None else str(args.feature_bank),
        "n_feature_nodes": sum(1 for row in all_node_rows if row["node_type"] == "sae_feature"),
        "n_error_nodes": sum(1 for row in all_node_rows if row["node_type"] == "reconstruction_error"),
        "n_edges": len(all_edge_rows),
        "n_circuits": len(circuit_payloads),
        "max_images": args.max_images,
        "max_probe_tokens": args.max_probe_tokens,
        "max_eval_tokens": args.max_eval_tokens,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "circuit.json").write_text(json.dumps(circuit_payloads, indent=2), encoding="utf-8")
    (out_dir / "faithfulness.json").write_text(
        json.dumps({"summary": payload, "circuits": circuit_payloads}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# SAE Concept EAP",
        "",
        "> [!summary] TL;DR",
        f"> **SAE concept-margin EAP** scored =={payload['n_feature_nodes']}== feature nodes across =={payload['n_circuits']}== circuits.",
        "> Edges are decoder-direction attributions to binary concept probes; reconstruction-error nodes are explicit.",
        "> This is a candidate-discovery layer: final mechanistic claims still require activation/sampling patch validation.",
        "",
        "## Method",
        "",
        "- Train a binary RidgeClassifier probe on y-null residual tokens for each selected condition/cell/concept.",
        "- Score SAE features by `(mean activation on concept-positive tokens - mean activation on negatives) * decoder-probe direction score`.",
        "- Evaluate top-k sufficiency and necessity using SAE decode with the reconstruction-error node tracked separately.",
        "",
        "## Outputs",
        "",
        "- `metrics/eap_feature_nodes.csv`",
        "- `metrics/eap_feature_edges.csv`",
        "- `metrics/topn_faithfulness.csv`",
        "- `circuit.json`",
        "- `faithfulness.json`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    ok(f"concept EAP complete: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    p.add_argument("--t_bins", nargs="+", type=int, default=list(T_BINS))
    p.add_argument("--cells", nargs="+", default=None)
    p.add_argument("--concepts", nargs="+", default=list(DEFAULT_CONCEPTS))
    p.add_argument("--dit_step", type=int, default=200_000)
    p.add_argument("--activations_root", type=Path, default=ACTIVATIONS_YNULL)
    p.add_argument("--latents_root", type=Path, default=LATENTS_VAL)
    p.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    p.add_argument("--feature_bank", type=Path, default=None)
    p.add_argument("--candidate_manifest", type=Path, default=None)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_images", type=int, default=1024)
    p.add_argument("--max_probe_tokens", type=int, default=100_000)
    p.add_argument("--max_eval_tokens", type=int, default=16_384)
    p.add_argument("--max_features_per_cell", type=int, default=512)
    p.add_argument("--max_circuit_features", type=int, default=100)
    p.add_argument("--topk_values", nargs="+", type=int, default=[1, 5, 10, 25, 50, 100])
    p.add_argument("--ridge_probe_alpha", type=float, default=10.0)
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--sae_batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    for concept_name in args.concepts:
        concept = get_concept(concept_name)
        if concept.num_classes != 2:
            raise ValueError(f"concept-eap currently supports binary concepts only, got {concept_name}")
    return run_concept_eap(args)


if __name__ == "__main__":
    raise SystemExit(main())
