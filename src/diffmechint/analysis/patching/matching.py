"""Cross-tokenizer feature matching (Hungarian assignment) and match plots."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from diffmechint.analysis.alignment import (
    T_CENTERS,
    aligned_positions,
    cell_path,
    fit_ridge_affine,
    read_rows,
)
from diffmechint.sae import load_matryoshka_sae, resolve_sae_ckpt
from diffmechint.utils import info, make_run_dir, ok, warn, write_summary_md

from .common import (
    PB,
    FeatureRow,
    _row_to_feature,
    _sample_tokens_np,
    decoder_weight,
    read_csv,
    selected_cells,
    write_csv,
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
    dec = decoder_weight(sae).float().to(device)
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
                src_pos, tgt_pos, _ = aligned_positions(
                    args.activations_root, source, target, args.dit_step, args.max_images, args.seed
                )
                src_acts = read_rows(cell_path(args.activations_root, source, args.dit_step, layer, t_bin), src_pos)
                tgt_acts = read_rows(cell_path(args.activations_root, target, args.dit_step, layer, t_bin), tgt_pos)
                src_sae = load_matryoshka_sae(
                    resolve_sae_ckpt(args.sae_root, source, layer, t_bin, args.dit_step), device
                )
                tgt_sae = load_matryoshka_sae(
                    resolve_sae_ckpt(args.sae_root, target, layer, t_bin, args.dit_step), device
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
    write_summary_md(
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


def _bank_lookup(bank_index: dict[tuple[str, int, int], list[FeatureRow]]) -> dict[tuple[str, int, int, int], FeatureRow]:
    return {
        (feat.condition, feat.layer, feat.t_bin, feat.feature_id): feat
        for feats in bank_index.values()
        for feat in feats
    }
