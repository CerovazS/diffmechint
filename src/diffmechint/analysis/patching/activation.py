"""Single-feature and group-level SAE activation patching."""

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

from diffmechint.analysis.alignment import (
    T_CENTERS,
    aligned_positions,
    alignment_metrics,
    cell_path,
    fit_ridge_affine,
    read_rows,
)
from diffmechint.sae import load_matryoshka_sae, resolve_sae_ckpt
from diffmechint.utils import info, make_run_dir, ok, warn, write_summary_md

from .common import (
    PB,
    FeatureRow,
    MetricAccumulator,
    _parse_optional_float,
    _sample_tokens_np,
    decoder_weight,
    read_csv,
    selected_cells,
    write_csv,
)
from .matching import _bank_lookup, _load_bank_features, _load_match_rows


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
        src_pos, tgt_pos, _ = aligned_positions(
            args.activations_root, source, target, args.dit_step, args.max_images, args.seed
        )
        src_acts = read_rows(cell_path(args.activations_root, source, args.dit_step, layer, t_bin), src_pos)
        tgt_acts = read_rows(cell_path(args.activations_root, target, args.dit_step, layer, t_bin), tgt_pos)
        src_fit, _ = _sample_tokens_np(src_acts, args.max_fit_tokens, args.seed + 101)
        tgt_fit, _ = _sample_tokens_np(tgt_acts, args.max_fit_tokens, args.seed + 101)
        tgt_eval_img = _sample_image_block_np(tgt_acts, args.max_eval_tokens, args.seed + 103)
        tgt_eval = tgt_eval_img.reshape(-1, tgt_eval_img.shape[-1])
        tgt_to_src = fit_ridge_affine(tgt_fit, src_fit, alpha=args.ridge_alpha)

        target_sae = load_matryoshka_sae(resolve_sae_ckpt(args.sae_root, target, layer, t_bin, args.dit_step), device)
        source_sae = load_matryoshka_sae(resolve_sae_ckpt(args.sae_root, source, layer, t_bin, args.dit_step), device)
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
        patched_token_count = 0
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
            patched_token_count += int(mask.sum().item())
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

        reconstruction_metrics = accums["reconstruction_only"].finalize()
        native_decode_metrics = accums["native_reconstruction"].finalize()
        source_all = np.concatenate(coeff_source, axis=0)
        target_all = np.concatenate(coeff_target, axis=0)
        calibrated_all = np.concatenate(coeff_calibrated, axis=0)
        group_corrs = [_corr(calibrated_all[:, j], target_all[:, j]) for j in range(target_all.shape[1])]
        group_r2s = [_r2_np(calibrated_all[:, j], target_all[:, j]) for j in range(target_all.shape[1])]
        threshold_active_frac = float(np.mean(np.max(source_all, axis=1) > active_threshold))
        patched_token_frac = float(patched_token_count / max(target_tensor.shape[0], 1))
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
                    "patch_active_frac": patched_token_frac,
                    "threshold_active_frac": threshold_active_frac,
                    "patched_token_count": patched_token_count,
                    "active_threshold": active_threshold,
                    "n_fit_tokens": int(src_fit.shape[0]),
                    "n_eval_tokens": int(tgt_eval.shape[0]),
                    "mse": metrics["mse"],
                    "cosine": metrics["cosine"],
                    "ev": metrics["ev"],
                    "delta_mse_vs_reconstruction_only": metrics["mse"] - reconstruction_metrics["mse"],
                    "delta_cosine_vs_reconstruction_only": metrics["cosine"] - reconstruction_metrics["cosine"],
                    "delta_ev_vs_reconstruction_only": metrics["ev"] - reconstruction_metrics["ev"],
                    "delta_mse_vs_native": metrics["mse"] - native_decode_metrics["mse"],
                    "delta_cosine_vs_native": metrics["cosine"] - native_decode_metrics["cosine"],
                    "delta_ev_vs_native": metrics["ev"] - native_decode_metrics["ev"],
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
    write_summary_md(
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
        src_pos, tgt_pos, _ = aligned_positions(
            args.activations_root, source, target, args.dit_step, args.max_images, args.seed
        )
        src_acts = read_rows(cell_path(args.activations_root, source, args.dit_step, layer, t_bin), src_pos)
        tgt_acts = read_rows(cell_path(args.activations_root, target, args.dit_step, layer, t_bin), tgt_pos)
        src_fit, _ = _sample_tokens_np(src_acts, args.max_fit_tokens, args.seed + 41)
        tgt_fit, _ = _sample_tokens_np(tgt_acts, args.max_fit_tokens, args.seed + 41)
        tgt_eval_img = _sample_image_block_np(tgt_acts, args.max_eval_tokens, args.seed + 43)
        tgt_eval = tgt_eval_img.reshape(-1, tgt_eval_img.shape[-1])
        tgt_to_src = fit_ridge_affine(tgt_fit, src_fit, alpha=args.ridge_alpha)
        src_to_tgt = fit_ridge_affine(src_fit, tgt_fit, alpha=args.ridge_alpha)
        native_probe = alignment_metrics(tgt_to_src.apply(tgt_fit[: min(2048, tgt_fit.shape[0])]), src_fit[: min(2048, src_fit.shape[0])])

        target_sae = load_matryoshka_sae(resolve_sae_ckpt(args.sae_root, target, layer, t_bin, args.dit_step), device)
        source_sae = load_matryoshka_sae(resolve_sae_ckpt(args.sae_root, source, layer, t_bin, args.dit_step), device)
        sae_dtype = next(target_sae.parameters()).dtype
        tgt_to_src_t = _torch_affine(tgt_to_src.weight, tgt_to_src.bias, device, sae_dtype)
        source_to_target_w = torch.from_numpy(src_to_tgt.weight).to(device=device, dtype=sae_dtype)
        src_decoder_rows = decoder_weight(source_sae).to(device=device, dtype=sae_dtype)

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
    write_summary_md(
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
