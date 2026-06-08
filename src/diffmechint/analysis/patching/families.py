"""Feature-family grouping and group-patching aggregation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diffmechint.utils import make_run_dir, ok, write_summary_md

from .common import (
    PB,
    _feature_key,
    _finite_mean,
    _finite_median,
    _nan_float,
    read_csv,
    write_csv,
)


def _load_group_patch_result_rows(args: argparse.Namespace) -> list[dict]:
    paths: list[Path] = []
    for run_dir in args.run_dirs:
        paths.append(run_dir / "metrics" / "group_activation_patching.csv")
    paths.extend(args.patch_csvs)
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"group activation CSV missing: {path}")
        for row in read_csv(path):
            row["source_patch_csv"] = str(path)
            row["source_run_dir"] = str(path.parents[1]) if len(path.parents) >= 2 else str(path.parent)
            rows.append(row)
    return rows


def _group_task_key(row: dict) -> tuple[str, str, str, int, int]:
    return (
        str(row["group_id"]),
        str(row["source"]),
        str(row["target"]),
        int(row["layer"]),
        int(row["t_bin"]),
    )


def _metric_delta(mode_rows: dict[str, dict], mode: str, baseline: str, field: str) -> float:
    if mode not in mode_rows or baseline not in mode_rows:
        return float("nan")
    return _nan_float(mode_rows[mode].get(field)) - _nan_float(mode_rows[baseline].get(field))


def _group_task_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[_group_task_key(row)][str(row["mode"])] = row

    summary_rows = []
    for key, mode_rows in sorted(grouped.items()):
        group_id, source, target, layer, t_bin = key
        first = mode_rows.get("group_transfer_replace") or next(iter(mode_rows.values()))
        transfer_delta_ev = _metric_delta(mode_rows, "group_transfer_replace", "reconstruction_only", "ev")
        random_delta_ev = _metric_delta(mode_rows, "random_group_control", "reconstruction_only", "ev")
        shuffled_delta_ev = _metric_delta(mode_rows, "shuffled_group_control", "reconstruction_only", "ev")
        ablate_delta_ev = _metric_delta(mode_rows, "group_native_ablate", "reconstruction_only", "ev")
        clamp_delta_ev = _metric_delta(mode_rows, "group_native_clamp", "reconstruction_only", "ev")
        transfer_minus_random = transfer_delta_ev - random_delta_ev
        transfer_minus_shuffled = transfer_delta_ev - shuffled_delta_ev
        patch_active_frac = _nan_float(first.get("patch_active_frac"))
        threshold_active_frac = _nan_float(first.get("threshold_active_frac"))
        status = "ok"
        if "group_transfer_replace" not in mode_rows:
            status = "missing_transfer"
        elif not np.isfinite(patch_active_frac) or patch_active_frac <= 0.0:
            status = "no_active_tokens"
        elif np.isfinite(transfer_minus_random) and np.isfinite(transfer_minus_shuffled):
            if transfer_minus_random < 0.0 and transfer_minus_shuffled < 0.0:
                status = "transfer_more_disruptive_than_random_and_shuffled"
            elif abs(transfer_minus_shuffled) <= 1e-8:
                status = "transfer_matches_shuffled_pairing"
            elif transfer_minus_random < 0.0:
                status = "transfer_more_disruptive_than_random_only"
            elif transfer_minus_shuffled < 0.0:
                status = "transfer_more_disruptive_than_shuffled_only"
            else:
                status = "transfer_not_stronger_than_controls"
        summary_rows.append(
            {
                "task_id": f"{group_id}:{source}->{target}:L{layer}:T{t_bin}",
                "group_id": group_id,
                "family_label": first.get("family_label", ""),
                "source": source,
                "target": target,
                "layer": layer,
                "t_bin": t_bin,
                "t_center": first.get("t_center", ""),
                "source_feature_ids": first.get("source_feature_ids", ""),
                "target_feature_ids": first.get("target_feature_ids", ""),
                "n_source_features": first.get("n_source_features", ""),
                "n_target_features": first.get("n_target_features", ""),
                "group_calibrated_corr": _nan_float(first.get("group_calibrated_corr")),
                "group_calibrated_r2": _nan_float(first.get("group_calibrated_r2")),
                "calibration_corr": _nan_float(first.get("calibration_corr")),
                "calibration_r2": _nan_float(first.get("calibration_r2")),
                "patch_active_frac": patch_active_frac,
                "threshold_active_frac": threshold_active_frac,
                "patched_token_count": first.get("patched_token_count", ""),
                "n_fit_tokens": first.get("n_fit_tokens", ""),
                "n_eval_tokens": first.get("n_eval_tokens", ""),
                "native_reconstruction_delta_ev": _metric_delta(
                    mode_rows, "native_reconstruction", "reconstruction_only", "ev"
                ),
                "transfer_delta_ev": transfer_delta_ev,
                "random_control_delta_ev": random_delta_ev,
                "shuffled_control_delta_ev": shuffled_delta_ev,
                "native_ablate_delta_ev": ablate_delta_ev,
                "native_clamp_delta_ev": clamp_delta_ev,
                "transfer_minus_random_delta_ev": transfer_minus_random,
                "transfer_minus_shuffled_delta_ev": transfer_minus_shuffled,
                "transfer_delta_mse": _metric_delta(
                    mode_rows, "group_transfer_replace", "reconstruction_only", "mse"
                ),
                "random_control_delta_mse": _metric_delta(
                    mode_rows, "random_group_control", "reconstruction_only", "mse"
                ),
                "shuffled_control_delta_mse": _metric_delta(
                    mode_rows, "shuffled_group_control", "reconstruction_only", "mse"
                ),
                "n_modes_present": len(mode_rows),
                "present_modes": sorted(mode_rows),
                "status": status,
            }
        )
    return summary_rows


def _group_pair_cell_summary(task_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for row in task_rows:
        grouped[(str(row["source"]), str(row["target"]), int(row["layer"]), int(row["t_bin"]))].append(row)
    summary_rows = []
    for (source, target, layer, t_bin), vals in sorted(grouped.items()):
        summary_rows.append(
            {
                "source": source,
                "target": target,
                "layer": layer,
                "t_bin": t_bin,
                "n_tasks": len(vals),
                "n_groups": len({str(row["group_id"]) for row in vals}),
                "mean_group_calibrated_corr": _finite_mean(
                    [_nan_float(row["group_calibrated_corr"]) for row in vals]
                ),
                "mean_group_calibrated_r2": _finite_mean([_nan_float(row["group_calibrated_r2"]) for row in vals]),
                "mean_patch_active_frac": _finite_mean([_nan_float(row["patch_active_frac"]) for row in vals]),
                "mean_threshold_active_frac": _finite_mean(
                    [_nan_float(row["threshold_active_frac"]) for row in vals]
                ),
                "mean_transfer_delta_ev": _finite_mean([_nan_float(row["transfer_delta_ev"]) for row in vals]),
                "median_transfer_delta_ev": _finite_median([_nan_float(row["transfer_delta_ev"]) for row in vals]),
                "mean_random_control_delta_ev": _finite_mean(
                    [_nan_float(row["random_control_delta_ev"]) for row in vals]
                ),
                "mean_shuffled_control_delta_ev": _finite_mean(
                    [_nan_float(row["shuffled_control_delta_ev"]) for row in vals]
                ),
                "mean_native_ablate_delta_ev": _finite_mean(
                    [_nan_float(row["native_ablate_delta_ev"]) for row in vals]
                ),
                "mean_native_clamp_delta_ev": _finite_mean([_nan_float(row["native_clamp_delta_ev"]) for row in vals]),
                "mean_transfer_minus_random_delta_ev": _finite_mean(
                    [_nan_float(row["transfer_minus_random_delta_ev"]) for row in vals]
                ),
                "mean_transfer_minus_shuffled_delta_ev": _finite_mean(
                    [_nan_float(row["transfer_minus_shuffled_delta_ev"]) for row in vals]
                ),
                "n_transfer_more_disruptive_than_random": sum(
                    _nan_float(row["transfer_minus_random_delta_ev"]) < 0.0 for row in vals
                ),
                "n_transfer_more_disruptive_than_shuffled": sum(
                    _nan_float(row["transfer_minus_shuffled_delta_ev"]) < 0.0 for row in vals
                ),
                "status_counts": dict(Counter(str(row["status"]) for row in vals)),
            }
        )
    return summary_rows


def _group_pair_summary(task_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in task_rows:
        grouped[(str(row["source"]), str(row["target"]))].append(row)
    summary_rows = []
    for (source, target), vals in sorted(grouped.items()):
        summary_rows.append(
            {
                "source": source,
                "target": target,
                "n_tasks": len(vals),
                "n_groups": len({str(row["group_id"]) for row in vals}),
                "mean_group_calibrated_corr": _finite_mean(
                    [_nan_float(row["group_calibrated_corr"]) for row in vals]
                ),
                "mean_group_calibrated_r2": _finite_mean([_nan_float(row["group_calibrated_r2"]) for row in vals]),
                "mean_patch_active_frac": _finite_mean([_nan_float(row["patch_active_frac"]) for row in vals]),
                "mean_threshold_active_frac": _finite_mean(
                    [_nan_float(row["threshold_active_frac"]) for row in vals]
                ),
                "mean_transfer_delta_ev": _finite_mean([_nan_float(row["transfer_delta_ev"]) for row in vals]),
                "mean_random_control_delta_ev": _finite_mean(
                    [_nan_float(row["random_control_delta_ev"]) for row in vals]
                ),
                "mean_shuffled_control_delta_ev": _finite_mean(
                    [_nan_float(row["shuffled_control_delta_ev"]) for row in vals]
                ),
                "mean_transfer_minus_random_delta_ev": _finite_mean(
                    [_nan_float(row["transfer_minus_random_delta_ev"]) for row in vals]
                ),
                "mean_transfer_minus_shuffled_delta_ev": _finite_mean(
                    [_nan_float(row["transfer_minus_shuffled_delta_ev"]) for row in vals]
                ),
                "status_counts": dict(Counter(str(row["status"]) for row in vals)),
            }
        )
    return summary_rows


def _plot_group_transfer_bars(pair_rows: list[dict], out_path: Path) -> None:
    if not pair_rows:
        return
    labels = [f"{row['source']}->{row['target']}" for row in pair_rows]
    x = np.arange(len(pair_rows), dtype=np.float32)
    fields = [
        ("mean_transfer_delta_ev", "transfer", PB["blue"]),
        ("mean_random_control_delta_ev", "random", PB["gold"]),
        ("mean_shuffled_control_delta_ev", "shuffled", PB["red"]),
    ]
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8.0, 0.8 * len(pair_rows)), 4.4))
    ax.axhline(0.0, color=PB["dark"], linewidth=1.0)
    for i, (field, label, color) in enumerate(fields):
        vals = [_nan_float(row[field]) for row in pair_rows]
        ax.bar(x + (i - 1) * width, vals, width=width, color=color, label=label)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("mean delta EV vs reconstruction-only")
    ax.set_title("Group patch residual effect by directed pair")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_group_calibration_heatmap(pair_cell_rows: list[dict], out_path: Path) -> None:
    if not pair_cell_rows:
        return
    pairs = sorted({(str(row["source"]), str(row["target"])) for row in pair_cell_rows})
    cells = sorted({(int(row["layer"]), int(row["t_bin"])) for row in pair_cell_rows})
    arr = np.full((len(pairs), len(cells)), np.nan, dtype=np.float32)
    for i, pair in enumerate(pairs):
        for j, cell in enumerate(cells):
            vals = [
                _nan_float(row["mean_group_calibrated_corr"])
                for row in pair_cell_rows
                if (str(row["source"]), str(row["target"])) == pair
                and (int(row["layer"]), int(row["t_bin"])) == cell
            ]
            arr[i, j] = _finite_mean(vals)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.8 * len(cells)), max(3.8, 0.45 * len(pairs))))
    im = ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cells)), [f"L{layer}/T{t_bin}" for layer, t_bin in cells], rotation=35, ha="right")
    ax.set_yticks(range(len(pairs)), [f"{source}->{target}" for source, target in pairs])
    ax.set_title("Mean group coefficient calibration correlation")
    fig.colorbar(im, ax=ax, label="corr")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_group_transfer_vs_controls(task_rows: list[dict], out_path: Path) -> None:
    vals = [
        (
            _nan_float(row["random_control_delta_ev"]),
            _nan_float(row["shuffled_control_delta_ev"]),
            _nan_float(row["transfer_delta_ev"]),
        )
        for row in task_rows
        if np.isfinite(_nan_float(row["transfer_delta_ev"]))
    ]
    vals = [row for row in vals if np.isfinite(row[0]) and np.isfinite(row[1]) and np.isfinite(row[2])]
    if not vals:
        return
    random_vals = np.asarray([row[0] for row in vals], dtype=np.float64)
    shuffled_vals = np.asarray([row[1] for row in vals], dtype=np.float64)
    transfer_vals = np.asarray([row[2] for row in vals], dtype=np.float64)
    lo = float(np.nanmin([random_vals.min(), shuffled_vals.min(), transfer_vals.min()]))
    hi = float(np.nanmax([random_vals.max(), shuffled_vals.max(), transfer_vals.max()]))
    pad = max((hi - lo) * 0.08, 1e-8)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=PB["dark"], linewidth=1.0)
    ax.scatter(random_vals, transfer_vals, s=28, color=PB["blue"], alpha=0.75, label="random")
    ax.scatter(shuffled_vals, transfer_vals, s=28, color=PB["red"], alpha=0.75, label="shuffled")
    ax.set_xlabel("control delta EV")
    ax.set_ylabel("transfer delta EV")
    ax.set_title("Transfer effect versus controls")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _group_summary_payload(rows: list[dict], task_rows: list[dict], pair_rows: list[dict]) -> dict:
    transfer_minus_random = [_nan_float(row["transfer_minus_random_delta_ev"]) for row in task_rows]
    transfer_minus_shuffled = [_nan_float(row["transfer_minus_shuffled_delta_ev"]) for row in task_rows]
    top_abs = sorted(
        task_rows,
        key=lambda row: abs(_nan_float(row["transfer_delta_ev"]))
        if np.isfinite(_nan_float(row["transfer_delta_ev"]))
        else -1.0,
        reverse=True,
    )[:10]
    return {
        "analysis": "group-activation-patching-aggregate",
        "n_raw_rows": len(rows),
        "n_group_tasks": len(task_rows),
        "n_directed_pairs": len(pair_rows),
        "n_unique_groups": len({str(row["group_id"]) for row in task_rows}),
        "mean_group_calibrated_corr": _finite_mean([_nan_float(row["group_calibrated_corr"]) for row in task_rows]),
        "mean_group_calibrated_r2": _finite_mean([_nan_float(row["group_calibrated_r2"]) for row in task_rows]),
        "mean_patch_active_frac": _finite_mean([_nan_float(row["patch_active_frac"]) for row in task_rows]),
        "mean_threshold_active_frac": _finite_mean([_nan_float(row["threshold_active_frac"]) for row in task_rows]),
        "mean_transfer_delta_ev": _finite_mean([_nan_float(row["transfer_delta_ev"]) for row in task_rows]),
        "mean_random_control_delta_ev": _finite_mean([_nan_float(row["random_control_delta_ev"]) for row in task_rows]),
        "mean_shuffled_control_delta_ev": _finite_mean(
            [_nan_float(row["shuffled_control_delta_ev"]) for row in task_rows]
        ),
        "mean_transfer_minus_random_delta_ev": _finite_mean(transfer_minus_random),
        "mean_transfer_minus_shuffled_delta_ev": _finite_mean(transfer_minus_shuffled),
        "n_transfer_more_disruptive_than_random": sum(value < 0.0 for value in transfer_minus_random),
        "n_transfer_more_disruptive_than_shuffled": sum(value < 0.0 for value in transfer_minus_shuffled),
        "status_counts": dict(Counter(str(row["status"]) for row in task_rows)),
        "top_abs_transfer_delta_ev": [
            {
                "task_id": row["task_id"],
                "family_label": row["family_label"],
                "source": row["source"],
                "target": row["target"],
                "layer": row["layer"],
                "t_bin": row["t_bin"],
                "transfer_delta_ev": row["transfer_delta_ev"],
                "transfer_minus_random_delta_ev": row["transfer_minus_random_delta_ev"],
                "transfer_minus_shuffled_delta_ev": row["transfer_minus_shuffled_delta_ev"],
                "group_calibrated_corr": row["group_calibrated_corr"],
            }
            for row in top_abs
        ],
    }


def _write_group_summary_md(out_dir: Path, payload: dict) -> None:
    lines = [
        f"**Group-level activation patching** aggregated =={payload['n_group_tasks']}== directed feature-family tasks.",
        (
            f"Mean source-to-target coefficient correlation is =={payload['mean_group_calibrated_corr']:.3f}== "
            f"with mean R2 =={payload['mean_group_calibrated_r2']:.3f}==."
        ),
        (
            f"Mean transfer delta EV is =={payload['mean_transfer_delta_ev']:.2e}==; "
            f"transfer minus shuffled is =={payload['mean_transfer_minus_shuffled_delta_ev']:.2e}==."
        ),
        "This is activation-space residual evidence; concept-margin/EAP and sampling-time validation are still required.",
    ]
    write_summary_md(out_dir, "Group-Level Activation Patching Aggregate", lines)


def run_summarize_group_patching(args: argparse.Namespace) -> int:
    out_dir = make_run_dir(args.out_root, args.run_id, resume=args.resume)
    rows = _load_group_patch_result_rows(args)
    if not rows:
        raise ValueError("no group activation patching rows selected")
    task_rows = _group_task_summary(rows)
    pair_cell_rows = _group_pair_cell_summary(task_rows)
    pair_rows = _group_pair_summary(task_rows)
    write_csv(out_dir / "metrics" / "group_activation_patching_all.csv", rows)
    write_csv(out_dir / "metrics" / "group_activation_task_summary.csv", task_rows)
    write_csv(out_dir / "metrics" / "group_activation_pair_cell_summary.csv", pair_cell_rows)
    write_csv(out_dir / "metrics" / "group_activation_pair_summary.csv", pair_rows)
    _plot_group_transfer_bars(pair_rows, out_dir / "plots" / "group_transfer_delta_ev_by_pair.png")
    _plot_group_calibration_heatmap(pair_cell_rows, out_dir / "plots" / "group_calibration_corr_heatmap.png")
    _plot_group_transfer_vs_controls(task_rows, out_dir / "plots" / "group_transfer_vs_controls.png")
    payload = _group_summary_payload(rows, task_rows, pair_rows)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "metrics" / "group_activation_aggregate_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_group_summary_md(out_dir, payload)
    ok(f"group activation aggregate summary complete: {out_dir}")
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
    write_summary_md(
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
