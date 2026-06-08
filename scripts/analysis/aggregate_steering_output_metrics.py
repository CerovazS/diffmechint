"""Aggregate Phase 4.18 full output-specific steering metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from diffmechint.utils import read_csv, write_csv

METRICS = {
    "classifier_prob": "classifier_target_probability",
    "clip_text": "clip_target_text_similarity",
    "dino_top": "dino_top_example_similarity",
}
CONTROL_MODES = ("random_matched_control", "wrong_window_control")


def _float(raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def _paired_values(rows: list[dict[str, str]], metric_col: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        out[row["mode"]][int(row["sample_id"])] = _float(row[metric_col])
    return out


def _paired_delta(
    by_mode: dict[str, dict[int, float]],
    mode_a: str,
    mode_b: str,
) -> np.ndarray:
    a = by_mode.get(mode_a, {})
    b = by_mode.get(mode_b, {})
    sample_ids = sorted(set(a) & set(b))
    if not sample_ids:
        return np.empty(0, dtype=np.float64)
    return np.asarray([a[idx] - b[idx] for idx in sample_ids], dtype=np.float64)


def _bootstrap_ci(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1 or n_boot <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, float(lo), float(hi)


def _mode_summary_rows(metrics_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(metrics_root.glob("*/metrics/steering_dino_mode_summary.csv")):
        rows.extend(read_csv(path))
    return rows


def _candidate_metrics(
    candidate_dir: Path,
    manifest_row: dict[str, str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    per_image_path = candidate_dir / "metrics" / "steering_dino_per_image.csv"
    mode_summary_path = candidate_dir / "metrics" / "steering_dino_mode_summary.csv"
    per_image = read_csv(per_image_path)
    mode_rows = {row["mode"]: row for row in read_csv(mode_summary_path)}
    out: dict[str, Any] = {
        "candidate_id": manifest_row["candidate_id"],
        "selection_role": manifest_row["selection_role"],
        "target_top_class_idx": manifest_row["target_top_class_idx"],
        "target_top_label": manifest_row["target_top_label"],
        "n_modes": len(mode_rows),
    }
    transfer = mode_rows.get("transfer_replace", {})
    out["transfer_preservation_median"] = _float(transfer.get("dino_preservation_vs_baseline_median"))
    out["transfer_preservation_mean"] = _float(transfer.get("dino_preservation_vs_baseline_mean"))

    confirmed_flags: list[bool] = []
    for short, column in METRICS.items():
        by_mode = _paired_values(per_image, column)
        base_delta = _paired_delta(by_mode, "transfer_replace", "baseline")
        mean, lo, hi = _bootstrap_ci(base_delta, n_boot=n_boot, seed=seed)
        out[f"{short}_transfer_minus_baseline_mean"] = mean
        out[f"{short}_transfer_minus_baseline_ci_low"] = lo
        out[f"{short}_transfer_minus_baseline_ci_high"] = hi
        metric_ok = lo > 0.0
        for control in CONTROL_MODES:
            delta = _paired_delta(by_mode, "transfer_replace", control)
            c_mean, c_lo, c_hi = _bootstrap_ci(delta, n_boot=n_boot, seed=seed + len(control))
            prefix = f"{short}_transfer_minus_{control}"
            out[f"{prefix}_mean"] = c_mean
            out[f"{prefix}_ci_low"] = c_lo
            out[f"{prefix}_ci_high"] = c_hi
            metric_ok = metric_ok and c_lo > 0.0
        out[f"{short}_confirmed"] = metric_ok and out["transfer_preservation_median"] >= 0.75
        confirmed_flags.append(bool(out[f"{short}_confirmed"]))
    out["strict_all_metric_confirmed"] = all(confirmed_flags)
    out["any_metric_confirmed"] = any(confirmed_flags)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--expected_candidates", type=int, default=18)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    manifest_rows = {row["candidate_id"]: row for row in read_csv(args.manifest)}
    candidate_rows: list[dict[str, Any]] = []
    for candidate_dir in sorted(path for path in args.metrics_root.iterdir() if path.is_dir()):
        candidate_id = candidate_dir.name
        if candidate_id not in manifest_rows:
            continue
        candidate_rows.append(
            _candidate_metrics(candidate_dir, manifest_rows[candidate_id], n_boot=args.n_boot, seed=args.seed)
        )

    mode_rows = _mode_summary_rows(args.metrics_root)
    mode_fields = list(mode_rows[0]) if mode_rows else []
    candidate_fields = [
        "candidate_id",
        "selection_role",
        "target_top_class_idx",
        "target_top_label",
        "n_modes",
        "transfer_preservation_median",
        "transfer_preservation_mean",
    ]
    for short in METRICS:
        candidate_fields.extend(
            [
                f"{short}_transfer_minus_baseline_mean",
                f"{short}_transfer_minus_baseline_ci_low",
                f"{short}_transfer_minus_baseline_ci_high",
                f"{short}_transfer_minus_random_matched_control_mean",
                f"{short}_transfer_minus_random_matched_control_ci_low",
                f"{short}_transfer_minus_random_matched_control_ci_high",
                f"{short}_transfer_minus_wrong_window_control_mean",
                f"{short}_transfer_minus_wrong_window_control_ci_low",
                f"{short}_transfer_minus_wrong_window_control_ci_high",
                f"{short}_confirmed",
            ]
        )
    candidate_fields.extend(["strict_all_metric_confirmed", "any_metric_confirmed"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "full_output_metric_mode_summary.csv", mode_rows, mode_fields)
    write_csv(args.out_dir / "full_output_metric_candidate_summary.csv", candidate_rows, candidate_fields)

    counts_by_role = Counter(row["selection_role"] for row in candidate_rows)
    aggregate = {
        "expected_candidates": args.expected_candidates,
        "completed_candidates": len(candidate_rows),
        "missing_candidates": max(args.expected_candidates - len(candidate_rows), 0),
        "mode_rows": len(mode_rows),
        "counts_by_role": dict(sorted(counts_by_role.items())),
        "positive_strict_confirmed": sum(
            row["selection_role"] == "positive" and row["strict_all_metric_confirmed"] for row in candidate_rows
        ),
        "negative_strict_confirmed": sum(
            row["selection_role"] == "negative_control" and row["strict_all_metric_confirmed"] for row in candidate_rows
        ),
        "positive_any_metric_confirmed": sum(
            row["selection_role"] == "positive" and row["any_metric_confirmed"] for row in candidate_rows
        ),
        "negative_any_metric_confirmed": sum(
            row["selection_role"] == "negative_control" and row["any_metric_confirmed"] for row in candidate_rows
        ),
        "confirmed_candidate_ids": [
            row["candidate_id"] for row in candidate_rows if row["strict_all_metric_confirmed"]
        ],
        "any_metric_candidate_ids": [
            row["candidate_id"] for row in candidate_rows if row["any_metric_confirmed"]
        ],
    }
    (args.out_dir / "full_output_metric_aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))
    if args.require_complete and len(candidate_rows) != args.expected_candidates:
        raise SystemExit(f"expected {args.expected_candidates} candidates, found {len(candidate_rows)}")


if __name__ == "__main__":
    main()
