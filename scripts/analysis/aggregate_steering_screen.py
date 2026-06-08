"""Aggregate Phase 4.18 full steering screen FID outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from diffmechint.utils import write_csv

CLAMP_MODES = {"native_clamp_q95", "native_clamp_q99", "native_clamp_2x_q99"}
FIRST_ARRAY_SKIP_MODES = CLAMP_MODES


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _task_rows(first_task_file: Path, retry_task_file: Path) -> dict[tuple[str, str], dict[str, str]]:
    expected: dict[tuple[str, str], dict[str, str]] = {}
    for row in _read_tsv(first_task_file):
        mode = row["mode"]
        if mode in FIRST_ARRAY_SKIP_MODES:
            continue
        expected[(row["candidate_id"], mode)] = row
    for row in _read_tsv(retry_task_file):
        mode = row["mode"]
        if mode not in CLAMP_MODES:
            continue
        expected[(row["candidate_id"], mode)] = row
    return expected


def _candidate_id_from_fid_path(path: Path) -> str:
    parts = path.parts
    try:
        sampling_idx = parts.index("sampling")
    except ValueError as exc:
        raise ValueError(f"cannot recover candidate id from {path}") from exc
    return parts[sampling_idx + 1]


def _mode_from_fid(data: dict[str, Any]) -> str:
    mode = str(data.get("mode", ""))
    if mode != "native_clamp":
        return mode
    run_tag = str(data.get("run_tag", ""))
    if "__q0p95__m1p0__" in run_tag:
        return "native_clamp_q95"
    if "__q0p99__m1p0__" in run_tag:
        return "native_clamp_q99"
    if "__q0p99__m2p0__" in run_tag:
        return "native_clamp_2x_q99"
    return "native_clamp_ambiguous"


def _collect_fids(root: Path, source_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fid_path in sorted(root.glob("sampling/**/fid.json")):
        data = json.loads(fid_path.read_text(encoding="utf-8"))
        candidate_id = _candidate_id_from_fid_path(fid_path)
        mode = _mode_from_fid(data)
        if mode == "native_clamp_ambiguous":
            continue
        hook_stats = data.get("hook_stats") or {}
        rows.append(
            {
                "candidate_id": candidate_id,
                "mode": mode,
                "run_tag": data.get("run_tag", ""),
                "fid": float(data["fid"]),
                "n_samples": int(data.get("n_samples", 0) or 0),
                "hook_active": int(hook_stats.get("active", 0) or 0),
                "hook_skipped": int(hook_stats.get("skipped", 0) or 0),
                "hook_no_t": int(hook_stats.get("no_t", 0) or 0),
                "gen_minutes": float(data.get("gen_minutes", 0.0) or 0.0),
                "fid_minutes": float(data.get("fid_minutes", 0.0) or 0.0),
                "path": str(fid_path),
                "source_label": source_label,
            }
        )
    return rows


def _merge_rows(expected: dict[tuple[str, str], dict[str, str]], fid_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates: list[dict[str, str]] = []
    for fid_row in fid_rows:
        key = (fid_row["candidate_id"], fid_row["mode"])
        if key in by_key:
            duplicates.append({"candidate_id": key[0], "mode": key[1], "path": fid_row["path"]})
            continue
        by_key[key] = fid_row

    merged: list[dict[str, Any]] = []
    for key, task in sorted(expected.items()):
        fid = by_key.get(key)
        if fid is None:
            merged.append({**task, "status": "missing"})
            continue
        merged.append(
            {
                **task,
                "status": "completed",
                "run_tag": fid["run_tag"],
                "fid": fid["fid"],
                "hook_active": fid["hook_active"],
                "hook_skipped": fid["hook_skipped"],
                "hook_no_t": fid["hook_no_t"],
                "gen_minutes": fid["gen_minutes"],
                "fid_minutes": fid["fid_minutes"],
                "fid_path": fid["path"],
                "source_label": fid["source_label"],
            }
        )
    return merged, duplicates


def _candidate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_candidate[row["candidate_id"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for candidate_id, group in sorted(by_candidate.items()):
        mode_to_row = {row["mode"]: row for row in group}
        baseline = mode_to_row.get("baseline")
        baseline_fid = float(baseline["fid"]) if baseline else None
        template = group[0]
        out: dict[str, Any] = {
            "candidate_id": candidate_id,
            "selection_role": template.get("selection_role", ""),
            "source": template.get("source", ""),
            "target": template.get("target", ""),
            "layer": template.get("layer", ""),
            "t_bin": template.get("t_bin", ""),
            "source_feature_id": template.get("source_feature_id", ""),
            "target_feature_id": template.get("target_feature_id", ""),
            "target_top_class_idx": template.get("target_top_class_idx", ""),
            "target_top_label": template.get("target_top_label", ""),
            "completed_modes": len(group),
            "baseline_fid": baseline_fid if baseline_fid is not None else "",
        }
        for mode in [
            "native_ablate",
            "native_clamp_q95",
            "native_clamp_q99",
            "native_clamp_2x_q99",
            "transfer_replace",
            "random_matched_control",
            "wrong_window_control",
        ]:
            row = mode_to_row.get(mode)
            fid = float(row["fid"]) if row else None
            out[f"{mode}_fid"] = fid if fid is not None else ""
            if fid is not None and baseline_fid is not None:
                out[f"{mode}_delta_vs_baseline"] = fid - baseline_fid
            else:
                out[f"{mode}_delta_vs_baseline"] = ""
        transfer = out.get("transfer_replace_fid")
        random_control = out.get("random_matched_control_fid")
        wrong_window = out.get("wrong_window_control_fid")
        out["transfer_lower_than_both_controls"] = (
            isinstance(transfer, float)
            and isinstance(random_control, float)
            and isinstance(wrong_window, float)
            and transfer < random_control
            and transfer < wrong_window
        )
        summary_rows.append(out)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first_task_file", type=Path, required=True)
    parser.add_argument("--retry_task_file", type=Path, required=True)
    parser.add_argument("--first_root", type=Path, required=True)
    parser.add_argument("--retry_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    expected = _task_rows(args.first_task_file, args.retry_task_file)
    fid_rows = _collect_fids(args.first_root, "first_array_non_clamp")
    fid_rows.extend(_collect_fids(args.retry_root, "clamp_retry"))
    rows, duplicates = _merge_rows(expected, fid_rows)
    summary_rows = _candidate_summary(rows)

    result_fields = [
        "task_id",
        "candidate_id",
        "selection_role",
        "mode",
        "patch_mode",
        "source",
        "target",
        "layer",
        "t_bin",
        "source_feature_id",
        "target_feature_id",
        "target_top_class_idx",
        "target_top_label",
        "class_schedule",
        "error_mode",
        "n_samples",
        "seed",
        "status",
        "run_tag",
        "fid",
        "hook_active",
        "hook_skipped",
        "hook_no_t",
        "gen_minutes",
        "fid_minutes",
        "source_label",
        "fid_path",
    ]
    candidate_fields = [
        "candidate_id",
        "selection_role",
        "source",
        "target",
        "layer",
        "t_bin",
        "source_feature_id",
        "target_feature_id",
        "target_top_class_idx",
        "target_top_label",
        "completed_modes",
        "baseline_fid",
        "native_ablate_fid",
        "native_ablate_delta_vs_baseline",
        "native_clamp_q95_fid",
        "native_clamp_q95_delta_vs_baseline",
        "native_clamp_q99_fid",
        "native_clamp_q99_delta_vs_baseline",
        "native_clamp_2x_q99_fid",
        "native_clamp_2x_q99_delta_vs_baseline",
        "transfer_replace_fid",
        "transfer_replace_delta_vs_baseline",
        "random_matched_control_fid",
        "random_matched_control_delta_vs_baseline",
        "wrong_window_control_fid",
        "wrong_window_control_delta_vs_baseline",
        "transfer_lower_than_both_controls",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "full_steering_fid_results.csv", rows, result_fields)
    write_csv(args.out_dir / "full_steering_candidate_summary.csv", summary_rows, candidate_fields)

    missing = [
        {"candidate_id": row["candidate_id"], "mode": row["mode"], "task_id": row.get("task_id", "")}
        for row in rows
        if row.get("status") != "completed"
    ]
    counts_by_mode = Counter(row["mode"] for row in rows if row.get("status") == "completed")
    counts_by_role = Counter(row.get("selection_role", "") for row in rows if row.get("status") == "completed")
    aggregate = {
        "expected_rows": len(expected),
        "completed_rows": len(rows) - len(missing),
        "missing_rows": len(missing),
        "duplicates": duplicates,
        "counts_by_mode": dict(sorted(counts_by_mode.items())),
        "counts_by_role": dict(sorted(counts_by_role.items())),
        "positive_transfer_lower_than_both_controls": sum(
            row["selection_role"] == "positive" and row["transfer_lower_than_both_controls"]
            for row in summary_rows
        ),
        "negative_transfer_lower_than_both_controls": sum(
            row["selection_role"] == "negative_control" and row["transfer_lower_than_both_controls"]
            for row in summary_rows
        ),
        "missing": missing,
    }
    (args.out_dir / "full_steering_aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))
    if args.require_complete and missing:
        raise SystemExit(f"missing {len(missing)} expected rows")


if __name__ == "__main__":
    main()
