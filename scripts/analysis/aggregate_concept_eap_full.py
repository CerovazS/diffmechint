"""Aggregate Phase 4.18 full concept-EAP array outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from diffmechint.utils import read_csv, write_csv


def _run_scope(summary_path: Path) -> set[tuple[str, int, int, str]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    scopes: set[tuple[str, int, int, str]] = set()
    for condition in data["conditions"]:
        for cell in data["cells"]:
            layer_raw, t_raw = cell.replace("L", "").replace("T", "").split("_", 1)
            for concept in data["concepts"]:
                scopes.add((str(condition), int(layer_raw), int(t_raw), str(concept)))
    return scopes


def _collect_rank_rows(eap_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    rank_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    completed_runs: list[Path] = []
    for summary_path in sorted(eap_root.glob("*/summary.json")):
        run_dir = summary_path.parent
        completed_runs.append(run_dir)
        scope = _run_scope(summary_path)
        ranks_path = run_dir / "metrics" / "eap_candidate_ranks.csv"
        if ranks_path.exists():
            for row in read_csv(ranks_path):
                key = (row["condition"], int(row["layer"]), int(row["t_bin"]), row["concept"])
                if key in scope:
                    row = {**row, "run_id": run_dir.name, "run_dir": str(run_dir)}
                    rank_rows.append(row)
        error_path = run_dir / "metrics" / "eap_error_node_share.csv"
        if error_path.exists():
            for row in read_csv(error_path):
                row = {**row, "run_id": run_dir.name, "run_dir": str(run_dir)}
                error_rows.append(row)
    return rank_rows, error_rows, completed_runs


def _bool(raw: Any) -> bool:
    return str(raw).lower() == "true"


def _summary_rows(rank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rank_rows:
        grouped[(row["candidate_id"], row["selection_role"])].append(row)

    out: list[dict[str, Any]] = []
    for (candidate_id, selection_role), rows in sorted(grouped.items()):
        source = next((row for row in rows if row["candidate_side"] == "source"), None)
        target = next((row for row in rows if row["candidate_side"] == "target"), None)
        source_found = bool(source and _bool(source.get("found_in_eap")))
        target_found = bool(target and _bool(target.get("found_in_eap")))
        row_out: dict[str, Any] = {
            "candidate_id": candidate_id,
            "selection_role": selection_role,
            "n_sides": len(rows),
            "source_found": source_found,
            "target_found": target_found,
            "both_found": source_found and target_found,
            "source_rank_abs": source.get("rank_abs", "") if source else "",
            "target_rank_abs": target.get("rank_abs", "") if target else "",
        }
        for cutoff in [1, 5, 10, 25, 50, 100]:
            row_out[f"source_in_top{cutoff}"] = bool(source and _bool(source.get(f"in_top{cutoff}")))
            row_out[f"target_in_top{cutoff}"] = bool(target and _bool(target.get(f"in_top{cutoff}")))
            row_out[f"both_in_top{cutoff}"] = row_out[f"source_in_top{cutoff}"] and row_out[f"target_in_top{cutoff}"]
        out.append(row_out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap_root", type=Path, required=True)
    parser.add_argument("--expected_runs", type=int, default=10)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    rank_rows, error_rows, completed_runs = _collect_rank_rows(args.eap_root)
    candidate_rows = _summary_rows(rank_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rank_fields = [
        "run_id",
        "candidate_id",
        "selection_role",
        "candidate_side",
        "condition",
        "layer",
        "t_bin",
        "concept",
        "feature_id",
        "found_in_eap",
        "rank_abs",
        "in_top1",
        "in_top5",
        "in_top10",
        "in_top25",
        "in_top50",
        "in_top100",
        "signed_attribution",
        "absolute_attribution",
        "top_label",
        "vlm_interpretation",
        "run_dir",
    ]
    error_fields = [
        "run_id",
        "condition",
        "dit_step",
        "layer",
        "t_bin",
        "concept",
        "clean_margin_gap",
        "sae_reconstruction_margin_gap",
        "error_node_abs_gap_fraction",
        "run_dir",
    ]
    candidate_fields = [
        "candidate_id",
        "selection_role",
        "n_sides",
        "source_found",
        "target_found",
        "both_found",
        "source_rank_abs",
        "target_rank_abs",
        "source_in_top1",
        "target_in_top1",
        "both_in_top1",
        "source_in_top5",
        "target_in_top5",
        "both_in_top5",
        "source_in_top10",
        "target_in_top10",
        "both_in_top10",
        "source_in_top25",
        "target_in_top25",
        "both_in_top25",
        "source_in_top50",
        "target_in_top50",
        "both_in_top50",
        "source_in_top100",
        "target_in_top100",
        "both_in_top100",
    ]
    write_csv(args.out_dir / "full_eap_candidate_ranks.csv", rank_rows, rank_fields)
    write_csv(args.out_dir / "full_eap_error_node_share.csv", error_rows, error_fields)
    write_csv(args.out_dir / "full_eap_candidate_summary.csv", candidate_rows, candidate_fields)

    counts_by_role = Counter(row["selection_role"] for row in candidate_rows)
    aggregate = {
        "expected_runs": args.expected_runs,
        "completed_runs": len(completed_runs),
        "missing_runs": max(args.expected_runs - len(completed_runs), 0),
        "rank_rows": len(rank_rows),
        "error_rows": len(error_rows),
        "candidate_rows": len(candidate_rows),
        "counts_by_role": dict(sorted(counts_by_role.items())),
        "positive_both_in_top10": sum(
            row["selection_role"] == "positive" and row["both_in_top10"] for row in candidate_rows
        ),
        "negative_both_in_top10": sum(
            row["selection_role"] == "negative_control" and row["both_in_top10"] for row in candidate_rows
        ),
        "positive_both_in_top25": sum(
            row["selection_role"] == "positive" and row["both_in_top25"] for row in candidate_rows
        ),
        "negative_both_in_top25": sum(
            row["selection_role"] == "negative_control" and row["both_in_top25"] for row in candidate_rows
        ),
        "completed_run_ids": [path.name for path in completed_runs],
    }
    (args.out_dir / "full_eap_aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))
    if args.require_complete and len(completed_runs) != args.expected_runs:
        raise SystemExit(f"expected {args.expected_runs} runs, found {len(completed_runs)}")


if __name__ == "__main__":
    main()
