"""Build Phase 4.18 steering candidate manifests from E28 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from diffmechint.utils import ok, warn

POSITIVE_STATUS = "transfer_lower_fid_than_random_and_wrong_window"
NEGATIVE_STATUS = "no_transfer_specific_fid_advantage"

FEATURE_FIELDS = (
    "density",
    "density_count",
    "entropy",
    "unique_classes",
    "mean_act",
    "top_activation",
    "top_label",
    "top_class_idx",
    "top_synset",
    "vlm_interpretation",
    "top9_class_idx",
    "top9_dataset_idx",
    "top9_activation",
    "top9_token_pos",
    "n_top_examples",
    "decoder_norm",
)

TASK_FIELDS = (
    "task_id",
    "mode",
    "random_source_feature_id",
    "wrong_t_bin",
    "scale",
    "bias",
    "target_run",
    "target_adapter",
    "normalize",
    "sae_root",
    "selection_rank",
    "calibrated_feature_corr",
    "transfer_minus_random_delta_ev",
    "transfer_minus_shuffled_delta_ev",
)


class ManifestBuildError(RuntimeError):
    """Raised when E28 materials are inconsistent with the Phase 4.18 plan."""


def read_delimited(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ManifestBuildError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_int(raw: object) -> int:
    try:
        return int(str(raw))
    except (TypeError, ValueError) as exc:
        raise ManifestBuildError(f"expected integer, got {raw!r}") from exc


def parse_float(raw: object, default: float = math.nan) -> float:
    try:
        return float(str(raw))
    except (TypeError, ValueError):
        return default


def feature_key(row: dict[str, str], *, role: str | None = None) -> tuple[str, int, int, int]:
    condition = row[f"{role}_condition"] if role else row["condition"]
    layer = parse_int(row["layer"])
    t_bin = parse_int(row["t_bin"])
    feature_id = parse_int(row[f"{role}_feature_id"] if role else row["feature_id"])
    return condition, layer, t_bin, feature_id


def candidate_feature_key(row: dict[str, str], role: str) -> tuple[str, int, int, int]:
    condition = str(row["source"] if role == "source" else row["target"])
    return condition, parse_int(row["layer"]), parse_int(row["t_bin"]), parse_int(row[f"{role}_feature_id"])


def task_key(row: dict[str, str]) -> tuple[str, str]:
    return str(row["candidate_id"]), str(row["mode"])


def prefixed(prefix: str, row: dict[str, str], fields: tuple[str, ...]) -> dict[str, str]:
    return {f"{prefix}_{field}": row.get(field, "") for field in fields}


def e28_metrics(row: dict[str, str]) -> dict[str, str]:
    skip = {
        "candidate_id",
        "selection_role",
        "source",
        "target",
        "layer",
        "t_bin",
        "source_feature_id",
        "target_feature_id",
        "screen_status",
    }
    return {f"e28_{key}": value for key, value in row.items() if key not in skip}


def row_density_norm(
    row: dict[str, str],
    bank_by_key: dict[tuple[str, int, int, int], dict[str, str]],
) -> tuple[float, float]:
    target_meta = bank_by_key.get(candidate_feature_key(row, "target"), {})
    density = parse_float(target_meta.get("density"), default=0.0)
    norm = parse_float(target_meta.get("decoder_norm"), default=0.0)
    return max(density, 1e-12), max(norm, 1e-12)


def target_label(
    row: dict[str, str],
    bank_by_key: dict[tuple[str, int, int, int], dict[str, str]],
) -> str:
    meta = bank_by_key.get(candidate_feature_key(row, "target"), {})
    return str(meta.get("top_label") or row.get("target_top_label") or "")


def target_class_idx(
    row: dict[str, str],
    bank_by_key: dict[tuple[str, int, int, int], dict[str, str]],
) -> str:
    meta = bank_by_key.get(candidate_feature_key(row, "target"), {})
    return str(meta.get("top_class_idx") or "")


def negative_match_score(
    positive: dict[str, str],
    negative: dict[str, str],
    bank_by_key: dict[tuple[str, int, int, int], dict[str, str]],
) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []
    for field, penalty in (("target", 1000.0), ("layer", 100.0), ("t_bin", 100.0)):
        if str(positive[field]) == str(negative[field]):
            reasons.append(f"same_{field}")
        else:
            score += penalty
            reasons.append(f"diff_{field}")
    pos_label = target_label(positive, bank_by_key)
    neg_label = target_label(negative, bank_by_key)
    if pos_label and neg_label and pos_label == neg_label:
        reasons.append("same_target_label")
    else:
        score += 10.0
        reasons.append("diff_target_label")
    pos_density, pos_norm = row_density_norm(positive, bank_by_key)
    neg_density, neg_norm = row_density_norm(negative, bank_by_key)
    score += abs(math.log10(pos_density) - math.log10(neg_density))
    score += 0.1 * abs(math.log10(pos_norm) - math.log10(neg_norm))
    return score, ",".join(reasons)


def select_negative_controls(
    positives: list[dict[str, str]],
    negative_pool: list[dict[str, str]],
    bank_by_key: dict[tuple[str, int, int, int], dict[str, str]],
    n_controls: int,
) -> list[tuple[dict[str, str], dict[str, str], float, str]]:
    unused = {row["candidate_id"]: row for row in negative_pool}
    selected: list[tuple[dict[str, str], dict[str, str], float, str]] = []
    for positive in positives:
        if not unused:
            break
        scored = []
        for negative in unused.values():
            score, reason = negative_match_score(positive, negative, bank_by_key)
            scored.append((score, str(negative["candidate_id"]), negative, reason))
        score, _, chosen, reason = min(scored, key=lambda item: (item[0], item[1]))
        selected.append((positive, chosen, score, reason))
        unused.pop(str(chosen["candidate_id"]))
        if len(selected) == n_controls:
            break
    if len(selected) != n_controls:
        raise ManifestBuildError(f"expected {n_controls} negative controls, found {len(selected)}")
    return selected


def build_manifest_row(
    row: dict[str, str],
    *,
    role: str,
    bank_by_key: dict[tuple[str, int, int, int], dict[str, str]],
    task_by_key: dict[tuple[str, str], dict[str, str]],
    matched_positive_id: str = "",
    control_match_score: float | None = None,
    control_match_reason: str = "",
) -> dict[str, object]:
    source_key = candidate_feature_key(row, "source")
    target_key = candidate_feature_key(row, "target")
    source_meta = bank_by_key.get(source_key)
    target_meta = bank_by_key.get(target_key)
    missing = []
    if source_meta is None:
        missing.append(f"source {source_key}")
    if target_meta is None:
        missing.append(f"target {target_key}")
    if missing:
        raise ManifestBuildError(f"{row['candidate_id']} lacks feature-bank metadata: {missing}")

    task = task_by_key.get((str(row["candidate_id"]), "transfer_replace"), {})
    out: dict[str, object] = {
        "candidate_id": row["candidate_id"],
        "selection_role": role,
        "matched_positive_id": matched_positive_id,
        "control_match_score": "" if control_match_score is None else f"{control_match_score:.6f}",
        "control_match_reason": control_match_reason,
        "screen_status": row["screen_status"],
        "source": row["source"],
        "target": row["target"],
        "layer": parse_int(row["layer"]),
        "t_bin": parse_int(row["t_bin"]),
        "source_feature_id": parse_int(row["source_feature_id"]),
        "target_feature_id": parse_int(row["target_feature_id"]),
        "source_top_class_idx": source_meta["top_class_idx"],
        "source_top_label": source_meta["top_label"],
        "target_top_class_idx": target_meta["top_class_idx"],
        "target_top_label": target_meta["top_label"],
        "source_vlm_interpretation": source_meta.get("vlm_interpretation", ""),
        "target_vlm_interpretation": target_meta.get("vlm_interpretation", ""),
        "sampling_task_found": bool(task),
    }
    out.update(e28_metrics(row))
    out.update(prefixed("source", source_meta, FEATURE_FIELDS))
    out.update(prefixed("target", target_meta, FEATURE_FIELDS))
    out.update(prefixed("sampling_task", task, TASK_FIELDS))
    return out


def write_materials_manifest(
    path: Path,
    *,
    candidate_summary: Path,
    feature_bank: Path,
    sampling_tasks: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Materials Manifest",
        "",
        "## E28 Inputs",
        "",
        f"- Candidate summary: `{candidate_summary}`",
        f"- Feature bank: `{feature_bank}`",
        f"- Sampling task TSV: `{sampling_tasks}`",
        "",
        "## Resolved Counts",
        "",
        f"- Candidate rows: `{summary['n_candidate_rows']}`",
        f"- Feature-bank rows: `{summary['n_feature_bank_rows']}`",
        f"- Sampling-task rows: `{summary['n_sampling_task_rows']}`",
        f"- Positive candidates: `{summary['n_positive_candidates']}`",
        f"- Negative controls: `{summary['n_negative_controls']}`",
        "",
        "## Screen Status Counts",
        "",
    ]
    for status, count in dict(summary["screen_status_counts"]).items():
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `metrics/candidates_manifest.csv`",
            "- `metrics/negative_controls_manifest.csv`",
            "- `metrics/candidate_manifest_summary.json`",
            "- `candidates_manifest.csv`",
            "",
            "## Negative-Control Matching",
            "",
        ]
    )
    for match in summary["negative_control_matches"]:
        lines.append(
            "- "
            f"`{match['negative_candidate_id']}` matched to "
            f"`{match['positive_candidate_id']}` "
            f"(score `{match['score']}`, {match['reason']})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifests(
    *,
    candidate_summary: Path,
    feature_bank: Path,
    sampling_tasks: Path,
    out_dir: Path,
    n_negative_controls: int = 9,
    expected_positive_count: int = 9,
) -> dict[str, object]:
    for path in (candidate_summary, feature_bank, sampling_tasks):
        if not path.exists():
            raise ManifestBuildError(f"missing required input: {path}")

    candidate_rows = read_delimited(candidate_summary)
    feature_rows = read_delimited(feature_bank)
    task_rows = read_delimited(sampling_tasks, delimiter="\t")
    bank_by_key = {feature_key(row): row for row in feature_rows}
    task_by_key = {task_key(row): row for row in task_rows}

    positives = [row for row in candidate_rows if row.get("screen_status") == POSITIVE_STATUS]
    if len(positives) != expected_positive_count:
        raise ManifestBuildError(
            f"expected {expected_positive_count} positive candidates, found {len(positives)}"
        )
    negative_pool = [row for row in candidate_rows if row.get("screen_status") == NEGATIVE_STATUS]
    matches = select_negative_controls(positives, negative_pool, bank_by_key, n_negative_controls)

    positive_manifest = [
        build_manifest_row(row, role="positive", bank_by_key=bank_by_key, task_by_key=task_by_key)
        for row in positives
    ]
    negative_manifest = [
        build_manifest_row(
            negative,
            role="negative_control",
            bank_by_key=bank_by_key,
            task_by_key=task_by_key,
            matched_positive_id=str(positive["candidate_id"]),
            control_match_score=score,
            control_match_reason=reason,
        )
        for positive, negative, score, reason in matches
    ]
    metrics_dir = out_dir / "metrics"
    write_csv(metrics_dir / "candidates_manifest.csv", positive_manifest)
    write_csv(metrics_dir / "negative_controls_manifest.csv", negative_manifest)
    write_csv(out_dir / "candidates_manifest.csv", positive_manifest)

    match_rows = [
        {
            "positive_candidate_id": positive["candidate_id"],
            "negative_candidate_id": negative["candidate_id"],
            "score": f"{score:.6f}",
            "reason": reason,
        }
        for positive, negative, score, reason in matches
    ]
    summary: dict[str, object] = {
        "candidate_summary": str(candidate_summary),
        "feature_bank": str(feature_bank),
        "sampling_tasks": str(sampling_tasks),
        "out_dir": str(out_dir),
        "n_candidate_rows": len(candidate_rows),
        "n_feature_bank_rows": len(feature_rows),
        "n_sampling_task_rows": len(task_rows),
        "n_positive_candidates": len(positive_manifest),
        "n_negative_controls": len(negative_manifest),
        "positive_status": POSITIVE_STATUS,
        "negative_status": NEGATIVE_STATUS,
        "screen_status_counts": dict(Counter(row.get("screen_status", "") for row in candidate_rows)),
        "positive_candidate_ids": [str(row["candidate_id"]) for row in positive_manifest],
        "negative_candidate_ids": [str(row["candidate_id"]) for row in negative_manifest],
        "negative_control_matches": match_rows,
        "missing_positive_sampling_tasks": [
            str(row["candidate_id"]) for row in positive_manifest if not row["sampling_task_found"]
        ],
        "missing_negative_sampling_tasks": [
            str(row["candidate_id"]) for row in negative_manifest if not row["sampling_task_found"]
        ],
    }
    (metrics_dir / "candidate_manifest_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_materials_manifest(
        out_dir / "materials_manifest.md",
        candidate_summary=candidate_summary,
        feature_bank=feature_bank,
        sampling_tasks=sampling_tasks,
        summary=summary,
    )
    if summary["missing_positive_sampling_tasks"] or summary["missing_negative_sampling_tasks"]:
        warn("some manifest rows have no transfer_replace sampling task metadata")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate_summary", type=Path, required=True)
    p.add_argument("--feature_bank", type=Path, required=True)
    p.add_argument("--sampling_tasks", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_negative_controls", type=int, default=9)
    p.add_argument("--expected_positive_count", type=int, default=9)
    return p


def main() -> int:
    args = build_parser().parse_args()
    summary = build_manifests(
        candidate_summary=args.candidate_summary,
        feature_bank=args.feature_bank,
        sampling_tasks=args.sampling_tasks,
        out_dir=args.out_dir,
        n_negative_controls=args.n_negative_controls,
        expected_positive_count=args.expected_positive_count,
    )
    ok(
        "built steering manifests: "
        f"{summary['n_positive_candidates']} positives, {summary['n_negative_controls']} controls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
