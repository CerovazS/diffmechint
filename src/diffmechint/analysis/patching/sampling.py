"""Sampling-time feature patching aggregation (FID screen/final)."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diffmechint.utils import ok

from .common import (
    PB,
    _clean_optional,
    _json_dict,
    _optional_bool,
    _optional_float,
    _optional_int,
    read_csv,
    read_tsv,
    write_csv,
)


def _sampling_key_from_task(row: dict) -> tuple:
    mode = _clean_optional(row.get("mode"))
    target = _clean_optional(row.get("target"))
    source = _clean_optional(row.get("source"))
    layer = _optional_int(row.get("layer"))
    t_bin = _optional_int(row.get("t_bin"))
    source_feature_id = _optional_int(row.get("source_feature_id"))
    target_feature_id = _optional_int(row.get("target_feature_id"))
    random_source_feature_id = _optional_int(row.get("random_source_feature_id"))
    wrong_t_bin = _optional_int(row.get("wrong_t_bin"))
    if mode == "baseline":
        return ("baseline", target)
    if mode in {"native_ablate", "native_clamp"}:
        return ("native", mode, target, layer, t_bin, target_feature_id)
    if mode == "random_matched_control":
        return ("random", mode, source, target, layer, t_bin, target_feature_id, random_source_feature_id)
    if mode == "wrong_window_control":
        return ("wrong", mode, source, target, layer, t_bin, source_feature_id, target_feature_id, wrong_t_bin)
    return ("transfer", mode, source, target, layer, t_bin, source_feature_id, target_feature_id)


def _sampling_key_from_result(row: dict) -> tuple:
    mode = _clean_optional(row.get("mode"))
    target = _clean_optional(row.get("target_condition"))
    source = _clean_optional(row.get("source_condition"))
    layer = _optional_int(row.get("layer"))
    t_bin = _optional_int(row.get("t_bin"))
    source_feature_id = _optional_int(row.get("source_feature_id"))
    target_feature_id = _optional_int(row.get("target_feature_id"))
    random_source_feature_id = _optional_int(row.get("random_source_feature_id"))
    wrong_t_bin = _optional_int(row.get("wrong_t_bin"))
    if mode == "baseline":
        return ("baseline", target)
    if mode in {"native_ablate", "native_clamp"}:
        return ("native", mode, target, layer, t_bin, target_feature_id)
    if mode == "random_matched_control":
        return ("random", mode, source, target, layer, t_bin, target_feature_id, random_source_feature_id)
    if mode == "wrong_window_control":
        return ("wrong", mode, source, target, layer, t_bin, source_feature_id, target_feature_id, wrong_t_bin)
    return ("transfer", mode, source, target, layer, t_bin, source_feature_id, target_feature_id)


def _candidate_key(row: dict) -> tuple[str, str, int | None, int | None, int | None, int | None]:
    return (
        _clean_optional(row.get("source")),
        _clean_optional(row.get("target")),
        _optional_int(row.get("layer")),
        _optional_int(row.get("t_bin")),
        _optional_int(row.get("source_feature_id")),
        _optional_int(row.get("target_feature_id")),
    )


def _index_sampling_targets(target_rows: list[dict]) -> dict[tuple, dict]:
    return {_candidate_key(row): row for row in target_rows}


def _expand_sampling_rows(
    result_rows: list[dict],
    task_rows: list[dict],
    target_rows: list[dict],
) -> list[dict]:
    result_by_key = {_sampling_key_from_result(row): row for row in result_rows}
    target_by_key = _index_sampling_targets(target_rows)
    baselines: dict[str, float] = {}
    for row in result_rows:
        if _clean_optional(row.get("mode")) == "baseline":
            target = _clean_optional(row.get("target_condition"))
            fid = _optional_float(row.get("fid"))
            if fid is not None:
                baselines[target] = fid

    native_first_task: dict[tuple, str] = {}
    for task in task_rows:
        key = _sampling_key_from_task(task)
        if key and key[0] == "native":
            native_first_task.setdefault(key, _clean_optional(task.get("task_id")))

    rows = []
    for task in task_rows:
        key = _sampling_key_from_task(task)
        result = result_by_key.get(key)
        mode = _clean_optional(task.get("mode"))
        source = _clean_optional(task.get("source"))
        target = _clean_optional(task.get("target"))
        target_meta = target_by_key.get(_candidate_key(task), {})
        fid = _optional_float(result.get("fid") if result else None)
        baseline_fid = fid if mode == "baseline" else baselines.get(target)
        hook = _json_dict(result.get("hook_stats") if result else None)
        reused_native = bool(
            key
            and key[0] == "native"
            and _clean_optional(task.get("task_id")) != native_first_task.get(key, "")
        )
        delta_fid = (
            float(fid - baseline_fid)
            if fid is not None and baseline_fid is not None
            else float("nan")
        )
        rows.append(
            {
                "task_id": _clean_optional(task.get("task_id")),
                "candidate_id": _clean_optional(task.get("candidate_id")),
                "selection_role": _clean_optional(target_meta.get("selection_role"))
                or ("baseline" if mode == "baseline" else ""),
                "mode": mode,
                "source": source,
                "target": target,
                "layer": _optional_int(task.get("layer")) or "",
                "t_bin": _optional_int(task.get("t_bin")) or "",
                "source_feature_id": _optional_int(task.get("source_feature_id")) or "",
                "target_feature_id": _optional_int(task.get("target_feature_id")) or "",
                "random_source_feature_id": _optional_int(task.get("random_source_feature_id")) or "",
                "wrong_t_bin": _optional_int(task.get("wrong_t_bin")) or "",
                "source_top_label": _clean_optional(target_meta.get("source_top_label")),
                "target_top_label": _clean_optional(target_meta.get("target_top_label")),
                "match_score": _optional_float(target_meta.get("match_score")),
                "specificity_delta_ev": _optional_float(target_meta.get("specificity_delta_ev")),
                "activation_transfer_delta_ev": _optional_float(target_meta.get("transfer_delta_ev")),
                "activation_random_control_delta_ev": _optional_float(target_meta.get("random_control_delta_ev")),
                "activation_shuffled_control_delta_ev": _optional_float(target_meta.get("shuffled_control_delta_ev")),
                "calibrated_feature_corr": _optional_float(target_meta.get("calibrated_feature_corr")),
                "calibration_slope": _optional_float(task.get("scale")),
                "calibration_intercept": _optional_float(task.get("bias")),
                "patch_active_frac": _optional_float(target_meta.get("patch_active_frac")),
                "target_baseline_fid": baseline_fid if baseline_fid is not None else "",
                "fid": fid if fid is not None else "",
                "delta_fid": delta_fid,
                "hook_active": int(hook.get("active", 0) or 0),
                "hook_skipped": int(hook.get("skipped", 0) or 0),
                "hook_no_t": int(hook.get("no_t", 0) or 0),
                "n_samples": _optional_int(result.get("n_samples") if result else None) or "",
                "cfg_scale": _optional_float(result.get("cfg_scale") if result else None),
                "sampler": _clean_optional(result.get("sampler") if result else ""),
                "sample_steps": _optional_int(result.get("sample_steps") if result else None) or "",
                "seed": _optional_int(result.get("seed") if result else None) or "",
                "no_normalize": _optional_bool(result.get("no_normalize") if result else None),
                "raw_run_tag": _clean_optional(result.get("run_tag") if result else ""),
                "reused_native_control": reused_native,
                "coverage_status": "ok" if result is not None else "missing",
                "gen_minutes": _optional_float(result.get("gen_minutes") if result else None),
                "fid_minutes": _optional_float(result.get("fid_minutes") if result else None),
            }
        )
    return rows


def _mode_value(mode_rows: dict[str, dict], mode: str, field: str) -> float:
    row = mode_rows.get(mode, {})
    value = _optional_float(row.get(field))
    return float(value) if value is not None else float("nan")


def _summarize_sampling_candidates(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["mode"] != "baseline":
            grouped[str(row["candidate_id"])].append(row)

    summary_rows = []
    expected_modes = {
        "transfer_replace",
        "native_ablate",
        "native_clamp",
        "random_matched_control",
        "wrong_window_control",
    }
    for candidate_id, vals in sorted(grouped.items()):
        mode_rows = {str(row["mode"]): row for row in vals}
        first = vals[0]
        transfer = mode_rows.get("transfer_replace", {})
        transfer_delta = _mode_value(mode_rows, "transfer_replace", "delta_fid")
        random_delta = _mode_value(mode_rows, "random_matched_control", "delta_fid")
        wrong_delta = _mode_value(mode_rows, "wrong_window_control", "delta_fid")
        ablate_delta = _mode_value(mode_rows, "native_ablate", "delta_fid")
        clamp_delta = _mode_value(mode_rows, "native_clamp", "delta_fid")
        transfer_minus_random = transfer_delta - random_delta
        transfer_minus_wrong = transfer_delta - wrong_delta
        status = "missing_transfer"
        if "transfer_replace" in mode_rows:
            active = int(transfer.get("hook_active", 0) or 0)
            if active <= 0:
                status = "transfer_hook_inactive"
            elif transfer_minus_random < 0 and transfer_minus_wrong < 0:
                status = "transfer_lower_fid_than_random_and_wrong_window"
            elif transfer_minus_random < 0:
                status = "transfer_lower_fid_than_random_only"
            elif transfer_minus_wrong < 0:
                status = "transfer_lower_fid_than_wrong_window_only"
            else:
                status = "no_transfer_specific_fid_advantage"
        present_modes = sorted(mode_rows)
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "selection_role": first.get("selection_role", ""),
                "source": first.get("source", ""),
                "target": first.get("target", ""),
                "layer": first.get("layer", ""),
                "t_bin": first.get("t_bin", ""),
                "source_feature_id": first.get("source_feature_id", ""),
                "target_feature_id": first.get("target_feature_id", ""),
                "source_top_label": first.get("source_top_label", ""),
                "target_top_label": first.get("target_top_label", ""),
                "match_score": first.get("match_score", ""),
                "specificity_delta_ev": first.get("specificity_delta_ev", ""),
                "calibrated_feature_corr": first.get("calibrated_feature_corr", ""),
                "calibration_slope": first.get("calibration_slope", ""),
                "patch_active_frac": first.get("patch_active_frac", ""),
                "target_baseline_fid": first.get("target_baseline_fid", ""),
                "transfer_fid": _mode_value(mode_rows, "transfer_replace", "fid"),
                "transfer_delta_fid": transfer_delta,
                "native_ablate_delta_fid": ablate_delta,
                "native_clamp_delta_fid": clamp_delta,
                "random_control_delta_fid": random_delta,
                "wrong_window_delta_fid": wrong_delta,
                "transfer_minus_random_control_delta_fid": transfer_minus_random,
                "transfer_minus_wrong_window_delta_fid": transfer_minus_wrong,
                "wrong_window_minus_transfer_delta_fid": wrong_delta - transfer_delta,
                "best_native_delta_fid": float(np.nanmin([ablate_delta, clamp_delta])),
                "transfer_hook_active": int(transfer.get("hook_active", 0) or 0),
                "transfer_hook_skipped": int(transfer.get("hook_skipped", 0) or 0),
                "transfer_hook_no_t": int(transfer.get("hook_no_t", 0) or 0),
                "n_modes_present": len(present_modes),
                "present_modes": present_modes,
                "complete_expected_modes": expected_modes.issubset(set(present_modes)),
                "screen_status": status,
            }
        )
    return summary_rows


def _plot_sampling_delta_by_candidate(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    modes = [
        ("transfer_delta_fid", "transfer", PB["blue"]),
        ("random_control_delta_fid", "random", PB["gold"]),
        ("wrong_window_delta_fid", "wrong-window", PB["red"]),
        ("native_ablate_delta_fid", "native ablate", "#8a9868"),
        ("native_clamp_delta_fid", "native clamp", "#7890a0"),
    ]
    labels = [
        f"{row['source']}->{row['target']}\nL{row['layer']}/T{row['t_bin']} F{row['source_feature_id']}->{row['target_feature_id']}"
        for row in rows
    ]
    x = np.arange(len(rows), dtype=np.float32)
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(10.0, 0.78 * len(rows)), 5.2))
    ax.axhline(0.0, color=PB["dark"], linewidth=1.0)
    for i, (field, label, color) in enumerate(modes):
        vals = [float(row.get(field, float("nan"))) for row in rows]
        ax.bar(x + (i - 2) * width, vals, width=width, label=label, color=color)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("FID - target baseline (lower is better)")
    ax.set_title("Sampling-time feature patching screen")
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_sampling_transfer_vs_controls(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    labels = [
        f"{row['source']}->{row['target']} L{row['layer']}/T{row['t_bin']}\n{row['target_top_label']}"
        for row in rows
    ]
    x = np.arange(len(rows), dtype=np.float32)
    width = 0.34
    random_vals = [float(row["transfer_minus_random_control_delta_fid"]) for row in rows]
    wrong_vals = [float(row["transfer_minus_wrong_window_delta_fid"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(9.5, 0.72 * len(rows)), 4.8))
    ax.axhline(0.0, color=PB["dark"], linewidth=1.0)
    ax.bar(x - width / 2, random_vals, width=width, color=PB["blue"], label="transfer - random")
    ax.bar(x + width / 2, wrong_vals, width=width, color=PB["red"], label="transfer - wrong-window")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Delta FID difference (negative favors transfer)")
    ax.set_title("Transfer specificity against sampling controls")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _sampling_summary_payload(
    rows: list[dict],
    candidate_rows: list[dict],
    *,
    sampling_csv: Path,
    task_tsv: Path,
    target_csv: Path | None,
) -> dict:
    baselines = {
        str(row["target"]): float(row["fid"])
        for row in rows
        if row["mode"] == "baseline" and _optional_float(row.get("fid")) is not None
    }
    n_samples_values = [
        int(row["n_samples"])
        for row in rows
        if _optional_int(row.get("n_samples")) is not None
    ]
    n_samples = max(n_samples_values) if n_samples_values else 0
    run_stage = "final" if n_samples >= 5000 else "screen"
    transfer_rows = [row for row in candidate_rows if not math.isnan(float(row["transfer_delta_fid"]))]
    recommended = sorted(
        [
            row
            for row in transfer_rows
            if int(row["transfer_hook_active"]) > 0
            and float(row["transfer_minus_random_control_delta_fid"]) < 0
        ],
        key=lambda row: (
            float(row["transfer_minus_random_control_delta_fid"]),
            float(row["transfer_minus_wrong_window_delta_fid"]),
        ),
    )[:6]
    if not recommended:
        recommended = sorted(
            [row for row in transfer_rows if int(row["transfer_hook_active"]) > 0],
            key=lambda row: float(row["transfer_minus_random_control_delta_fid"]),
        )[:6]
    return {
        "analysis": f"sampling-time-feature-patching-{run_stage}",
        "run_stage": run_stage,
        "n_samples": n_samples,
        "sampling_csv": str(sampling_csv),
        "task_tsv": str(task_tsv),
        "target_csv": "" if target_csv is None else str(target_csv),
        "n_task_rows": len(rows),
        "n_candidate_rows": len(candidate_rows),
        "n_missing_logical_tasks": sum(1 for row in rows if row["coverage_status"] != "ok"),
        "n_reused_native_control_rows": sum(1 for row in rows if bool(row["reused_native_control"])),
        "baseline_fid": baselines,
        "n_transfer_candidates": len(transfer_rows),
        "n_transfer_hook_inactive": sum(1 for row in candidate_rows if int(row["transfer_hook_active"]) <= 0),
        "n_transfer_lower_fid_than_baseline": sum(
            1 for row in candidate_rows if float(row["transfer_delta_fid"]) < 0
        ),
        "n_transfer_lower_fid_than_random": sum(
            1 for row in candidate_rows if float(row["transfer_minus_random_control_delta_fid"]) < 0
        ),
        "n_transfer_lower_fid_than_wrong_window": sum(
            1 for row in candidate_rows if float(row["transfer_minus_wrong_window_delta_fid"]) < 0
        ),
        "n_transfer_lower_fid_than_random_and_wrong_window": sum(
            1
            for row in candidate_rows
            if float(row["transfer_minus_random_control_delta_fid"]) < 0
            and float(row["transfer_minus_wrong_window_delta_fid"]) < 0
        ),
        "screen_status_counts": dict(Counter(str(row["screen_status"]) for row in candidate_rows)),
        "selection_role_counts": dict(Counter(str(row["selection_role"]) for row in candidate_rows)),
        "recommended_final_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "source": row["source"],
                "target": row["target"],
                "layer": row["layer"],
                "t_bin": row["t_bin"],
                "source_feature_id": row["source_feature_id"],
                "target_feature_id": row["target_feature_id"],
                "target_top_label": row["target_top_label"],
                "transfer_delta_fid": row["transfer_delta_fid"],
                "transfer_minus_random_control_delta_fid": row["transfer_minus_random_control_delta_fid"],
                "transfer_minus_wrong_window_delta_fid": row["transfer_minus_wrong_window_delta_fid"],
                "transfer_hook_active": row["transfer_hook_active"],
                "screen_status": row["screen_status"],
            }
            for row in recommended
        ],
    }


def _write_sampling_summary_md(out_dir: Path, payload: dict, candidate_rows: list[dict]) -> None:
    candidates = sorted(
        candidate_rows,
        key=lambda row: float(row["transfer_minus_random_control_delta_fid"]),
    )
    run_stage = str(payload.get("run_stage", "screen"))
    n_samples = int(payload.get("n_samples", 0) or 0)
    title = "Sampling-Time Feature Patching Final" if run_stage == "final" else "Sampling-Time Feature Patching Screen"
    caveat = (
        "This is the final N=5000 sampling validation for selected candidates."
        if run_stage == "final"
        else "This is a screening result for selecting final N=5000 interventions, not a final FID claim."
    )
    reuse_line = (
        f"- {payload['n_reused_native_control_rows']} native control rows were reused because native target-feature interventions are independent of the source tokenizer."
        if int(payload["n_reused_native_control_rows"]) > 0
        else "- No native-control rows were reused."
    )
    lines = [
        f"# {title}",
        "",
        "> [!summary] TL;DR",
        (
            f"> **N={n_samples} sampling-time feature patching** covered "
            f"=={payload['n_task_rows']} logical task slots== with "
            f"{payload['n_reused_native_control_rows']} reused native controls."
        ),
        (
            f"> Transfer hooks were inactive for "
            f"=={payload['n_transfer_hook_inactive']}/{payload['n_transfer_candidates']}== candidates; "
            f"{payload['n_transfer_lower_fid_than_random_and_wrong_window']} beat both random and wrong-window controls."
        ),
        f"> {caveat}",
        "",
        "## Method",
        "",
        f"- Same-noise ImageNet-class sampling was run with `N={n_samples}`, CFG `1.5`, ODE dopri5, 250 sampler steps, seed `0`.",
        "- Modes were `baseline`, `transfer_replace`, `native_ablate`, `native_clamp`, `random_matched_control`, and `wrong_window_control`.",
        "- `eq_vae` sampling used the non-normalized/noz convention.",
        reuse_line,
        "",
        "## Baselines",
        "",
    ]
    for target, fid in sorted(payload["baseline_fid"].items()):
        lines.append(f"- `{target}`: FID `{fid:.3f}`")
    lines.extend(["", "## Candidate Ranking", ""])
    for row in candidates:
        lines.append(
            "- "
            f"`{row['candidate_id']}`: transfer delta FID `{float(row['transfer_delta_fid']):+.3f}`, "
            f"transfer-random `{float(row['transfer_minus_random_control_delta_fid']):+.3f}`, "
            f"transfer-wrong-window `{float(row['transfer_minus_wrong_window_delta_fid']):+.3f}`, "
            f"hook active `{row['transfer_hook_active']}`, status `{row['screen_status']}`."
        )
    lines.extend(["", "## Recommended Final Candidates", ""])
    for row in payload["recommended_final_candidates"]:
        lines.append(
            "- "
            f"`{row['candidate_id']}`: `{row['source']}->{row['target']}` "
            f"L{row['layer']}/T{row['t_bin']} F{row['source_feature_id']}->{row['target_feature_id']} "
            f"({row['target_top_label']}), transfer-random "
            f"`{float(row['transfer_minus_random_control_delta_fid']):+.3f}`."
        )
    lines.append("")
    summary_name = "sampling_final_summary.md" if run_stage == "final" else "sampling_screen_summary.md"
    (out_dir / summary_name).write_text("\n".join(lines), encoding="utf-8")


def run_summarize_sampling(args: argparse.Namespace) -> int:
    out_dir = args.out_dir or args.sampling_csv.parent
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    result_rows = read_csv(args.sampling_csv)
    task_rows = read_tsv(args.task_tsv)
    target_rows = read_csv(args.target_csv) if args.target_csv is not None else []
    analysis_rows = _expand_sampling_rows(result_rows, task_rows, target_rows)
    candidate_rows = _summarize_sampling_candidates(analysis_rows)
    write_csv(out_dir / "metrics" / "sampling_feature_patching_analysis.csv", analysis_rows)
    write_csv(out_dir / "metrics" / "sampling_candidate_summary.csv", candidate_rows)
    _plot_sampling_delta_by_candidate(candidate_rows, out_dir / "plots" / "sampling_delta_fid_by_candidate.png")
    _plot_sampling_transfer_vs_controls(candidate_rows, out_dir / "plots" / "sampling_transfer_vs_controls.png")
    payload = _sampling_summary_payload(
        analysis_rows,
        candidate_rows,
        sampling_csv=args.sampling_csv,
        task_tsv=args.task_tsv,
        target_csv=args.target_csv,
    )
    (out_dir / "metrics" / "sampling_feature_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_sampling_summary_md(out_dir, payload, candidate_rows)
    ok(f"sampling summary complete: {out_dir}")
    return 0
