"""Plot the Phase 4.18 steering/EAP/output-metric evidence summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PALETTE = {
    "blue": "#335C67",
    "yellow": "#FFF3B0",
    "gold": "#E09F3E",
    "red": "#9E2A2B",
    "dark": "#540B0E",
}
METRICS = [
    ("classifier_prob", "classifier"),
    ("clip_text", "CLIP text"),
    ("dino_top", "DINO top"),
]
ROLE_COLORS = {
    "positive": PALETTE["blue"],
    "negative_control": PALETTE["gold"],
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _bool(raw: str) -> bool:
    return raw.strip().lower() == "true"


def _count(rows: list[dict[str, str]], *, role: str, key: str) -> int:
    return sum(row["selection_role"] == role and _bool(row[key]) for row in rows)


def _count_ci_positive(rows: list[dict[str, str]], *, role: str, metric: str) -> int:
    key = f"{metric}_transfer_minus_baseline_ci_low"
    return sum(row["selection_role"] == role and float(row[key]) > 0.0 for row in rows)


def _label_bars(ax: plt.Axes, bars: Any) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.18,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111111",
        )


def _short_id(candidate_id: str, idx: int) -> str:
    bits = candidate_id.split("__")
    if len(bits) >= 3:
        return f"{idx + 1}: {bits[2]}"
    return f"{idx + 1}"


def _plot_eap_candidate_ranks(metrics_dir: Path, plots_dir: Path) -> None:
    rows = _read_csv(metrics_dir / "full_eap_candidate_summary.csv")
    rows = sorted(rows, key=lambda row: (row["selection_role"], int(row["source_rank_abs"])))
    labels = [_short_id(row["candidate_id"], idx) for idx, row in enumerate(rows)]
    x = list(range(len(rows)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)
    source = [int(row["source_rank_abs"]) for row in rows]
    target = [int(row["target_rank_abs"]) for row in rows]
    colors = [ROLE_COLORS.get(row["selection_role"], PALETTE["dark"]) for row in rows]
    ax.bar([i - width / 2 for i in x], source, width, label="source rank", color=colors, alpha=0.9)
    ax.bar([i + width / 2 for i in x], target, width, label="target rank", color=PALETTE["red"], alpha=0.75)
    ax.axhline(10, color="#111111", linewidth=1.0, linestyle="--", label="top10")
    ax.set_title("Concept EAP candidate ranks")
    ax.set_ylabel("absolute attribution rank")
    ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "eap_candidate_rank_by_candidate.png", dpi=200, facecolor="#FFFFFF")
    plt.close(fig)


def _plot_eap_error_share(metrics_dir: Path, plots_dir: Path) -> None:
    rank_rows = _read_csv(metrics_dir / "full_eap_candidate_ranks.csv")
    error_rows = _read_csv(metrics_dir / "full_eap_error_node_share.csv")
    error_by_key = {
        (row["condition"], row["layer"], row["t_bin"], row["concept"]): float(row["error_node_abs_gap_fraction"])
        for row in error_rows
    }
    by_candidate: dict[str, dict[str, Any]] = {}
    for row in rank_rows:
        key = (row["condition"], row["layer"], row["t_bin"], row["concept"])
        if key not in error_by_key:
            continue
        item = by_candidate.setdefault(
            row["candidate_id"],
            {"selection_role": row["selection_role"], "values": []},
        )
        item["values"].append(error_by_key[key])
    rows = [
        {
            "candidate_id": candidate_id,
            "selection_role": item["selection_role"],
            "mean_error_share": sum(item["values"]) / len(item["values"]),
        }
        for candidate_id, item in by_candidate.items()
        if item["values"]
    ]
    rows = sorted(rows, key=lambda row: (row["selection_role"], -row["mean_error_share"]))
    labels = [_short_id(row["candidate_id"], idx) for idx, row in enumerate(rows)]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)
    colors = [ROLE_COLORS.get(row["selection_role"], PALETTE["dark"]) for row in rows]
    bars = ax.bar(x, [row["mean_error_share"] for row in rows], color=colors)
    ax.set_title("Concept EAP reconstruction-error node share")
    ax.set_ylabel("mean absolute gap fraction")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111111",
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "eap_error_node_share_by_candidate.png", dpi=200, facecolor="#FFFFFF")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    metrics_dir = args.run_root / "metrics"
    plots_dir = args.run_root / "plots"
    out = args.out or plots_dir / "phase4_18_evidence_summary.png"
    plots_dir.mkdir(parents=True, exist_ok=True)

    steering = _read_json(metrics_dir / "full_steering_aggregate_summary.json")
    eap = _read_json(metrics_dir / "full_eap_aggregate_summary.json")
    output = _read_json(metrics_dir / "full_output_metric_aggregate_summary.json")
    output_rows = _read_csv(metrics_dir / "full_output_metric_candidate_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor="#FFFFFF")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)
        ax.set_ylim(0, 9)
        ax.set_ylabel("candidates / 9")

    labels = ["FID screen", "EAP top10", "output any", "output strict"]
    positive_counts = [
        steering["positive_transfer_lower_than_both_controls"],
        eap["positive_both_in_top10"],
        output["positive_any_metric_confirmed"],
        output["positive_strict_confirmed"],
    ]
    control_counts = [
        steering["negative_transfer_lower_than_both_controls"],
        eap["negative_both_in_top10"],
        output["negative_any_metric_confirmed"],
        output["negative_strict_confirmed"],
    ]
    x = list(range(len(labels)))
    width = 0.36
    bars_a = axes[0].bar(
        [i - width / 2 for i in x],
        positive_counts,
        width,
        label="positive",
        color=PALETTE["blue"],
    )
    bars_b = axes[0].bar(
        [i + width / 2 for i in x],
        control_counts,
        width,
        label="control",
        color=PALETTE["gold"],
    )
    axes[0].set_title("Evidence funnel")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].legend(frameon=False)
    _label_bars(axes[0], bars_a)
    _label_bars(axes[0], bars_b)

    for ax, role, title in [
        (axes[1], "positive", "Positive candidates"),
        (axes[2], "negative_control", "Matched controls"),
    ]:
        metric_labels = [label for _, label in METRICS]
        baseline_ci = [_count_ci_positive(output_rows, role=role, metric=metric) for metric, _ in METRICS]
        confirmed = [_count(output_rows, role=role, key=f"{metric}_confirmed") for metric, _ in METRICS]
        x_metric = list(range(len(METRICS)))
        bars_c = ax.bar(
            [i - width / 2 for i in x_metric],
            baseline_ci,
            width,
            label="baseline CI > 0",
            color=PALETTE["red"],
        )
        bars_d = ax.bar(
            [i + width / 2 for i in x_metric],
            confirmed,
            width,
            label="beats controls",
            color=PALETTE["blue"],
        )
        ax.set_title(title)
        ax.set_xticks(x_metric, metric_labels)
        ax.legend(frameon=False, fontsize=8)
        _label_bars(ax, bars_c)
        _label_bars(ax, bars_d)

    fig.suptitle("Phase 4.18: no output-specific steering confirmation", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="#FFFFFF")
    plt.close(fig)
    _plot_eap_candidate_ranks(metrics_dir, plots_dir)
    _plot_eap_error_share(metrics_dir, plots_dir)
    print(out)


if __name__ == "__main__":
    main()
