"""Aggregate SAE concept-EAP run directories into one report."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diffmechint.analysis.patching import PB, write_csv
from diffmechint.utils import read_csv

DEFAULT_OUT_ROOT = Path("outputs/phase4_11_feature_activation_patching")


def _float(raw: object) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _mean(values: list[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return float(np.mean(vals)) if vals else float("nan")


def _make_run_dir(out_root: Path, run_id: str | None, *, resume: bool) -> Path:
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / run_id
    if out_dir.exists() and not resume:
        raise FileExistsError(f"run directory already exists: {out_dir}")
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_rows(run_dirs: list[Path], relpath: str) -> list[dict]:
    rows: list[dict] = []
    for run_dir in run_dirs:
        path = run_dir / relpath
        if not path.exists():
            raise FileNotFoundError(f"missing {relpath}: {run_dir}")
        for row in read_csv(path):
            row["source_run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def _plot_metric_heatmap(rows: list[dict], metric: str, out_path: Path, title: str) -> None:
    if not rows:
        return
    row_keys = sorted({(str(r["condition"]), int(r["layer"]), int(r["t_bin"])) for r in rows})
    concepts = sorted({str(r["concept"]) for r in rows})
    arr = np.full((len(row_keys), len(concepts)), np.nan, dtype=np.float32)
    for i, key in enumerate(row_keys):
        for j, concept in enumerate(concepts):
            vals = [
                _float(r[metric])
                for r in rows
                if (str(r["condition"]), int(r["layer"]), int(r["t_bin"])) == key and str(r["concept"]) == concept
            ]
            arr[i, j] = _mean(vals)
    fig, ax = plt.subplots(figsize=(max(6.4, 1.2 * len(concepts)), max(4.2, 0.36 * len(row_keys) + 1.6)))
    im = ax.imshow(arr, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(concepts)), concepts, rotation=30, ha="right")
    ax.set_yticks(range(len(row_keys)), [f"{c} L{layer}/T{t_bin}" for c, layer, t_bin in row_keys])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_top1_label_counts(node_rows: list[dict], out_path: Path) -> None:
    labels = [
        str(row.get("top_label", ""))
        for row in node_rows
        if row.get("node_type") == "sae_feature" and str(row.get("rank_abs")) == "1"
    ]
    counts = Counter(label for label in labels if label)
    if not counts:
        return
    top = counts.most_common(20)
    names = [name for name, _ in top][::-1]
    vals = [count for _, count in top][::-1]
    fig, ax = plt.subplots(figsize=(7.0, max(4.0, 0.3 * len(names) + 1.4)))
    ax.barh(range(len(names)), vals, color=PB["blue"])
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("top-1 circuit count")
    ax.set_title("Most frequent top SAE feature labels")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _faith_topk_summary(rows: list[dict]) -> list[dict]:
    out = []
    for top_k in sorted({int(row["top_k"]) for row in rows}):
        vals = [row for row in rows if int(row["top_k"]) == top_k]
        out.append(
            {
                "top_k": top_k,
                "mean_sufficiency_with_error_retention": _mean(
                    [_float(row["sufficiency_with_error_retention"]) for row in vals]
                ),
                "mean_sufficiency_features_only_retention": _mean(
                    [_float(row["sufficiency_features_only_retention"]) for row in vals]
                ),
                "mean_necessity_drop_fraction": _mean([_float(row["necessity_drop_fraction"]) for row in vals]),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = _make_run_dir(args.out_root, args.run_id, resume=args.resume)
    summary_rows = _load_rows(args.run_dirs, "metrics/concept_eap_summary.csv")
    faith_rows = _load_rows(args.run_dirs, "metrics/topn_faithfulness.csv")
    node_rows = _load_rows(args.run_dirs, "metrics/eap_feature_nodes.csv")
    edge_rows = _load_rows(args.run_dirs, "metrics/eap_feature_edges.csv")
    top1_rows = [
        row
        for row in node_rows
        if row.get("node_type") == "sae_feature" and str(row.get("rank_abs")) == "1"
    ]
    faith_summary = _faith_topk_summary(faith_rows)

    write_csv(out_dir / "metrics" / "concept_eap_summary.csv", summary_rows)
    write_csv(out_dir / "metrics" / "topn_faithfulness.csv", faith_rows)
    write_csv(out_dir / "metrics" / "eap_feature_nodes.csv", node_rows)
    write_csv(out_dir / "metrics" / "eap_feature_edges.csv", edge_rows)
    write_csv(out_dir / "metrics" / "top1_feature_nodes.csv", top1_rows)
    write_csv(out_dir / "metrics" / "topk_faithfulness_summary.csv", faith_summary)

    _plot_metric_heatmap(
        summary_rows,
        "probe_test_accuracy",
        out_dir / "plots" / "concept_probe_accuracy_heatmap.png",
        "Concept probe accuracy",
    )
    _plot_metric_heatmap(
        summary_rows,
        "error_node_abs_gap_fraction",
        out_dir / "plots" / "error_node_fraction_heatmap.png",
        "SAE reconstruction-error share of concept margin",
    )
    top100 = [row for row in faith_rows if int(row["top_k"]) == 100]
    _plot_metric_heatmap(
        top100,
        "sufficiency_with_error_retention",
        out_dir / "plots" / "top100_sufficiency_retention_heatmap.png",
        "Top-100 SAE feature sufficiency with error node",
    )
    _plot_top1_label_counts(node_rows, out_dir / "plots" / "top1_feature_label_counts.png")

    payload = {
        "analysis": "sae-concept-eap-aggregate",
        "n_run_dirs": len(args.run_dirs),
        "n_circuits": len(summary_rows),
        "n_feature_nodes": sum(1 for row in node_rows if row.get("node_type") == "sae_feature"),
        "n_error_nodes": sum(1 for row in node_rows if row.get("node_type") == "reconstruction_error"),
        "n_edges": len(edge_rows),
        "mean_probe_test_accuracy": _mean([_float(row["probe_test_accuracy"]) for row in summary_rows]),
        "mean_error_node_abs_gap_fraction": _mean(
            [_float(row["error_node_abs_gap_fraction"]) for row in summary_rows]
        ),
        "topk_faithfulness_summary": faith_summary,
        "top1_label_counts": dict(Counter(row.get("top_label", "") for row in top1_rows if row.get("top_label", ""))),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# SAE Concept EAP Aggregate",
        "",
        "> [!summary] TL;DR",
        f"> **Concept-margin SAE EAP** aggregated =={payload['n_circuits']}== circuits and =={payload['n_feature_nodes']}== feature nodes.",
        f"> Mean probe accuracy is =={payload['mean_probe_test_accuracy']:.3f}==; mean error-node margin share is =={payload['mean_error_node_abs_gap_fraction']:.3f}==.",
        "> Top-k faithfulness is candidate-discovery evidence, not a final causal transfer claim.",
        "",
        "## Outputs",
        "",
        "- `metrics/concept_eap_summary.csv`",
        "- `metrics/eap_feature_nodes.csv`",
        "- `metrics/eap_feature_edges.csv`",
        "- `metrics/topn_faithfulness.csv`",
        "- `metrics/topk_faithfulness_summary.csv`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"OK aggregate complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
