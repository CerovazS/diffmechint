"""Synthetic tests for Phase 4.18 steering candidate manifest building."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analysis.build_steering_candidates import ManifestBuildError, build_manifests


def _write_rows(path: Path, rows: list[dict[str, object]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _candidate(
    candidate_id: str,
    status: str,
    *,
    source: str = "eq_vae",
    target: str = "repa_e",
    layer: int = 12,
    t_bin: int = 2,
    source_feature_id: int = 10,
    target_feature_id: int = 20,
    target_top_label: str = "ambulance",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source": source,
        "target": target,
        "layer": layer,
        "t_bin": t_bin,
        "source_feature_id": source_feature_id,
        "target_feature_id": target_feature_id,
        "source_top_label": target_top_label,
        "target_top_label": target_top_label,
        "match_score": 0.5,
        "screen_status": status,
    }


def _feature(
    condition: str,
    feature_id: int,
    *,
    layer: int = 12,
    t_bin: int = 2,
    label: str = "ambulance",
    class_idx: int = 407,
    density: float = 0.001,
    decoder_norm: float = 2.0,
) -> dict[str, object]:
    return {
        "condition": condition,
        "layer": layer,
        "t_bin": t_bin,
        "t_center": 0.5,
        "feature_id": feature_id,
        "density": density,
        "density_count": 10,
        "entropy": 1.0,
        "unique_classes": 2,
        "mean_act": 0.8,
        "top_activation": 2.5,
        "top_label": label,
        "top_class_idx": class_idx,
        "top_synset": "n02701002",
        "vlm_interpretation": label,
        "top9_class_idx": json.dumps([class_idx] * 9),
        "top9_dataset_idx": json.dumps(list(range(9))),
        "top9_activation": json.dumps([1.0] * 9),
        "top9_token_pos": json.dumps(list(range(9))),
        "n_top_examples": 9,
        "decoder_norm": decoder_norm,
    }


def test_build_manifests_filters_positives_and_selects_controls(tmp_path: Path) -> None:
    pos = "transfer_lower_fid_than_random_and_wrong_window"
    neg = "no_transfer_specific_fid_advantage"
    candidates = [
        _candidate("pos_a", pos, source_feature_id=10, target_feature_id=20),
        _candidate("pos_b", pos, source_feature_id=11, target_feature_id=21, target="sd_vae"),
        _candidate("neg_a", neg, source_feature_id=12, target_feature_id=22),
        _candidate("neg_b", neg, source_feature_id=13, target_feature_id=23, target="sd_vae"),
        _candidate("neg_c", neg, source_feature_id=14, target_feature_id=24, layer=18),
    ]
    features = [
        _feature("eq_vae", 10),
        _feature("eq_vae", 11),
        _feature("eq_vae", 12),
        _feature("eq_vae", 13),
        _feature("eq_vae", 14, layer=18),
        _feature("repa_e", 20),
        _feature("sd_vae", 21),
        _feature("repa_e", 22),
        _feature("sd_vae", 23),
        _feature("repa_e", 24, layer=18),
    ]
    tasks = [
        {
            "task_id": "task_pos_a",
            "candidate_id": "pos_a",
            "mode": "transfer_replace",
            "random_source_feature_id": 99,
            "wrong_t_bin": 0,
            "scale": 1.5,
            "bias": 0.1,
        },
        {
            "task_id": "task_neg_a",
            "candidate_id": "neg_a",
            "mode": "transfer_replace",
            "random_source_feature_id": 98,
            "wrong_t_bin": 0,
            "scale": 1.0,
            "bias": 0.0,
        },
    ]
    candidate_path = tmp_path / "candidate_summary.csv"
    feature_path = tmp_path / "feature_bank.csv"
    task_path = tmp_path / "sampling_tasks.tsv"
    _write_rows(candidate_path, candidates)
    _write_rows(feature_path, features)
    _write_rows(task_path, tasks, delimiter="\t")

    summary = build_manifests(
        candidate_summary=candidate_path,
        feature_bank=feature_path,
        sampling_tasks=task_path,
        out_dir=tmp_path / "out",
        n_negative_controls=2,
        expected_positive_count=2,
    )

    assert summary["n_positive_candidates"] == 2
    assert summary["n_negative_controls"] == 2
    positives = list(csv.DictReader((tmp_path / "out" / "metrics" / "candidates_manifest.csv").open()))
    controls = list(
        csv.DictReader((tmp_path / "out" / "metrics" / "negative_controls_manifest.csv").open())
    )
    assert positives[0]["target_top_class_idx"] == "407"
    assert positives[0]["sampling_task_task_id"] == "task_pos_a"
    assert controls[0]["selection_role"] == "negative_control"
    assert controls[0]["matched_positive_id"] in {"pos_a", "pos_b"}
    assert (tmp_path / "out" / "materials_manifest.md").exists()


def test_build_manifests_fails_on_missing_feature_metadata(tmp_path: Path) -> None:
    pos = "transfer_lower_fid_than_random_and_wrong_window"
    candidate_path = tmp_path / "candidate_summary.csv"
    feature_path = tmp_path / "feature_bank.csv"
    task_path = tmp_path / "sampling_tasks.tsv"
    _write_rows(candidate_path, [_candidate("pos_a", pos)])
    _write_rows(feature_path, [_feature("eq_vae", 10)])
    _write_rows(task_path, [{"task_id": "task_pos_a", "candidate_id": "pos_a", "mode": "transfer_replace"}], delimiter="\t")

    with pytest.raises(ManifestBuildError, match="lacks feature-bank metadata"):
        build_manifests(
            candidate_summary=candidate_path,
            feature_bank=feature_path,
            sampling_tasks=task_path,
            out_dir=tmp_path / "out",
            n_negative_controls=0,
            expected_positive_count=1,
        )
