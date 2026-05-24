"""Synthetic tests for Phase 4.11 feature activation patching."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffmechint.hooks.timestep_router import timestep_context
from scripts.analysis.sae_feature_patching import (
    FeatureRow,
    _active_mask_from_scores,
    _expand_sampling_rows,
    _fit_ridge_group_calibration,
    _linear_calibration,
    _select_density_norm_control,
    _summarize_sampling_candidates,
    feature_match_score,
    is_monosemantic,
)
from scripts.eval.cross_tokenizer_feature_patching import make_feature_patch_hook


def _feature(feature_id: int, top: int, top9: tuple[int, ...], entropy: float = 1.0) -> FeatureRow:
    return FeatureRow(
        condition="a",
        layer=3,
        t_bin=1,
        feature_id=feature_id,
        density=0.001,
        density_count=10,
        entropy=entropy,
        unique_classes=2,
        mean_act=2.0,
        top_activation=4.0,
        top_class_idx=top,
        top_label=str(top),
        top_synset=f"n{top}",
        top9_class_idx=top9,
        top9_dataset_idx=tuple(range(len(top9))),
        top9_activation=tuple(float(i) for i in range(len(top9))),
        top9_token_pos=tuple(range(len(top9))),
        vlm_interpretation="",
        decoder_norm=1.0,
    )


def test_monosemantic_filter_uses_density_and_entropy() -> None:
    assert is_monosemantic(
        {"live": True, "density": 0.001, "entropy": 2.0, "top": [{}] * 9},
        min_density=1e-4,
        max_density=0.1,
        max_entropy=2.5,
    )
    assert not is_monosemantic(
        {"live": True, "density": 0.001, "entropy": 3.0, "top": [{}] * 9},
        min_density=1e-4,
        max_density=0.1,
        max_entropy=2.5,
    )


def test_feature_match_score_rewards_same_class_overlap() -> None:
    left = _feature(1, 388, (388, 388, 389, 390))
    right = _feature(2, 388, (388, 388, 391, 392))
    wrong = _feature(3, 12, (12, 13, 14, 15))
    assert feature_match_score(left, right)["match_score"] > feature_match_score(left, wrong)["match_score"]


def test_linear_calibration_recovers_affine_coefficients() -> None:
    source = torch.arange(10, dtype=torch.float32).numpy()
    target = 2.5 * source - 1.0
    calib = _linear_calibration(source, target)
    assert calib["slope"] == pytest.approx(2.5)
    assert calib["intercept"] == pytest.approx(-1.0)
    assert calib["r2"] == pytest.approx(1.0)


def test_group_ridge_calibration_recovers_multivariate_affine_map() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(32, 3))
    weight = np.array([[1.0, -0.5], [0.25, 2.0], [-1.0, 0.1]])
    bias = np.array([0.75, -1.25])
    target = source @ weight + bias
    calib = _fit_ridge_group_calibration(source, target, alpha=1e-9)
    np.testing.assert_allclose(np.asarray(calib["weight"]), weight, atol=1e-8)
    np.testing.assert_allclose(np.asarray(calib["bias"]), bias, atol=1e-8)
    assert calib["mean_r2"] == pytest.approx(1.0)


def test_active_mask_falls_back_to_per_image_top_tokens() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.01, 0.02])
    mask = _active_mask_from_scores(
        scores,
        threshold=0.99,
        n_images=2,
        n_tokens=3,
        min_active_tokens=2,
    )
    assert mask.tolist() == [False, False, True, True, False, False]


def test_control_selector_never_returns_target_without_bank() -> None:
    selected = _select_density_norm_control(
        {},
        condition="a",
        layer=3,
        t_bin=1,
        target_feature_id=3,
        target_density=0.001,
        target_decoder_norm=None,
        rng=np.random.default_rng(3),
        fallback_d_sae=4,
    )
    assert selected != 3


def test_sampling_summary_reuses_duplicate_native_controls() -> None:
    tasks = [
        {
            "task_id": "task_000",
            "candidate_id": "baseline_t",
            "mode": "baseline",
            "source": "__NA__",
            "target": "t",
        },
        {
            "task_id": "task_001",
            "candidate_id": "cand_a",
            "mode": "native_ablate",
            "source": "a",
            "target": "t",
            "layer": "3",
            "t_bin": "1",
            "source_feature_id": "10",
            "target_feature_id": "20",
        },
        {
            "task_id": "task_002",
            "candidate_id": "cand_b",
            "mode": "native_ablate",
            "source": "b",
            "target": "t",
            "layer": "3",
            "t_bin": "1",
            "source_feature_id": "11",
            "target_feature_id": "20",
        },
    ]
    results = [
        {
            "mode": "baseline",
            "target_condition": "t",
            "fid": "100.0",
            "hook_stats": '{"active": 0, "skipped": 0, "no_t": 0}',
        },
        {
            "mode": "native_ablate",
            "target_condition": "t",
            "layer": "3",
            "t_bin": "1",
            "target_feature_id": "20",
            "fid": "101.5",
            "hook_stats": '{"active": 4, "skipped": 1, "no_t": 0}',
        },
    ]
    rows = _expand_sampling_rows(results, tasks, [])
    assert len(rows) == 3
    assert rows[1]["coverage_status"] == "ok"
    assert rows[1]["reused_native_control"] is False
    assert rows[2]["coverage_status"] == "ok"
    assert rows[2]["reused_native_control"] is True
    assert rows[2]["delta_fid"] == pytest.approx(1.5)


def test_sampling_candidate_summary_flags_inactive_transfer() -> None:
    rows = [
        {
            "mode": "transfer_replace",
            "candidate_id": "cand",
            "source": "a",
            "target": "b",
            "layer": 3,
            "t_bin": 1,
            "source_feature_id": 1,
            "target_feature_id": 2,
            "delta_fid": 0.2,
            "fid": 10.2,
            "target_baseline_fid": 10.0,
            "hook_active": 0,
            "hook_skipped": 5,
            "hook_no_t": 0,
        },
        {"mode": "random_matched_control", "candidate_id": "cand", "delta_fid": 0.1},
        {"mode": "wrong_window_control", "candidate_id": "cand", "delta_fid": 0.3},
        {"mode": "native_ablate", "candidate_id": "cand", "delta_fid": 0.4},
        {"mode": "native_clamp", "candidate_id": "cand", "delta_fid": 0.5},
    ]
    summary = _summarize_sampling_candidates(rows)
    assert summary[0]["screen_status"] == "transfer_hook_inactive"
    assert summary[0]["complete_expected_modes"] is True


class _ToySae(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.shape[0], 4, device=x.device, dtype=x.dtype)
        z[:, : x.shape[1]] = x
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :2]


def test_zero_target_hook_edits_one_feature_and_preserves_shape() -> None:
    sae = _ToySae()
    stats = {"active": 0, "skipped": 0, "no_t": 0}
    hook = make_feature_patch_hook(
        sae,
        None,
        None,
        mode="zero_target",
        target_feature_id=1,
        source_feature_id=None,
        random_source_feature_id=None,
        source_to_target_scale=1.0,
        t_center=0.20,
        t_tol=0.01,
        stats=stats,
    )
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    with timestep_context(0.20):
        y = hook(None, (), x)
    assert y is not None
    assert y.shape == x.shape
    assert y[0, 0, 1].item() == pytest.approx(0.0)
    assert stats["active"] == 1


def test_zero_scale_source_hook_uses_bias_without_source_sae() -> None:
    sae = _ToySae()
    stats = {"active": 0, "skipped": 0, "no_t": 0}
    hook = make_feature_patch_hook(
        sae,
        None,
        None,
        mode="transfer_replace",
        target_feature_id=1,
        source_feature_id=3,
        random_source_feature_id=None,
        source_to_target_scale=0.0,
        source_to_target_bias=7.0,
        t_center=0.20,
        t_tol=0.01,
        stats=stats,
    )
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    with timestep_context(0.20):
        y = hook(None, (), x)
    assert y is not None
    assert y[0, 0, 1].item() == pytest.approx(7.0)
    assert stats["active"] == 1
