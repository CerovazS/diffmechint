"""Feature-level SAE activation patching analyses (Phase 4.11).

Split from ``scripts/analysis/sae_feature_patching.py``; that script remains a
thin CLI shell. Importing this package pulls torch/matplotlib-heavy submodules.
"""

from .activation import (
    _active_mask_from_scores,
    _fit_ridge_group_calibration,
    _linear_calibration,
    _select_density_norm_control,
    run_activation_patch,
    run_group_activation_patch,
)
from .bank import is_monosemantic, run_build_bank
from .cli import main
from .common import (
    DEFAULT_CONDITIONS,
    DEFAULT_DASHBOARD_ROOT,
    DEFAULT_OUT_ROOT,
    LAYERS,
    PB,
    T_BINS,
    FeatureRow,
    decoder_weight,
    read_csv,
    selected_cells,
    write_csv,
)
from .families import run_group_features, run_summarize_group_patching
from .matching import feature_match_score, run_match
from .sampling import (
    _expand_sampling_rows,
    _summarize_sampling_candidates,
    run_summarize_sampling,
)

__all__ = [
    "DEFAULT_CONDITIONS",
    "DEFAULT_DASHBOARD_ROOT",
    "DEFAULT_OUT_ROOT",
    "LAYERS",
    "PB",
    "T_BINS",
    "FeatureRow",
    "_active_mask_from_scores",
    "_expand_sampling_rows",
    "_fit_ridge_group_calibration",
    "_linear_calibration",
    "_select_density_norm_control",
    "_summarize_sampling_candidates",
    "decoder_weight",
    "feature_match_score",
    "is_monosemantic",
    "main",
    "read_csv",
    "run_activation_patch",
    "run_build_bank",
    "run_group_activation_patch",
    "run_group_features",
    "run_match",
    "run_summarize_group_patching",
    "run_summarize_sampling",
    "selected_cells",
    "write_csv",
]
