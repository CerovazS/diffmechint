"""Argument parsing and dispatch for the feature-patching CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from diffmechint.analysis.alignment import ACTIVATIONS_YNULL, SAE_ROOT, T_CENTERS
from diffmechint.utils import model_subdir, model_variant_spec, parse_layers

from .activation import run_activation_patch, run_group_activation_patch
from .bank import run_build_bank
from .common import (
    DEFAULT_CONDITIONS,
    DEFAULT_DASHBOARD_ROOT,
    DEFAULT_OUT_ROOT,
    T_BINS,
)
from .families import run_group_features, run_summarize_group_patching
from .matching import run_match
from .sampling import run_summarize_sampling

SCRATCH_PROJECT_ROOT = ACTIVATIONS_YNULL.parent
OUTPUT_ROOT = Path("outputs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    bank = sub.add_parser("build-bank")
    _add_common_args(bank)
    bank.add_argument("--dashboard_root", type=Path, default=DEFAULT_DASHBOARD_ROOT)
    bank.add_argument("--atlas_csv", type=Path, default=None)
    bank.add_argument("--hydrate_dashboard_json", action="store_true")
    bank.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    bank.add_argument("--with_decoder_norms", action="store_true")
    bank.add_argument("--device", type=str, default=None)
    bank.add_argument("--monosemantic_only", action="store_true", default=True)
    bank.add_argument("--include_all_live", dest="monosemantic_only", action="store_false")
    bank.add_argument("--min_density", type=float, default=1e-4)
    bank.add_argument("--max_density", type=float, default=0.10)
    bank.add_argument("--max_entropy", type=float, default=2.5)
    bank.add_argument("--max_features_per_cell", type=int, default=None)
    bank.set_defaults(func=run_build_bank)

    match = sub.add_parser("match")
    _add_common_args(match)
    match.add_argument("--feature_bank", type=Path, required=True)
    match.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    match.add_argument("--metadata_only", action="store_true")
    match.add_argument("--device", type=str, default=None)
    match.add_argument("--profile_batch_images", type=int, default=2)
    match.add_argument("--max_fit_tokens", type=int, default=8192)
    match.add_argument("--ridge_alpha", type=float, default=10.0)
    match.add_argument("--cache_profiles", action="store_true")
    match.add_argument("--same_top_class_only", action="store_true", default=True)
    match.add_argument("--allow_cross_class", dest="same_top_class_only", action="store_false")
    match.add_argument("--min_candidate_score", type=float, default=0.25)
    match.add_argument("--min_hungarian_score", type=float, default=0.35)
    match.add_argument("--shared_score_threshold", type=float, default=0.65)
    match.add_argument("--shared_activation_corr_threshold", type=float, default=0.40)
    match.add_argument("--shared_class_jaccard_threshold", type=float, default=0.30)
    match.set_defaults(func=run_match)

    patch = sub.add_parser("activation-patch")
    _add_common_args(patch)
    patch.add_argument("--match_csv", type=Path, required=True)
    patch.add_argument("--feature_bank", type=Path, default=None)
    patch.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    patch.add_argument("--max_fit_tokens", type=int, default=50_000)
    patch.add_argument("--max_eval_tokens", type=int, default=8192)
    patch.add_argument("--ridge_alpha", type=float, default=10.0)
    patch.add_argument("--sae_batch_size", type=int, default=512)
    patch.add_argument("--device", type=str, default=None)
    patch.add_argument("--max_pairs", type=int, default=None)
    patch.add_argument("--max_pairs_per_group", type=int, default=16)
    patch.add_argument("--directed_pairs", nargs="+", default=None, help="Optional filters like sd_vae->eq_vae.")
    patch.add_argument("--active_quantile", type=float, default=0.95)
    patch.add_argument("--clamp_quantile", type=float, default=0.99)
    patch.add_argument("--min_active_tokens", type=int, default=32)
    patch.set_defaults(func=run_activation_patch)

    group = sub.add_parser("group-features")
    _add_common_args(group)
    group.add_argument("--match_csv", type=Path, required=True)
    group.add_argument("--min_group_score", type=float, default=0.65)
    group.add_argument("--min_group_activation_corr", type=float, default=0.40)
    group.add_argument("--min_group_class_jaccard", type=float, default=0.30)
    group.set_defaults(func=run_group_features)

    group_patch = sub.add_parser("group-activation-patch")
    _add_common_args(group_patch)
    group_patch.add_argument("--group_members_csv", type=Path, required=True)
    group_patch.add_argument("--feature_bank", type=Path, default=None)
    group_patch.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    group_patch.add_argument("--max_fit_tokens", type=int, default=50_000)
    group_patch.add_argument("--max_eval_tokens", type=int, default=8192)
    group_patch.add_argument("--ridge_alpha", type=float, default=10.0)
    group_patch.add_argument("--group_ridge_alpha", type=float, default=10.0)
    group_patch.add_argument("--sae_batch_size", type=int, default=512)
    group_patch.add_argument("--device", type=str, default=None)
    group_patch.add_argument("--group_ids", nargs="+", default=None)
    group_patch.add_argument("--directed_pairs", nargs="+", default=None, help="Optional filters like sd_vae->eq_vae.")
    group_patch.add_argument("--max_groups", type=int, default=None)
    group_patch.add_argument("--max_source_features", type=int, default=8)
    group_patch.add_argument("--max_target_features", type=int, default=8)
    group_patch.add_argument("--min_group_conditions", type=int, default=2)
    group_patch.add_argument("--active_quantile", type=float, default=0.95)
    group_patch.add_argument("--clamp_quantile", type=float, default=0.99)
    group_patch.add_argument("--min_active_tokens", type=int, default=32)
    group_patch.set_defaults(func=run_group_activation_patch)

    sampling = sub.add_parser("summarize-sampling")
    sampling.add_argument("--sampling_csv", type=Path, required=True)
    sampling.add_argument("--task_tsv", type=Path, required=True)
    sampling.add_argument("--target_csv", type=Path, default=None)
    sampling.add_argument("--out_dir", type=Path, default=None)
    sampling.set_defaults(func=run_summarize_sampling)

    group_summary = sub.add_parser("summarize-group-patching")
    group_summary.add_argument("--run_dirs", nargs="*", type=Path, default=[])
    group_summary.add_argument("--patch_csvs", nargs="*", type=Path, default=[])
    group_summary.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    group_summary.add_argument("--run_id", type=str, default=None)
    group_summary.add_argument("--resume", action="store_true")
    group_summary.set_defaults(func=run_summarize_group_patching)
    return p


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--model_variant", type=str, default="sit_b_2",
                   help="SiT variant id or model name; used for auto layers and namespaced roots.")
    p.add_argument("--layers", nargs="+", default=["auto"])
    p.add_argument("--t_bins", nargs="+", type=int, default=list(T_BINS))
    p.add_argument("--cells", nargs="+", default=None)
    p.add_argument("--dit_step", type=int, default=200_000)
    p.add_argument("--activations_root", type=Path, default=None)
    p.add_argument("--out_root", type=Path, default=None)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_images", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "model_variant"):
        spec = model_variant_spec(args.model_variant)
        args.model_variant = spec.variant_id
        args.layers = parse_layers(args.layers, spec)
        if args.activations_root is None:
            namespaced = model_subdir(SCRATCH_PROJECT_ROOT, spec.variant_id, "activations_ynull")
            args.activations_root = (
                namespaced if namespaced.exists() or spec.variant_id != "sit_b_2" else ACTIVATIONS_YNULL
            )
        if args.out_root is None:
            args.out_root = model_subdir(OUTPUT_ROOT, spec.variant_id, "patching")
    if hasattr(args, "t_bins") and any(t not in T_CENTERS for t in args.t_bins):
        parser.error(f"--t_bins must be drawn from {sorted(T_CENTERS)}")
    return int(args.func(args))
