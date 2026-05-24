#!/bin/bash
#SBATCH --job-name=dmi_featpatch
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Phase 4.11 feature-level SAE activation patching.
#
# ANALYSIS:
#   build-bank             dashboard JSON -> metrics/feature_bank.csv
#   match                  feature_bank.csv -> feature_match_candidates/hungarian.csv
#   activation-patch       activation-space feature replacement controls
#   group-features         Hungarian matches -> feature family components
#   group-activation-patch feature-family group replacement controls
#   sampling-feature-fid   sampling-time feature patching FID

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
ANALYSIS="${ANALYSIS:?export ANALYSIS before sbatch}"
CONDITIONS="${CONDITIONS:-sd_vae repa_e eq_vae}"
LAYERS="${LAYERS:-3 6 9}"
T_BINS="${T_BINS:-0 1 2}"
DIT_STEP="${DIT_STEP:-200000}"
RUN_ID="${RUN_ID:-${ANALYSIS}_${SLURM_JOB_ID}}"
OUT_ROOT="${OUT_ROOT:-outputs/phase4_11_feature_activation_patching}"

module load cuda/12.2 || true
cd "$REPO"

echo "=== feature_activation_patching ANALYSIS=$ANALYSIS RUN_ID=$RUN_ID ==="
echo "CONDITIONS=$CONDITIONS LAYERS=$LAYERS T_BINS=$T_BINS DIT_STEP=$DIT_STEP"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

CELL_ARGS=()
if [[ -n "${CELLS:-}" ]]; then
    CELL_ARGS+=("--cells")
    # shellcheck disable=SC2206
    _CELLS_ARRAY=($CELLS)
    CELL_ARGS+=("${_CELLS_ARRAY[@]}")
    echo "CELLS=$CELLS"
fi

RESUME_ARGS=()
if [[ "${RESUME:-false}" == "true" ]]; then
    RESUME_ARGS+=("--resume")
fi

if [[ "$ANALYSIS" == "build-bank" ]]; then
    EXTRA=()
    if [[ "${INCLUDE_ALL_LIVE:-false}" == "true" ]]; then
        EXTRA+=("--include_all_live")
    fi
    if [[ "${WITH_DECODER_NORMS:-false}" == "true" ]]; then
        EXTRA+=("--with_decoder_norms" "--device" "${DEVICE:-cuda}")
    fi
    if [[ -n "${ATLAS_CSV:-}" ]]; then
        EXTRA+=("--atlas_csv" "$ATLAS_CSV")
    fi
    if [[ -n "${SAE_ROOT:-}" ]]; then
        EXTRA+=("--sae_root" "$SAE_ROOT")
    fi
    if [[ "${HYDRATE_DASHBOARD_JSON:-false}" == "true" ]]; then
        EXTRA+=("--hydrate_dashboard_json")
    fi
    uv run python scripts/analysis/sae_feature_patching.py build-bank \
        --conditions $CONDITIONS \
        --layers $LAYERS \
        --t_bins $T_BINS \
        --dit_step "$DIT_STEP" \
        --out_root "$OUT_ROOT" \
        --run_id "$RUN_ID" \
        --min_density "${MIN_DENSITY:-0.0001}" \
        --max_density "${MAX_DENSITY:-0.10}" \
        --max_entropy "${MAX_ENTROPY:-2.5}" \
        --max_features_per_cell "${MAX_FEATURES_PER_CELL:-0}" \
        "${EXTRA[@]}" \
        "${RESUME_ARGS[@]}" \
        "${CELL_ARGS[@]}"
elif [[ "$ANALYSIS" == "match" ]]; then
    FEATURE_BANK="${FEATURE_BANK:?export FEATURE_BANK for match}"
    EXTRA=()
    if [[ "${ALLOW_CROSS_CLASS:-false}" == "true" ]]; then
        EXTRA+=("--allow_cross_class")
    fi
    if [[ "${METADATA_ONLY:-false}" == "true" ]]; then
        EXTRA+=("--metadata_only")
    fi
    if [[ "${CACHE_PROFILES:-false}" == "true" ]]; then
        EXTRA+=("--cache_profiles")
    fi
    if [[ -n "${SAE_ROOT:-}" ]]; then
        EXTRA+=("--sae_root" "$SAE_ROOT")
    fi
    uv run python scripts/analysis/sae_feature_patching.py match \
        --conditions $CONDITIONS \
        --layers $LAYERS \
        --t_bins $T_BINS \
        --dit_step "$DIT_STEP" \
        --feature_bank "$FEATURE_BANK" \
        --out_root "$OUT_ROOT" \
        --run_id "$RUN_ID" \
        --min_candidate_score "${MIN_CANDIDATE_SCORE:-0.25}" \
        --min_hungarian_score "${MIN_HUNGARIAN_SCORE:-0.35}" \
        --max_images "${MAX_IMAGES:-512}" \
        --max_fit_tokens "${MAX_FIT_TOKENS:-8192}" \
        --profile_batch_images "${PROFILE_BATCH_IMAGES:-2}" \
        --device "${DEVICE:-cuda}" \
        "${EXTRA[@]}" \
        "${RESUME_ARGS[@]}" \
        "${CELL_ARGS[@]}"
elif [[ "$ANALYSIS" == "activation-patch" ]]; then
    MATCH_CSV="${MATCH_CSV:?export MATCH_CSV for activation-patch}"
    PAIR_ARGS=()
    if [[ -n "${DIRECTED_PAIRS:-}" ]]; then
        PAIR_ARGS+=("--directed_pairs")
        # shellcheck disable=SC2206
        _PAIR_ARRAY=($DIRECTED_PAIRS)
        PAIR_ARGS+=("${_PAIR_ARRAY[@]}")
    fi
    if [[ -n "${FEATURE_BANK:-}" ]]; then
        PAIR_ARGS+=("--feature_bank" "$FEATURE_BANK")
    fi
    if [[ -n "${SAE_ROOT:-}" ]]; then
        PAIR_ARGS+=("--sae_root" "$SAE_ROOT")
    fi
    uv run python scripts/analysis/sae_feature_patching.py activation-patch \
        --conditions $CONDITIONS \
        --layers $LAYERS \
        --t_bins $T_BINS \
        --dit_step "$DIT_STEP" \
        --match_csv "$MATCH_CSV" \
        --out_root "$OUT_ROOT" \
        --run_id "$RUN_ID" \
        --max_images "${MAX_IMAGES:-512}" \
        --max_fit_tokens "${MAX_FIT_TOKENS:-50000}" \
        --max_eval_tokens "${MAX_EVAL_TOKENS:-8192}" \
        --max_pairs "${MAX_PAIRS:-0}" \
        --max_pairs_per_group "${MAX_PAIRS_PER_GROUP:-16}" \
        --sae_batch_size "${SAE_BATCH_SIZE:-512}" \
        --active_quantile "${ACTIVE_QUANTILE:-0.95}" \
        --clamp_quantile "${CLAMP_QUANTILE:-0.99}" \
        --min_active_tokens "${MIN_ACTIVE_TOKENS:-32}" \
        --device "${DEVICE:-cuda}" \
        "${PAIR_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        "${CELL_ARGS[@]}"
elif [[ "$ANALYSIS" == "group-features" ]]; then
    MATCH_CSV="${MATCH_CSV:?export MATCH_CSV for group-features}"
    uv run python scripts/analysis/sae_feature_patching.py group-features \
        --conditions $CONDITIONS \
        --layers $LAYERS \
        --t_bins $T_BINS \
        --dit_step "$DIT_STEP" \
        --match_csv "$MATCH_CSV" \
        --out_root "$OUT_ROOT" \
        --run_id "$RUN_ID" \
        --min_group_score "${MIN_GROUP_SCORE:-0.65}" \
        --min_group_activation_corr "${MIN_GROUP_ACTIVATION_CORR:-0.40}" \
        --min_group_class_jaccard "${MIN_GROUP_CLASS_JACCARD:-0.30}" \
        "${RESUME_ARGS[@]}" \
        "${CELL_ARGS[@]}"
elif [[ "$ANALYSIS" == "group-activation-patch" ]]; then
    GROUP_MEMBERS_CSV="${GROUP_MEMBERS_CSV:?export GROUP_MEMBERS_CSV for group-activation-patch}"
    GROUP_ARGS=()
    if [[ -n "${DIRECTED_PAIRS:-}" ]]; then
        GROUP_ARGS+=("--directed_pairs")
        # shellcheck disable=SC2206
        _PAIR_ARRAY=($DIRECTED_PAIRS)
        GROUP_ARGS+=("${_PAIR_ARRAY[@]}")
    fi
    if [[ -n "${GROUP_IDS:-}" ]]; then
        GROUP_ARGS+=("--group_ids")
        # shellcheck disable=SC2206
        _GROUP_ARRAY=($GROUP_IDS)
        GROUP_ARGS+=("${_GROUP_ARRAY[@]}")
    fi
    if [[ -n "${FEATURE_BANK:-}" ]]; then
        GROUP_ARGS+=("--feature_bank" "$FEATURE_BANK")
    fi
    if [[ -n "${SAE_ROOT:-}" ]]; then
        GROUP_ARGS+=("--sae_root" "$SAE_ROOT")
    fi
    uv run python scripts/analysis/sae_feature_patching.py group-activation-patch \
        --conditions $CONDITIONS \
        --layers $LAYERS \
        --t_bins $T_BINS \
        --dit_step "$DIT_STEP" \
        --group_members_csv "$GROUP_MEMBERS_CSV" \
        --out_root "$OUT_ROOT" \
        --run_id "$RUN_ID" \
        --max_images "${MAX_IMAGES:-512}" \
        --max_fit_tokens "${MAX_FIT_TOKENS:-50000}" \
        --max_eval_tokens "${MAX_EVAL_TOKENS:-8192}" \
        --max_groups "${MAX_GROUPS:-0}" \
        --max_source_features "${MAX_SOURCE_FEATURES:-8}" \
        --max_target_features "${MAX_TARGET_FEATURES:-8}" \
        --min_group_conditions "${MIN_GROUP_CONDITIONS:-2}" \
        --sae_batch_size "${SAE_BATCH_SIZE:-512}" \
        --ridge_alpha "${RIDGE_ALPHA:-10.0}" \
        --group_ridge_alpha "${GROUP_RIDGE_ALPHA:-10.0}" \
        --active_quantile "${ACTIVE_QUANTILE:-0.95}" \
        --clamp_quantile "${CLAMP_QUANTILE:-0.99}" \
        --min_active_tokens "${MIN_ACTIVE_TOKENS:-32}" \
        --device "${DEVICE:-cuda}" \
        "${GROUP_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        "${CELL_ARGS[@]}"
elif [[ "$ANALYSIS" == "sampling-feature-fid" ]]; then
    TARGET_RUN="${TARGET_RUN:?export TARGET_RUN for sampling-feature-fid}"
    TARGET_ADAPTER="${TARGET_ADAPTER:?export TARGET_ADAPTER for sampling-feature-fid}"
    TARGET_CONDITION="${TARGET_CONDITION:?export TARGET_CONDITION for sampling-feature-fid}"
    MODE="${MODE:?export MODE=baseline|transfer_replace|native_ablate|native_clamp|random_matched_control|wrong_window_control}"
    EXTRA=()
    if [[ "$MODE" != "baseline" ]]; then
        EXTRA+=("--layer" "${LAYER:?export LAYER}")
        EXTRA+=("--t_bin" "${T_BIN:?export T_BIN}")
        EXTRA+=("--target_feature_id" "${TARGET_FEATURE_ID:?export TARGET_FEATURE_ID}")
    fi
    if [[ "$MODE" == "matched_patch" || "$MODE" == "transfer_replace" || "$MODE" == "random_patch" || "$MODE" == "random_matched_control" || "$MODE" == "wrong_window_control" ]]; then
        EXTRA+=("--source_condition" "${SOURCE_CONDITION:?export SOURCE_CONDITION}")
    fi
    if [[ "$MODE" == "matched_patch" || "$MODE" == "transfer_replace" || "$MODE" == "wrong_window_control" ]]; then
        EXTRA+=("--source_feature_id" "${SOURCE_FEATURE_ID:?export SOURCE_FEATURE_ID}")
    fi
    if [[ -n "${SOURCE_MEAN_ACT:-}" ]]; then
        EXTRA+=("--source_mean_act" "$SOURCE_MEAN_ACT")
    fi
    if [[ -n "${TARGET_MEAN_ACT:-}" ]]; then
        EXTRA+=("--target_mean_act" "$TARGET_MEAN_ACT")
    fi
    if [[ -n "${SOURCE_TO_TARGET_SCALE:-}" ]]; then
        EXTRA+=("--source_to_target_scale" "$SOURCE_TO_TARGET_SCALE")
    fi
    if [[ -n "${RANDOM_SOURCE_FEATURE_ID:-}" ]]; then
        EXTRA+=("--random_source_feature_id" "$RANDOM_SOURCE_FEATURE_ID")
    fi
    if [[ -n "${SOURCE_TO_TARGET_BIAS:-}" ]]; then
        EXTRA+=("--source_to_target_bias" "$SOURCE_TO_TARGET_BIAS")
    fi
    if [[ -n "${TARGET_CLAMP_VALUE:-}" ]]; then
        EXTRA+=("--target_clamp_value" "$TARGET_CLAMP_VALUE")
    fi
    if [[ -n "${CLAMP_QUANTILE:-}" ]]; then
        EXTRA+=("--clamp_quantile" "$CLAMP_QUANTILE")
    fi
    if [[ -n "${WRONG_T_BIN:-}" ]]; then
        EXTRA+=("--wrong_t_bin" "$WRONG_T_BIN")
    fi
    if [[ "${NORMALIZE:-true}" == "false" ]]; then
        EXTRA+=("--no_normalize")
    fi
    if [[ -n "${SAE_ROOT:-}" ]]; then
        EXTRA+=("--sae_root" "$SAE_ROOT")
    fi
    if [[ -n "${SLOPE_GATE_EPSILON:-}" ]]; then
        EXTRA+=("--slope_gate_epsilon" "$SLOPE_GATE_EPSILON")
    fi
    echo "=== DEBUG SAE_ROOT env=${SAE_ROOT:-<unset>} ==="
    echo "=== DEBUG ls SAE_ROOT/eq_vae/L9_T2/step_200000 ==="
    ls -la "${SAE_ROOT:-/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k}/eq_vae/L9_T2/step_200000/" 2>&1 | head -5
    echo "=== DEBUG EXTRA argv ==="
    printf '  %s\n' "${EXTRA[@]}"
    uv run python scripts/eval/cross_tokenizer_feature_patching.py \
        --target_run "$TARGET_RUN" \
        --target_adapter "$TARGET_ADAPTER" \
        --target_condition "$TARGET_CONDITION" \
        --mode "$MODE" \
        --dit_step "$DIT_STEP" \
        --out_root "$OUT_ROOT/$RUN_ID" \
        --n_samples "${N_SAMPLES:-1000}" \
        --batch_size "${BATCH_SIZE:-32}" \
        --sampler "${SAMPLER:-ode}" \
        --sample_steps "${SAMPLE_STEPS:-250}" \
        --cfg "${CFG:-1.5}" \
        --seed "${SEED:-0}" \
        --t_tol "${T_TOL:-0.01}" \
        --fit_max_images "${FIT_MAX_IMAGES:-512}" \
        --fit_max_tokens "${FIT_MAX_TOKENS:-50000}" \
        --ridge_alpha "${RIDGE_ALPHA:-10.0}" \
        "${EXTRA[@]}"
else
    echo "ERROR: unknown ANALYSIS=$ANALYSIS" >&2
    exit 2
fi

echo "=== feature_activation_patching done: $(date) ==="
