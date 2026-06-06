#!/bin/bash
#SBATCH --job-name=dmi_tokdict
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Phase 4.10 tokenizer dictionary validation.
#
# ANALYSIS:
#   probe-transfer       residual probe transfer + ridge / Procrustes alignment
#   residual-similarity  feature-free residual CKA/RSA/kNN
#   activation-proxy     target residual → source SAE → target proxy reconstruction
#   cross-sae-fid        sampling-time baseline/native/transfer FID
#
# Common optional inputs:
#   CONDITIONS="sd_vae repa_e eq_vae"
#   MODEL_VARIANT=sit_b_2
#   LAYERS="3 6 9"
#   T_BINS="0 1 2"
#   DIT_STEP=200000
#   ACTIVATIONS_ROOT=<optional explicit activation root>
#   LATENTS_ROOT=<optional explicit latent-label root>
#   SAE_ROOT=<optional explicit SAE root for activation-proxy / cross-sae-fid>
#   RUN_ID=<unique name>
#   OUT_ROOT=outputs/phase4_10_tokenizer_dictionary_validation
#   CELLS="L3_T1 L6_T2 L9_T2"  # optional exact cell subset
#   RESUME=true                  # resume an existing RUN_ID and skip completed rows

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
ANALYSIS="${ANALYSIS:?export ANALYSIS before sbatch}"
CONDITIONS="${CONDITIONS:-sd_vae repa_e eq_vae}"
MODEL_VARIANT="${MODEL_VARIANT:-sit_b_2}"
LAYERS="${LAYERS:-3 6 9}"
T_BINS="${T_BINS:-0 1 2}"
DIT_STEP="${DIT_STEP:-200000}"
RUN_ID="${RUN_ID:-${ANALYSIS}_${SLURM_JOB_ID}}"
OUT_ROOT="${OUT_ROOT:-outputs/phase4_10_tokenizer_dictionary_validation}"
SEED="${SEED:-0}"

module load cuda/12.2 || true
cd "$REPO"

echo "=== tokenizer_dictionary_validation ANALYSIS=$ANALYSIS RUN_ID=$RUN_ID ==="
echo "CONDITIONS=$CONDITIONS MODEL_VARIANT=$MODEL_VARIANT LAYERS=$LAYERS T_BINS=$T_BINS DIT_STEP=$DIT_STEP"
echo "OUT_ROOT=$OUT_ROOT"
[[ -n "${ACTIVATIONS_ROOT:-}" ]] && echo "ACTIVATIONS_ROOT=$ACTIVATIONS_ROOT"
[[ -n "${LATENTS_ROOT:-}" ]] && echo "LATENTS_ROOT=$LATENTS_ROOT"
[[ -n "${SAE_ROOT:-}" ]] && echo "SAE_ROOT=$SAE_ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

ROOT_ARGS=()
if [[ -n "${ACTIVATIONS_ROOT:-}" ]]; then
    ROOT_ARGS+=("--activations_root" "$ACTIVATIONS_ROOT")
fi
if [[ -n "${LATENTS_ROOT:-}" ]]; then
    ROOT_ARGS+=("--latents_root" "$LATENTS_ROOT")
fi

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

if [[ "$ANALYSIS" == "probe-transfer" ]]; then
    uv run python scripts/analysis/tokenizer_dictionary_validation.py probe-transfer \
        --conditions $CONDITIONS \
        --model_variant "$MODEL_VARIANT" \
        --layers $LAYERS \
        --t_bins $T_BINS \
        --dit_step "$DIT_STEP" \
        --out_root "$OUT_ROOT" \
        --run_id "$RUN_ID" \
        --max_images "${MAX_IMAGES:-512}" \
        --max_fit_tokens "${MAX_FIT_TOKENS:-50000}" \
        --max_eval_tokens "${MAX_EVAL_TOKENS:-50000}" \
        --ridge_alpha "${RIDGE_ALPHA:-10.0}" \
        --pool "${POOL:-mean}" \
        --probe_model "${PROBE_MODEL:-ridge}" \
        --probe_max_iter "${PROBE_MAX_ITER:-1000}" \
        --max_probe_train "${MAX_PROBE_TRAIN:-50000}" \
        --seed "$SEED" \
        "${ROOT_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        "${CELL_ARGS[@]}"
elif [[ "$ANALYSIS" == "residual-similarity" ]]; then
    PAIR_KIND_ARGS=()
    if [[ -n "${PAIR_KINDS:-}" ]]; then
        PAIR_KIND_ARGS+=("--pair_kinds")
        # shellcheck disable=SC2206
        _PAIR_KINDS_ARRAY=($PAIR_KINDS)
        PAIR_KIND_ARGS+=("${_PAIR_KINDS_ARRAY[@]}")
        echo "PAIR_KINDS=$PAIR_KINDS"
    fi
    uv run python scripts/analysis/tokenizer_dictionary_validation.py residual-similarity \
        --conditions $CONDITIONS \
        --model_variant "$MODEL_VARIANT" \
        --layers $LAYERS \
        --t_bins $T_BINS \
        --dit_step "$DIT_STEP" \
        --out_root "$OUT_ROOT" \
        --run_id "$RUN_ID" \
        --max_images "${MAX_IMAGES:-512}" \
        --max_tokens "${MAX_TOKENS:-50000}" \
        --knn_k "${KNN_K:-10}" \
        --seed "$SEED" \
        "${ROOT_ARGS[@]}" \
        "${PAIR_KIND_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        "${CELL_ARGS[@]}"
elif [[ "$ANALYSIS" == "activation-proxy" ]]; then
    SAE_ARGS=()
    if [[ -n "${SAE_ROOT:-}" ]]; then
        SAE_ARGS+=("--sae_root" "$SAE_ROOT")
    fi
    uv run python scripts/analysis/tokenizer_dictionary_validation.py activation-proxy \
        --conditions $CONDITIONS \
        --model_variant "$MODEL_VARIANT" \
        --layers $LAYERS \
        --t_bins $T_BINS \
        --dit_step "$DIT_STEP" \
        --out_root "$OUT_ROOT" \
        --run_id "$RUN_ID" \
        --max_images "${MAX_IMAGES:-512}" \
        --max_fit_tokens "${MAX_FIT_TOKENS:-50000}" \
        --max_eval_tokens "${MAX_EVAL_TOKENS:-50000}" \
        --ridge_alpha "${RIDGE_ALPHA:-10.0}" \
        --sae_batch_size "${SAE_BATCH_SIZE:-8192}" \
        --seed "$SEED" \
        "${ROOT_ARGS[@]}" \
        "${SAE_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        "${CELL_ARGS[@]}"
elif [[ "$ANALYSIS" == "cross-sae-fid" ]]; then
    TARGET_RUN="${TARGET_RUN:?export TARGET_RUN for cross-sae-fid}"
    TARGET_ADAPTER="${TARGET_ADAPTER:?export TARGET_ADAPTER for cross-sae-fid}"
    TARGET_CONDITION="${TARGET_CONDITION:?export TARGET_CONDITION for cross-sae-fid}"
    MODE="${MODE:?export MODE=baseline|native|transfer for cross-sae-fid}"
    EXTRA=()
    FID_SAE_ARGS=()
    if [[ -n "${SAE_ROOT:-}" ]]; then
        FID_SAE_ARGS+=("--sae_root" "$SAE_ROOT")
    fi
    if [[ "$MODE" != "baseline" ]]; then
        EXTRA+=("--layer" "${LAYER:?export LAYER for native/transfer}")
        EXTRA+=("--t_bin" "${T_BIN:?export T_BIN for native/transfer}")
    fi
    if [[ "$MODE" == "transfer" ]]; then
        EXTRA+=("--source_condition" "${SOURCE_CONDITION:?export SOURCE_CONDITION for transfer}")
    fi
    if [[ "${NORMALIZE:-true}" == "false" ]]; then
        EXTRA+=("--no_normalize")
    fi
    uv run python scripts/eval/cross_tokenizer_sae_substitution_fid.py \
        --target_run "$TARGET_RUN" \
        --target_adapter "$TARGET_ADAPTER" \
        --target_condition "$TARGET_CONDITION" \
        --mode "$MODE" \
        --dit_step "$DIT_STEP" \
        --out_root "$OUT_ROOT/$RUN_ID" \
        "${FID_SAE_ARGS[@]}" \
        --n_samples "${N_SAMPLES:-1000}" \
        --seed "${SEED:-0}" \
        "${EXTRA[@]}"
else
    echo "ERROR: unknown ANALYSIS=$ANALYSIS" >&2
    exit 2
fi

echo "=== tokenizer_dictionary_validation done: $(date) ==="
