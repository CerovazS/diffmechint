#!/bin/bash
#SBATCH --job-name=dmi_feature_dashboard
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Phase 4.5b — per-feature visualisation dashboard for one SAE cell.
#
# Required env (export before sbatch):
#   CONDITION=<sd_vae | repa_e | eq_vae>
#   LAYER=<3|6|9>
#   T_BIN=<0|1|2>
# Optional:
#   MODEL_VARIANT=sit_b_2
#   DIT_STEP=200000
#   OUT_ROOT=outputs/phase4_5b_feature_viz
#   SAE_ROOT=/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k
#   BATCH_IMAGES=64
#   LIMIT_IMAGES=<int>  # smoke testing only
#   SKIP_THUMBNAILS=1  # analysis-only; avoid flaky ImageNet JPEG thumbnail reads

CONDITION="${CONDITION:?must export CONDITION before sbatch}"
LAYER="${LAYER:?must export LAYER before sbatch}"
T_BIN="${T_BIN:?must export T_BIN before sbatch}"
MODEL_VARIANT="${MODEL_VARIANT:-sit_b_2}"
DIT_STEP="${DIT_STEP:-200000}"
OUT_ROOT="${OUT_ROOT:-outputs/phase4_5b_feature_viz}"
SAE_ROOT="${SAE_ROOT:-/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k}"
# Activation source. Default is the canonical y_true val tree; override to
# `.../activations_ynull` for y-null counterfactual dashboards (Phase 4.7).
ACT_ROOT="${ACT_ROOT:-/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_val}"
BATCH_IMAGES="${BATCH_IMAGES:-64}"
LIMIT_IMAGES="${LIMIT_IMAGES:-}"
SKIP_THUMBNAILS="${SKIP_THUMBNAILS:-0}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

export HF_HOME="/leonardo_scratch/fast/IscrC_YENDRI/lcerovaz/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
echo "=== feature_dashboard CONDITION=$CONDITION MODEL_VARIANT=$MODEL_VARIANT L=$LAYER T=$T_BIN DIT_STEP=$DIT_STEP ==="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

EXTRA=()
if [[ -n "$LIMIT_IMAGES" ]]; then EXTRA+=(--limit_images "$LIMIT_IMAGES"); fi

uv run python scripts/analysis/feature_dashboard.py \
    --condition "$CONDITION" \
    --model_variant "$MODEL_VARIANT" \
    --layer "$LAYER" \
    --t_bin "$T_BIN" \
    --dit_step "$DIT_STEP" \
    --sae_root "$SAE_ROOT" \
    --act_root "$ACT_ROOT" \
    --out_root "$OUT_ROOT" \
    --batch_images "$BATCH_IMAGES" \
    "${EXTRA[@]}" \
    $([[ "$SKIP_THUMBNAILS" == "1" ]] && printf '%s' '--skip_thumbnails')

echo
echo "=== feature_dashboard done: $(date) ==="
