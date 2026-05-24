#!/bin/bash
#SBATCH --job-name=dmi_recompute_hist
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Phase 4.5b — post-hoc activation histograms for one dashboard cell.
#
# Required env:
#   CELL_DIR=outputs/phase4_5b_feature_viz_ynull/<cond>_L<L>_T<T>
#   CONDITION=<sd_vae|repa_e|eq_vae>
#   LAYER=<3|6|9>
#   T_BIN=<0|1|2>
# Optional:
#   DIT_STEP=200000
#   ACT_ROOT=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_ynull
#   SAE_ROOT=/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k
#   BATCH_IMAGES=64
#   LIMIT=<int>       # smoke testing
#   FORCE=1           # overwrite existing activation_histogram entries

CELL_DIR="${CELL_DIR:?must export CELL_DIR before sbatch}"
CONDITION="${CONDITION:?must export CONDITION}"
LAYER="${LAYER:?must export LAYER}"
T_BIN="${T_BIN:?must export T_BIN}"
DIT_STEP="${DIT_STEP:-200000}"
ACT_ROOT="${ACT_ROOT:-/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_ynull}"
SAE_ROOT="${SAE_ROOT:-/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k}"
BATCH_IMAGES="${BATCH_IMAGES:-64}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

export HF_HOME="/leonardo_scratch/fast/IscrC_YENDRI/lcerovaz/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
echo "=== recompute_histograms CELL_DIR=$CELL_DIR  L=$LAYER T=$T_BIN ==="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

EXTRA=()
if [[ -n "${LIMIT:-}" ]]; then EXTRA+=(--limit "$LIMIT"); fi
if [[ "${FORCE:-0}" == "1" ]]; then EXTRA+=(--force); fi

uv run python scripts/analysis/recompute_histograms.py \
    --cell_dir "$CELL_DIR" \
    --condition "$CONDITION" \
    --layer "$LAYER" \
    --t_bin "$T_BIN" \
    --dit_step "$DIT_STEP" \
    --sae_root "$SAE_ROOT" \
    --act_root "$ACT_ROOT" \
    --batch_images "$BATCH_IMAGES" \
    "${EXTRA[@]}"

echo
echo "=== recompute_histograms done: $(date) ==="
