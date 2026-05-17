#!/bin/bash
#SBATCH --job-name=dmi_train_sae
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Optional env vars:
#   CONDITIONS=<space-separated, default "sd_vae repa_e eq_vae">
#   LAYERS / T_BINS: idem
#   BASE_SAMPLES / WARM_SAMPLES: per-stage token budget
#   ONLY: single-chain smoke, format "sd_vae/L6/T1"
CONDITIONS="${CONDITIONS:-sd_vae repa_e eq_vae}"
LAYERS="${LAYERS:-3 6 9}"
T_BINS="${T_BINS:-0 1 2}"
BASE_SAMPLES="${BASE_SAMPLES:-20000000}"
WARM_SAMPLES="${WARM_SAMPLES:-5000000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
ONLY="${ONLY:-}"
D_SAE="${D_SAE:-16384}"
K="${K:-32}"
LR_WARMUP="${LR_WARMUP:-200}"
WARM_LR_WARMUP="${WARM_LR_WARMUP:-0}"
# WARM_MODE: cold (default, canonical) | weights_only (diagnostic baseline reproducing
# the buggy Xu et al. 2412.17626 setup — 94% dead features, do NOT use for production).
# The earlier 'weights_and_optimizer' mode (Fix B) was removed: it didn't fix the issue
# and Xu et al. don't actually do it (they only transfer SAE weights).
WARM_MODE="${WARM_MODE:-cold}"
VARIANT_TAG="${VARIANT_TAG:-}"
OUT_ROOT="${OUT_ROOT:-/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae}"

REPO=/leonardo/home/userexternal/lcerovaz/diffmechint
module load cuda/12.2 || true

# CINECA boost nodes occasionally come up with no visible GPU — fail loud.
echo "=== train_sae start on $(hostname) job ${SLURM_JOB_ID:-local} $(date) ==="
if ! nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | grep -q "MiB"; then
    echo "ERROR: nvidia-smi sees no GPU on $(hostname). Re-submit; node unhealthy." >&2
    exit 6
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# wandb: ~/.netrc is sourced automatically; entity = ar_spectra by default.
export WANDB_ENTITY="${WANDB_ENTITY:-ar_spectra}"
export WANDB_PROJECT="${WANDB_PROJECT:-diffmechint}"
# Compute nodes have no internet by default; route via the squid proxy so
# wandb can upload logs (without this, wandb falls back to offline mode).
export http_proxy="${http_proxy:-http://login01:3133}"
export https_proxy="${https_proxy:-http://login01:3133}"

cd "$REPO"

EXTRA_ARGS=()
if [[ -n "$ONLY" ]]; then
    EXTRA_ARGS+=("--only" "$ONLY")
    echo "ONLY=$ONLY → single-chain smoke"
fi

uv run python scripts/train_sae.py \
    --conditions $CONDITIONS \
    --layers $LAYERS \
    --t_bins $T_BINS \
    --d_sae "$D_SAE" \
    --k "$K" \
    --lr_warm_up_steps "$LR_WARMUP" \
    --warm_lr_warm_up_steps "$WARM_LR_WARMUP" \
    --warm_mode "$WARM_MODE" \
    --variant_tag "$VARIANT_TAG" \
    --base_total_samples "$BASE_SAMPLES" \
    --warm_total_samples "$WARM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --out_root "$OUT_ROOT" \
    "${EXTRA_ARGS[@]}"

echo
echo "=== train_sae done: $(date) ==="
