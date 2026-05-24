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
#   SEED: optional replicate seed
CONDITIONS="${CONDITIONS:-sd_vae repa_e eq_vae}"
LAYERS="${LAYERS:-3 6 9}"
T_BINS="${T_BINS:-0 1 2}"
BASE_SAMPLES="${BASE_SAMPLES:-20000000}"
WARM_SAMPLES="${WARM_SAMPLES:-5000000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
ONLY="${ONLY:-}"
D_SAE="${D_SAE:-16384}"
K="${K:-32}"
VARIANT="${VARIANT:-topk}"
# Space-separated cumulative latent prefix widths. Used iff VARIANT=matryoshka.
# Example: MATRYOSHKA_WIDTHS="4096 8192 16384 32768" with K=256 d_sae=32768.
MATRYOSHKA_WIDTHS="${MATRYOSHKA_WIDTHS:-}"
LR_WARMUP="${LR_WARMUP:-200}"
WARM_LR_WARMUP="${WARM_LR_WARMUP:-0}"
# WARM_MODE: cold (default, canonical) | weights_only (diagnostic baseline reproducing
# the buggy Xu et al. 2412.17626 setup — 94% dead features, do NOT use for production).
# The earlier 'weights_and_optimizer' mode (Fix B) was removed: it didn't fix the issue
# and Xu et al. don't actually do it (they only transfer SAE weights).
WARM_MODE="${WARM_MODE:-cold}"
VARIANT_TAG="${VARIANT_TAG:-}"
OUT_ROOT="${OUT_ROOT:-/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae}"
# Activations source — default 7/class; set to ".../activations_s20" for the 20/class dataset.
ACTIVATIONS_ROOT="${ACTIVATIONS_ROOT:-/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations}"
# Held-out validation activations (same layout). Set NO_VAL=1 to disable inline eval.
VAL_ACTIVATIONS_ROOT="${VAL_ACTIVATIONS_ROOT:-/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_val}"
VAL_EVERY_N_STEPS="${VAL_EVERY_N_STEPS:-1500}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-200000}"
NO_VAL="${NO_VAL:-0}"
SEED="${SEED:-}"
FAIL_IF_CHAIN_EXISTS="${FAIL_IF_CHAIN_EXISTS:-0}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

# CINECA boost nodes occasionally come up with no visible GPU — fail loud.
echo "=== train_sae start on $(hostname) job ${SLURM_JOB_ID:-local} $(date) ==="
if ! nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | grep -q "MiB"; then
    echo "ERROR: nvidia-smi sees no GPU on $(hostname). Re-submit; node unhealthy." >&2
    exit 6
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv

if ! ./.venv/bin/python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(1)
print(torch.cuda.get_device_name(0))
PY
then
    echo "ERROR: torch.cuda.is_available() is false on $(hostname). Re-submit; node unhealthy." >&2
    exit 7
fi

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
if [[ "$NO_VAL" == "1" ]]; then
    EXTRA_ARGS+=("--no_val")
    echo "NO_VAL=1 → inline validation disabled"
fi
if [[ -n "$SEED" ]]; then
    EXTRA_ARGS+=("--seed" "$SEED")
    echo "SEED=$SEED"
fi
if [[ "$FAIL_IF_CHAIN_EXISTS" == "1" ]]; then
    EXTRA_ARGS+=("--fail_if_chain_exists")
    echo "FAIL_IF_CHAIN_EXISTS=1 → refuse non-empty chain outputs"
fi
if [[ -n "$MATRYOSHKA_WIDTHS" ]]; then
    # word-split intentionally — pass widths as separate argv tokens.
    EXTRA_ARGS+=("--matryoshka_widths")
    for w in $MATRYOSHKA_WIDTHS; do EXTRA_ARGS+=("$w"); done
    echo "MATRYOSHKA_WIDTHS=$MATRYOSHKA_WIDTHS"
fi

uv run python scripts/training/train_sae.py \
    --conditions $CONDITIONS \
    --layers $LAYERS \
    --t_bins $T_BINS \
    --variant "$VARIANT" \
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
    --activations_root "$ACTIVATIONS_ROOT" \
    --val_activations_root "$VAL_ACTIVATIONS_ROOT" \
    --val_every_n_steps "$VAL_EVERY_N_STEPS" \
    --val_max_tokens "$VAL_MAX_TOKENS" \
    "${EXTRA_ARGS[@]}"

echo
echo "=== train_sae done: $(date) ==="
