#!/bin/bash
#SBATCH --job-name=dmi_latent_sae
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err
set -euo pipefail

# Phase 4.19 (E30): Matryoshka SAEs on tokenizer latents — EF ablation grid.
# Submit as an array over the (condition x d_sae x k) cross-product:
#   RUN_ID=latent_sae_ef_<ts> sbatch --array=0-23 slurm/train_latent_sae.slurm
# Required env:
#   RUN_ID — shared run id so every task writes into the same root.
# Optional env:
#   CONDITIONS / D_SAE_LIST / K_LIST — space-separated grid axes.
#   TOTAL_SAMPLES / BATCH_SIZE / SEED / PATCH_SIZE.
CONDITIONS=(${CONDITIONS:-sd_vae eq_vae repa_e})
D_SAE_LIST=(${D_SAE_LIST:-256 768 2048 8192})
K_LIST=(${K_LIST:-8 16})
VARIANTS=(${VARIANTS:-matryoshka})
TOTAL_SAMPLES="${TOTAL_SAMPLES:-81920000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SEED="${SEED:-0}"
PATCH_SIZE="${PATCH_SIZE:-2}"
# Extra argv forwarded verbatim to the driver (e.g. EXTRA_ARGS="--smoke").
EXTRA_ARGS="${EXTRA_ARGS:-}"
: "${RUN_ID:?RUN_ID must be set (shared root for all array tasks)}"

N_D=${#D_SAE_LIST[@]}
N_K=${#K_LIST[@]}
N_V=${#VARIANTS[@]}
IDX="${SLURM_ARRAY_TASK_ID:?run as an sbatch array}"
COND="${CONDITIONS[$((IDX / (N_D * N_K * N_V)))]}"
REST=$((IDX % (N_D * N_K * N_V)))
D_SAE="${D_SAE_LIST[$((REST / (N_K * N_V)))]}"
REST=$((REST % (N_K * N_V)))
K="${K_LIST[$((REST / N_V))]}"
VARIANT="${VARIANTS[$((REST % N_V))]}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

echo "=== train_latent_sae task $IDX → $COND/d${D_SAE}_k${K}/${VARIANT} on $(hostname) $(date) ==="
if ! nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | grep -q "MiB"; then
    echo "ERROR: nvidia-smi sees no GPU on $(hostname). Re-submit; node unhealthy." >&2
    exit 6
fi

export WANDB_ENTITY="${WANDB_ENTITY:-ar_spectra}"
export WANDB_PROJECT="${WANDB_PROJECT:-diffmechint}"
export http_proxy="${http_proxy:-http://login01:3133}"
export https_proxy="${https_proxy:-http://login01:3133}"

cd "$REPO"

uv run python scripts/training/train_latent_sae.py \
    --conditions "$COND" \
    --d_sae_list "$D_SAE" \
    --k_list "$K" \
    --variants "$VARIANT" \
    --run_id "$RUN_ID" \
    --total_samples "$TOTAL_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --patch_size "$PATCH_SIZE" \
    $EXTRA_ARGS

echo "=== task $IDX done: $(date) ==="
