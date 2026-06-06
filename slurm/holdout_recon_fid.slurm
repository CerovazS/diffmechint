#!/bin/bash
#SBATCH --job-name=dmi_holdout_recon_fid
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

RUN="${RUN:?export RUN=/absolute/path/to/run_dir}"
ADAPTER="${ADAPTER:?export ADAPTER=sd_vae|repa_e|eq_vae}"
MODEL_NAME="${MODEL_NAME:-SiT-L/2}"
NORMALIZE="${NORMALIZE:-true}"
N_REF="${N_REF:-5000}"
N_SAMPLES="${N_SAMPLES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-64}"
STEPS="${STEPS:-}"
OUT_ROOT="${OUT_ROOT:-/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/by_model/sit_l_2/analysis/holdout_recon_fid}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
FAST_BASE=/leonardo_scratch/fast/IscrC_YENDRI

module load cuda/12.2 || true

INCEPTION_HOME="${HOME}/.cache/cleanfid_models/inception-2015-12-05.pt"
ln -sf "${INCEPTION_HOME}" /tmp/inception-2015-12-05.pt
echo "Inception symlinked: $(ls -lh /tmp/inception-2015-12-05.pt)"

export HF_HOME="${FAST_BASE}/lcerovaz/hf_cache"
export HF_HUB_CACHE="${FAST_BASE}/lcerovaz/hf_cache/hub"
export TRANSFORMERS_CACHE="${FAST_BASE}/lcerovaz/hf_cache/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
echo "=== holdout_recon_fid ${ADAPTER} start: $(date) ==="
echo "RUN=${RUN}"
echo "MODEL_NAME=${MODEL_NAME} N_REF=${N_REF} N_SAMPLES=${N_SAMPLES} NORMALIZE=${NORMALIZE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

ARGS=(
    "$RUN"
    "$ADAPTER"
    --model_name "$MODEL_NAME"
    --out_root "$OUT_ROOT"
    --n_ref "$N_REF"
    --n_samples "$N_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --reference_batch_size "$REFERENCE_BATCH_SIZE"
)

if [[ -n "$STEPS" ]]; then
    ARGS+=(--steps "$STEPS")
fi
if [[ "$NORMALIZE" == "false" ]]; then
    ARGS+=(--no_normalize)
fi

uv run python scripts/eval/holdout_reconstruction_fid.py "${ARGS[@]}"

echo
echo "=== holdout_recon_fid ${ADAPTER} done: $(date) ==="
echo "--- holdout_recon_fid.csv ---"
cat "$RUN/metrics/validation/holdout_recon_fid.csv"
