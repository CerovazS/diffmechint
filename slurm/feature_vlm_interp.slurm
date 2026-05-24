#!/bin/bash
#SBATCH --job-name=dmi_vlm_interp
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

# Phase 4.5b — VLM auto-interpretation for one dashboard cell.
#
# Required env:
#   CELL_DIR=outputs/phase4_5b_feature_viz_ynull/<cond>_L<L>_T<T>
# Optional:
#   MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
#   LIMIT=<int>       # smoke testing
#   FORCE=1           # re-caption features that already have vlm_interpretation
#   DRY_RUN=1         # print captions but do not write JSONs / vlm_log

CELL_DIR="${CELL_DIR:?must export CELL_DIR before sbatch}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

export HF_HOME="/leonardo_scratch/fast/IscrC_YENDRI/lcerovaz/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
echo "=== feature_vlm_interp CELL_DIR=$CELL_DIR MODEL_ID=$MODEL_ID ==="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

EXTRA=()
if [[ -n "${LIMIT:-}" ]]; then EXTRA+=(--limit "$LIMIT"); fi
if [[ "${FORCE:-0}" == "1" ]]; then EXTRA+=(--force); fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then EXTRA+=(--dry_run); fi

uv run python scripts/analysis/feature_vlm_interp.py \
    --cell_dir "$CELL_DIR" \
    --model_id "$MODEL_ID" \
    "${EXTRA[@]}"

echo
echo "=== feature_vlm_interp done: $(date) ==="
