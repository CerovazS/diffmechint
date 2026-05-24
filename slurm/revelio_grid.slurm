#!/bin/bash
#SBATCH --job-name=dmi_revelio
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

# Phase 5 Revelio grid — one (condition, dit_step) per job.
#
# Required inputs (export before sbatch):
#   CONDITION   sd_vae | repa_e | eq_vae
# Optional:
#   DIT_STEP    default 200000
#   OUT_ROOT    default outputs/phase5_revelio_grid
#   MAX_SAMPLES default 50000 (per-cell subsample cap)

CONDITION="${CONDITION:?must export CONDITION (sd_vae|repa_e|eq_vae)}"
DIT_STEP="${DIT_STEP:-200000}"
OUT_ROOT="${OUT_ROOT:-outputs/phase5_revelio_grid}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
# Activation source. Default = canonical y_true val tree; override to
# `.../activations_ynull` for y-null counterfactual probing (Phase 4.7).
ACTIVATIONS_ROOT="${ACTIVATIONS_ROOT:-/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_val}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

cd "$REPO"
echo "=== revelio_grid CONDITION=$CONDITION DIT_STEP=$DIT_STEP ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null | head -3

uv run python scripts/analysis/run_revelio_grid.py \
    --condition "$CONDITION" \
    --dit_step "$DIT_STEP" \
    --max_samples "$MAX_SAMPLES" \
    --activations_root "$ACTIVATIONS_ROOT" \
    --out_root "$OUT_ROOT"

echo "=== done $(date) ==="
ls -la "$OUT_ROOT/${CONDITION}__step_$(printf '%06d' $DIT_STEP)/"
