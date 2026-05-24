#!/bin/bash
# Phase 4.7 — Revelio probing grid on y-null activations.
#
# 3 jobs (1 per condition), each does all 9 cells × 6 concepts.
# Output goes to outputs/phase5_revelio_grid_ynull/ so y_true probing
# results stay intact.
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
cd "$REPO"

CONDITIONS=(sd_vae repa_e eq_vae)
DIT_STEP=200000
ACTIVATIONS_ROOT=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_ynull
OUT_ROOT=outputs/phase5_revelio_grid_ynull

mkdir -p "$OUT_ROOT" slurm

echo "=== Phase 4.7 y-null Revelio probing ==="
echo "ACTIVATIONS_ROOT=$ACTIVATIONS_ROOT"
echo "OUT_ROOT=$OUT_ROOT  DIT_STEP=$DIT_STEP"
echo

COUNT=0
for c in "${CONDITIONS[@]}"; do
    CONDITION=$c DIT_STEP=$DIT_STEP \
        ACTIVATIONS_ROOT=$ACTIVATIONS_ROOT OUT_ROOT=$OUT_ROOT \
        sbatch --job-name="dmi_revelio_ynull_${c}" \
               --export=ALL \
               slurm/revelio_grid.slurm
    COUNT=$((COUNT + 1))
done

echo "=== Submitted $COUNT jobs ==="
squeue -u "$USER" --format="%.10i %.40j %.8T %.10M" 2>/dev/null | grep dmi_revelio_ynull | head -5
