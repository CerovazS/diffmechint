#!/bin/bash
# Phase 4.5b — full 27-cell feature dashboard sweep.
#
# 3 conditions × 3 layers × 3 t-bins = 27 SLURM jobs, ~15 min each on A100.
# eq_vae L6/T2 already exists; skipped (re-run with FORCE=1).
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
cd "$REPO"

CONDITIONS=(sd_vae repa_e eq_vae)
LAYERS=(3 6 9)
T_BINS=(0 1 2)
DIT_STEP=200000
OUT_ROOT=outputs/phase4_5b_feature_viz
FORCE="${FORCE:-0}"

mkdir -p "$OUT_ROOT" slurm

echo "=== Phase 4.5b feature-dashboard full atlas ==="
echo "OUT_ROOT=$OUT_ROOT  DIT_STEP=$DIT_STEP"
COUNT=0
SKIPPED=0
for c in "${CONDITIONS[@]}"; do
    for l in "${LAYERS[@]}"; do
        for t in "${T_BINS[@]}"; do
            cell_dir="$OUT_ROOT/${c}_L${l}_T${t}"
            if [[ -f "$cell_dir/features.html" && "$FORCE" != "1" ]]; then
                echo "  [skip] $cell_dir already has features.html (FORCE=1 to override)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            CONDITION=$c LAYER=$l T_BIN=$t \
                DIT_STEP=$DIT_STEP OUT_ROOT=$OUT_ROOT \
                sbatch --job-name="dmi_featdash_${c}_L${l}_T${t}" \
                       --export=ALL \
                       slurm/feature_dashboard.slurm
            COUNT=$((COUNT + 1))
        done
    done
done

echo
echo "=== Summary ==="
echo "Submitted: $COUNT  Skipped: $SKIPPED"
echo
echo "Queue (top 10 dmi_featdash_* jobs):"
squeue -u "$USER" --format="%.10i %.30j %.8T %.10M" 2>/dev/null | grep dmi_featdash | head -10
TOTAL=$(squeue -u "$USER" --format='%.30j' 2>/dev/null | grep -c dmi_featdash || true)
echo "Total dmi_featdash jobs in queue: $TOTAL"
