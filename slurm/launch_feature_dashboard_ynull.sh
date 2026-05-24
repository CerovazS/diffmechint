#!/bin/bash
# Phase 4.7 — full 27-cell feature dashboard sweep on y-null activations.
#
# Mirror of launch_feature_dashboard_full.sh but pointed at
# `activations_ynull/` and writing to a parallel output tree so y-true
# dashboards stay untouched.
#
# Predicted result (from E09 SAE eval): the 4 top-mono-count cells of E08
# (repa_e L3/L6/L9 T0 and sd_vae L3 T0) drop sharply in mono_count, the
# other 23 stay close to their y-true values.
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
cd "$REPO"

CONDITIONS=(sd_vae repa_e eq_vae)
LAYERS=(3 6 9)
T_BINS=(0 1 2)
DIT_STEP=200000

# Y-null source + output tree (parallel to canonical y-true tree).
ACT_ROOT=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_ynull
OUT_ROOT=outputs/phase4_5b_feature_viz_ynull
# Reuse the existing Matryoshka SAEs (validated by E09; verdict Option A).
SAE_ROOT=/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k

FORCE="${FORCE:-0}"

mkdir -p "$OUT_ROOT" slurm

echo "=== Phase 4.7 y-null feature-dashboard atlas ==="
echo "ACT_ROOT=$ACT_ROOT"
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
                DIT_STEP=$DIT_STEP \
                ACT_ROOT=$ACT_ROOT OUT_ROOT=$OUT_ROOT SAE_ROOT=$SAE_ROOT \
                sbatch --job-name="dmi_featdash_ynull_${c}_L${l}_T${t}" \
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
echo "Queue:"
squeue -u "$USER" --format="%.10i %.40j %.8T %.10M" 2>/dev/null | grep dmi_featdash_ynull | head -10
TOTAL=$(squeue -u "$USER" --format='%.40j' 2>/dev/null | grep -c dmi_featdash_ynull || true)
echo "Total dmi_featdash_ynull jobs in queue: $TOTAL"
