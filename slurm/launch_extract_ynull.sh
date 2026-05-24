#!/bin/bash
# Phase 4.7 — Y-null activation re-extraction (Bridge counterfactual for E08).
#
# Fans out 3 jobs: 1 per tokenizer condition. Each job extracts ALL 3 layers
# × ALL 3 t_bins (9 cells per condition) into a separate clean tree under
# $SCRATCH/.../activations_ynull/, so existing y=true activations stay
# untouched.
#
# NB: Previously this script fanned out 9 jobs (one per t_bin) but the
# ResidualStreamTap uses the *local* index within `--bins` for filenames,
# so 3 single-bin jobs per condition all wrote to `<L>_0.h5` and clobbered
# each other. Running all bins inside a single job per condition avoids
# the race and matches the original y_true layout (<L>_<t_bin_idx>.h5).
#
# Usage:
#   bash slurm/launch_extract_ynull.sh
#
# Output:
#   $OUT_ROOT/<cond>/step_200000/<L>_<T>.h5  (one shard per (L, T))
#   $OUT_ROOT/<cond>/step_200000/manifest.json  (includes y_null:true)
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
cd "$REPO"

# Canonical SiT-B/2 run dirs at step 200k (same as Phase 4.5/5 atlas/probe).
RUN_SD=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/runs/sit_b_2_sd_vae_full_20260511_000511
RUN_REPA=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/runs/sit_b_2_repa_e_full_20260511_002652
RUN_EQ=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/runs/sit_b_2_eq_vae_noz_full_20260513_003701

# Output root — parallel to canonical activations/ tree, so y=true outputs
# stay intact and y_null shards live under a sibling directory.
OUT_ROOT=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_ynull

# Same stratified sampling as the original extraction (50/class = 50k val).
STRATIFIED=50
# Same input latents the SiT was trained against (val-encoded latents tree).
LATENTS_ROOT=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents_val
# Val-encoded latents → use the whole set (no holdout carve).
NO_HOLDOUT=true
# Pin to the canonical analysis checkpoint.
ONLY_STEP=200000

# 3 conditions → 3 jobs (each does all 3 t_bins for all 3 layers).
CONDITIONS=(sd_vae repa_e eq_vae)

mkdir -p "$OUT_ROOT"
mkdir -p slurm

echo "=== Phase 4.7 y-null re-extraction ==="
echo "OUT_ROOT=$OUT_ROOT"
echo "Submitting 3 jobs (1 per condition, each does all 3 t_bins × 3 layers)"
echo

COUNT=0
for COND in "${CONDITIONS[@]}"; do
    case "$COND" in
        sd_vae)
            RUN="$RUN_SD"
            NORMALIZE=true
            ;;
        repa_e)
            RUN="$RUN_REPA"
            NORMALIZE=true
            ;;
        eq_vae)
            RUN="$RUN_EQ"
            # eq_vae_noz was trained with +data.normalize=false.
            NORMALIZE=false
            ;;
    esac

    # NB: 3 t_bins × 3 layers × 50k samples × 256 tok × 768 dim × fp16 ≈ 170 GB
    # of buffered activations; bump --mem above the SLURM template's default 128G.
    JOB_NAME="dmi_ynull_${COND}"
    RUN="$RUN" \
    CONDITION="$COND" \
    STRATIFIED="$STRATIFIED" \
    OUT_ROOT="$OUT_ROOT" \
    LATENTS_ROOT="$LATENTS_ROOT" \
    NO_HOLDOUT="$NO_HOLDOUT" \
    NORMALIZE="$NORMALIZE" \
    Y_NULL=true \
    ONLY_STEP="$ONLY_STEP" \
    BINS="0.025 0.20 0.50" \
    sbatch --time=02:00:00 \
           --mem=256G \
           --job-name="$JOB_NAME" \
           --export=ALL \
           slurm/extract_activations.slurm
    COUNT=$((COUNT + 1))
done

echo
echo "=== Submitted $COUNT jobs ==="
echo "Queue snapshot:"
squeue -u "$USER" --format="%.10i %.30j %.8T %.10M" 2>/dev/null | grep dmi_ynull | head -15
TOTAL=$(squeue -u "$USER" --format='%.30j' 2>/dev/null | grep -c dmi_ynull || true)
echo "Total dmi_ynull jobs in queue: $TOTAL"
echo
echo "Monitor with: squeue -u \$USER | grep dmi_ynull"
echo "Outputs will land under: $OUT_ROOT/<cond>/step_200000/"
