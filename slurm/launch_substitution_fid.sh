#!/bin/bash
# Phase 4.5a — launch 27 substitution-FID jobs + 3 per-condition baselines.
#
# Cells: 3 conditions × 3 layers × 3 t-bins = 27 substitution jobs.
# Plus 1 baseline per condition (3 jobs). Total: 30 sbatch submissions.
#
# Each job: 5 000 samples × ODE-dopri5 250 steps × CFG=1.5, ~18 min wall on
# A100 → total ~9 GPU·h, ~3 h wall if 4-6 jobs run concurrent.
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
cd "$REPO"

# Condition → (SiT run dir, adapter for decode, normalize flag).
RUN_SD=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/runs/sit_b_2_sd_vae_full_20260511_000511
RUN_REPA=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/runs/sit_b_2_repa_e_full_20260511_002652
RUN_EQNOZ=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/runs/sit_b_2_eq_vae_noz_full_20260513_003701

declare -A RUNS=(
    [sd_vae]="$RUN_SD"
    [repa_e]="$RUN_REPA"
    [eq_vae]="$RUN_EQNOZ"
)
declare -A ADAPTERS=(
    [sd_vae]=sd_vae
    [repa_e]=repa_e
    [eq_vae]=eq_vae
)
# eq_vae SAEs were trained on the noz SiT run → no z-score on decode.
declare -A NORMALIZES=(
    [sd_vae]=true
    [repa_e]=true
    [eq_vae]=false
)

CONDITIONS=(sd_vae repa_e eq_vae)
LAYERS=(3 6 9)
T_BINS=(0 1 2)
DIT_STEP=200000
N_SAMPLES=5000
OUT_ROOT=outputs/phase4_5a_subst_fid

mkdir -p "$OUT_ROOT" slurm

echo "=== Phase 4.5a substitution-FID launch ==="
echo "OUT_ROOT=$OUT_ROOT  DIT_STEP=$DIT_STEP  N_SAMPLES=$N_SAMPLES"
echo

COUNT_SUB=0
COUNT_BASE=0

# Baselines first (1 per condition) — they're independent of layer/t.
for c in "${CONDITIONS[@]}"; do
    RUN="${RUNS[$c]}"
    ADAPTER="${ADAPTERS[$c]}"
    NORMALIZE="${NORMALIZES[$c]}"
    if [[ ! -d "$RUN" ]]; then
        echo "  [skip] $c: SiT run dir not found at $RUN"
        continue
    fi
    RUN=$RUN ADAPTER=$ADAPTER CONDITION=$c \
        MODE=baseline DIT_STEP=$DIT_STEP N_SAMPLES=$N_SAMPLES \
        NORMALIZE=$NORMALIZE OUT_ROOT=$OUT_ROOT \
        sbatch --job-name="dmi_subfid_${c}_baseline" \
               --export=ALL \
               slurm/sae_substitution_fid.slurm
    COUNT_BASE=$((COUNT_BASE + 1))
done

# Substitution sweep: 3 × 3 × 3 = 27 jobs.
for c in "${CONDITIONS[@]}"; do
    RUN="${RUNS[$c]}"
    ADAPTER="${ADAPTERS[$c]}"
    NORMALIZE="${NORMALIZES[$c]}"
    if [[ ! -d "$RUN" ]]; then
        echo "  [skip] $c: SiT run dir not found at $RUN"
        continue
    fi
    for l in "${LAYERS[@]}"; do
        for t in "${T_BINS[@]}"; do
            RUN=$RUN ADAPTER=$ADAPTER CONDITION=$c \
                MODE=sub LAYER=$l T_BIN=$t \
                DIT_STEP=$DIT_STEP N_SAMPLES=$N_SAMPLES \
                NORMALIZE=$NORMALIZE OUT_ROOT=$OUT_ROOT \
                sbatch --job-name="dmi_subfid_${c}_L${l}_T${t}" \
                       --export=ALL \
                       slurm/sae_substitution_fid.slurm
            COUNT_SUB=$((COUNT_SUB + 1))
        done
    done
done

echo
echo "=== Summary ==="
echo "Baseline jobs: $COUNT_BASE"
echo "Substitution jobs: $COUNT_SUB"
echo "Total: $((COUNT_BASE + COUNT_SUB))"
echo
echo "Queue (top 10 dmi_subfid_* jobs):"
squeue -u "$USER" --format="%.10i %.30j %.8T %.10M" 2>/dev/null | grep dmi_subfid | head -10
echo "Total dmi_subfid jobs in queue: $(squeue -u "$USER" --format='%.30j' 2>/dev/null | grep -c dmi_subfid || true)"
