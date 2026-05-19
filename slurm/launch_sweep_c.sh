#!/bin/bash
# Sweep C — BatchTopK k=128 d_sae=32768.
# 27 jobs, one per (cond × layer × t_bin) chain. Same per-stage budget as
# sweeps A and B (20 000 optimiser steps × 4 096 batch = 81 920 000 tokens).
#
# Non-overlapping checkpoint root: sae_batch_topk_k128_d32k.
# wandb tag: batch_topk_k128_d32k.
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
cd "$REPO"

CONDITIONS=(sd_vae repa_e eq_vae)
LAYERS=(3 6 9)
T_BINS=(0 1 2)
BASE=/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint

SAMPLES=81920000

TAG="batch_topk_k128_d32k"
ROOT="$BASE/sae_$TAG"
mkdir -p "$ROOT"
echo "=== Sweep C: $TAG → $ROOT ==="
COUNT=0
for c in "${CONDITIONS[@]}"; do
  for l in "${LAYERS[@]}"; do
    for t in "${T_BINS[@]}"; do
      ONLY="${c}/L${l}/T${t}" \
      VARIANT=batch_topk \
      K=128 \
      D_SAE=32768 \
      BASE_SAMPLES=$SAMPLES \
      WARM_SAMPLES=$SAMPLES \
      VARIANT_TAG=$TAG \
      OUT_ROOT=$ROOT \
      sbatch --time=12:00:00 \
             --job-name="dmi_sae_C_${c}_L${l}_T${t}" \
             --export=ALL \
             slurm/train_sae.slurm
      COUNT=$((COUNT + 1))
    done
  done
done

echo
echo "=== Summary ==="
echo "Sweep C (BatchTopK): $COUNT jobs submitted → $ROOT"
echo
echo "Queue (top 10 dmi_sae_C jobs):"
squeue -u "$USER" --format="%.10i %.30j %.8T %.10M" 2>/dev/null | grep dmi_sae_C | head -10
TOTAL=$(squeue -u "$USER" --format='%.30j' 2>/dev/null | grep -c dmi_sae_C || true)
echo "Total dmi_sae_C jobs in queue: $TOTAL"
