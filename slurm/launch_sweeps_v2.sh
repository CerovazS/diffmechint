#!/bin/bash
# Launch two SAE sweeps with non-overlapping checkpoint paths and distinct
# wandb tags. Each sweep submits 27 jobs (3 cond × 3 layer × 3 t_bin), one
# job per (cond, layer, t_bin) chain.
#
# Sweep A — TopK k=128 d_sae=32768
# Sweep B — Matryoshka k=256 d_sae=32768 widths=(4096,8192,16384,32768)
#           → effective readout K endpoints ≈ (32, 64, 128, 256)
#
# Per-stage budget: 20000 optimizer steps × batch 4096 = 81 920 000 tokens.
# All 7 DiT-ckpt stages cold-restart (warm_mode=cold) with the full budget.
#
# Total: 54 SLURM jobs.
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
cd "$REPO"

CONDITIONS=(sd_vae repa_e eq_vae)
LAYERS=(3 6 9)
T_BINS=(0 1 2)
BASE=/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint

# 20k optimizer steps × 4096 batch = 81_920_000 tokens / stage.
SAMPLES=81920000

# --- Sweep A: TopK k=128 d_sae=32768 ---
TAG_A="topk_k128_d32k"
ROOT_A="$BASE/sae_$TAG_A"
mkdir -p "$ROOT_A"
echo "=== Sweep A: $TAG_A → $ROOT_A ==="
COUNT_A=0
for c in "${CONDITIONS[@]}"; do
  for l in "${LAYERS[@]}"; do
    for t in "${T_BINS[@]}"; do
      ONLY="${c}/L${l}/T${t}" \
      VARIANT=topk \
      K=128 \
      D_SAE=32768 \
      BASE_SAMPLES=$SAMPLES \
      WARM_SAMPLES=$SAMPLES \
      VARIANT_TAG=$TAG_A \
      OUT_ROOT=$ROOT_A \
      sbatch --time=12:00:00 \
             --job-name="dmi_sae_A_${c}_L${l}_T${t}" \
             --export=ALL \
             slurm/train_sae.slurm
      COUNT_A=$((COUNT_A + 1))
    done
  done
done

# --- Sweep B: Matryoshka K=256 d_sae=32768 widths=(4096,8192,16384,32768) ---
TAG_B="matryoshka_k256_d32k"
ROOT_B="$BASE/sae_$TAG_B"
mkdir -p "$ROOT_B"
echo "=== Sweep B: $TAG_B → $ROOT_B ==="
COUNT_B=0
for c in "${CONDITIONS[@]}"; do
  for l in "${LAYERS[@]}"; do
    for t in "${T_BINS[@]}"; do
      ONLY="${c}/L${l}/T${t}" \
      VARIANT=matryoshka \
      K=256 \
      D_SAE=32768 \
      MATRYOSHKA_WIDTHS="4096 8192 16384 32768" \
      BASE_SAMPLES=$SAMPLES \
      WARM_SAMPLES=$SAMPLES \
      VARIANT_TAG=$TAG_B \
      OUT_ROOT=$ROOT_B \
      sbatch --time=18:00:00 \
             --job-name="dmi_sae_B_${c}_L${l}_T${t}" \
             --export=ALL \
             slurm/train_sae.slurm
      COUNT_B=$((COUNT_B + 1))
    done
  done
done

echo
echo "=== Summary ==="
echo "Sweep A (TopK):       $COUNT_A jobs submitted → $ROOT_A"
echo "Sweep B (Matryoshka): $COUNT_B jobs submitted → $ROOT_B"
echo
echo "Queue (top 10 dmi_sae jobs):"
squeue -u "$USER" --format="%.10i %.30j %.8T %.10M" 2>/dev/null | grep dmi_sae | head -10
TOTAL=$(squeue -u "$USER" --format='%.30j' 2>/dev/null | grep -c dmi_sae || true)
echo "Total dmi_sae jobs in queue: $TOTAL"
