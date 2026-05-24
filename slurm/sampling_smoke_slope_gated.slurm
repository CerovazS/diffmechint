#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
#SBATCH --job-name=phase411_smoke_slope_gated
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_PDR

set -euo pipefail
cd /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

SMOKE_DIR="outputs/phase4_11_feature_activation_patching/longplan_sampling_smoke_slope_gated_20260523"
export TASKS="$SMOKE_DIR/metrics/sampling_tasks.tsv"
export LOG_DIR="$SMOKE_DIR/logs"
export MAX_PARALLEL=1
export N_SAMPLES=64
export BATCH_SIZE=32
mkdir -p "$LOG_DIR"

# Reuse the same orchestrator but pointed at the smoke TSV.
# Override RUN_ID via env so outputs go to SMOKE_DIR, not the full-run dir.
export OVERRIDE_RUN_ID="longplan_sampling_smoke_slope_gated_20260523"
sed "s|longplan_sampling_final_n5000_slope_gated_20260523|$OVERRIDE_RUN_ID|g" \
    outputs/phase4_11_feature_activation_patching/longplan_sampling_final_n5000_slope_gated_20260523/run_sampling_slope_gated.sh \
    > "$SMOKE_DIR/run_smoke.sh"
chmod +x "$SMOKE_DIR/run_smoke.sh"
bash "$SMOKE_DIR/run_smoke.sh"
