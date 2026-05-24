#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
#SBATCH --job-name=phase411_final5000_chunk
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --account=IscrC_PDR
#SBATCH --array=0-3

set -euo pipefail

cd /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

RUN_DIR="outputs/phase4_11_feature_activation_patching/longplan_sampling_final_n5000_20260523_023818"
export TASKS="$RUN_DIR/metrics/sampling_tasks_chunk_${SLURM_ARRAY_TASK_ID}.tsv"
export LOG_DIR="$RUN_DIR/logs/chunk_${SLURM_ARRAY_TASK_ID}"
export MAX_PARALLEL=2
export N_SAMPLES=5000
export BATCH_SIZE=32

mkdir -p "$LOG_DIR"
bash "$RUN_DIR/run_sampling_final.sh"
