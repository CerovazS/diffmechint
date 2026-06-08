#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
#SBATCH --job-name=e36_band_alignment
#SBATCH --output=./slurm/logs/%x-%j.out
#SBATCH --error=./slurm/logs/%x-%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_PDR

set -euo pipefail
mkdir -p slurm/logs

OUT_DIR="${OUT_DIR:-outputs/phase7_2_band_alignment_${SLURM_JOB_ID}}"

uv run python scripts/analysis/band_alignment.py \
    --out_dir "$OUT_DIR" \
    --device cuda

echo "DONE $OUT_DIR"
