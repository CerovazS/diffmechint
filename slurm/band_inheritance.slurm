#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
#SBATCH --job-name=e37_band_inheritance
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_PDR
#SBATCH --exclude=lrdn2436,lrdn3001

set -euo pipefail
module load cuda/12.2

OUT_DIR="${OUT_DIR:-outputs/phase7_3_band_inheritance_${SLURM_JOB_ID}}"

uv run python scripts/analysis/latent_dit_inheritance.py \
    --bands all \
    --out_dir "$OUT_DIR" \
    --device cuda

echo "DONE $OUT_DIR"
