#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
#SBATCH --job-name=e38_spectral_probe
#SBATCH --output=./slurm/logs/%x-%j.out
#SBATCH --error=./slurm/logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_PDR

set -euo pipefail
module load cuda/12.2

OUT_DIR="${OUT_DIR:-outputs/phase7_4_spectral_probe_${SLURM_JOB_ID}}"

.venv/bin/python scripts/analysis/spectral_probe.py \
    --source latent \
    --out_dir "$OUT_DIR"

.venv/bin/python scripts/analysis/spectral_probe.py \
    --source dit \
    --out_dir "$OUT_DIR"

echo "DONE $OUT_DIR"
