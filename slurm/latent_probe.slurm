#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
#SBATCH --job-name=e34a_latent_probe
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_PDR
#SBATCH --exclude=lrdn2436,lrdn3001

set -euo pipefail
module load cuda/12.2

OUT_DIR="${OUT_DIR:-outputs/phase4_24_latent_probe_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-150}"
LR="${LR:-5e-3}"
BATCH="${BATCH:-8192}"

.venv/bin/python scripts/analysis/latent_probe.py \
    --out_dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch "$BATCH"

echo "DONE $OUT_DIR"
