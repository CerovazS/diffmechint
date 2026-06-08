#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
#SBATCH --job-name=spectral_smoke
#SBATCH --output=./slurm/logs/%x-%j.out
#SBATCH --error=./slurm/logs/%x-%j.err
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_PDR

set -euo pipefail
module load cuda/12.2
nvidia-smi -L

TS=$(date +%Y%m%d_%H%M%S)

echo "=== E36 smoke (GPU) ==="
.venv/bin/python scripts/analysis/band_alignment.py \
    --smoke --device cuda \
    --out_dir "outputs/phase7_0_smoke_${TS}/e36"

echo "=== E37 smoke (GPU, tiny) ==="
.venv/bin/python scripts/analysis/latent_dit_inheritance.py \
    --bands all --device cuda \
    --conditions sd_vae eq_vae --layers 6 --t_bins 1 \
    --n_images 128 --n_transfer_tokens 5000 \
    --out_dir "outputs/phase7_0_smoke_${TS}/e37"

echo "=== E38 smoke (GPU) ==="
.venv/bin/python scripts/analysis/spectral_probe.py \
    --source latent --smoke \
    --out_dir "outputs/phase7_0_smoke_${TS}/e38"

echo "SMOKE_ALL_DONE outputs/phase7_0_smoke_${TS}"
