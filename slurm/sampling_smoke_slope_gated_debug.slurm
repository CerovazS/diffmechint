#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
#SBATCH --job-name=phase411_smoke_slope_gated_debug
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

SAE_ROOT="/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/staged_sae_matryoshka_k256_d32k_phase411_fullactivation_20260522_231004"
echo "=== DEBUG: hostname=$(hostname) ==="
echo "=== DEBUG: ls $SAE_ROOT/eq_vae/L9_T2/step_200000/ ==="
ls -la "$SAE_ROOT/eq_vae/L9_T2/step_200000/" || true
echo "=== DEBUG: stat final_81920000 ==="
stat "$SAE_ROOT/eq_vae/L9_T2/step_200000/final_81920000" 2>&1 || true
echo "=== DEBUG: python glob ==="
uv run python -c "
from pathlib import Path
b = Path('$SAE_ROOT/eq_vae/L9_T2/step_200000')
print('exists:', b.exists(), 'is_dir:', b.is_dir())
print('iterdir:', list(b.iterdir())[:5])
print('glob final_*:', list(b.glob('final_*')))
"
