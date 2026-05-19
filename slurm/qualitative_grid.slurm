#!/bin/bash
#SBATCH --job-name=dmi_qual_grid
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true
INCEPTION_HOME="${HOME}/.cache/cleanfid_models/inception-2015-12-05.pt"
ln -sf "${INCEPTION_HOME}" /tmp/inception-2015-12-05.pt
cd "$REPO"

# Target: eq_vae L6/T2 — best-quality generator (baseline FID 25.3), worst-effect
# eq_vae cell (ΔFID +0.76). The eq_vae SiT was trained with +data.normalize=false
# (eq_vae_noz run) → pass --no_normalize.
RUN=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/runs/sit_b_2_eq_vae_noz_full_20260513_003701

# 4 classes that ImageNet-conditioned diffusion models typically generate cleanly.
# 207 golden retriever, 933 cheeseburger, 281 tabby cat, 980 volcano
# CFG bumped to 4.0 for sharper per-sample visuals (the headline ΔFID was
# measured at CFG 1.5 per SiT/REPA-E leaderboard recipe; here we're doing a
# visual demo, not a FID re-measurement).
uv run python scripts/eval/make_substitution_grid.py \
    --run "$RUN" \
    --adapter eq_vae \
    --condition eq_vae \
    --layer 6 \
    --t_bin 2 \
    --class_indices 207 933 281 980 \
    --seed 0 \
    --cfg 4.0 \
    --no_normalize \
    --out_path flywheel/sae/e06_substitution_fid/plots/qualitative_grid.png

echo "=== done $(date) ==="
ls -la flywheel/sae/e06_substitution_fid/plots/
