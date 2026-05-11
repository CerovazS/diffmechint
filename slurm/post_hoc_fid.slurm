#!/bin/bash
#SBATCH --job-name=dmi_post_hoc_fid
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_YENDRI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Inputs (must export before sbatch):
#   RUN=<absolute path to run dir>
#   ADAPTER=<tokenizer name: sd_vae | eq_vae | repa_e | dc_ae_1_0>
RUN="${RUN:?must export RUN before sbatch}"
ADAPTER="${ADAPTER:?must export ADAPTER before sbatch}"
MODEL_NAME="${MODEL_NAME:-SiT-B/2}"  # SiT-B/1 for DC-AE patch=1

REPO=/leonardo/home/userexternal/lcerovaz/diffmechint

module load cuda/12.2 || true

# Cleanfid Inception symlink (HOME-cached, no internet on compute).
INCEPTION_HOME="${HOME}/.cache/cleanfid_models/inception-2015-12-05.pt"
ln -sf "${INCEPTION_HOME}" /tmp/inception-2015-12-05.pt
echo "Inception symlinked: $(ls -lh /tmp/inception-2015-12-05.pt)"

cd "$REPO"
echo "=== post_hoc_fid for $ADAPTER ($RUN) start: $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

uv run python scripts/post_hoc_fid.py "$RUN" "$ADAPTER" --model_name "$MODEL_NAME"

echo
echo "=== post_hoc_fid done: $(date) ==="
echo "--- fid.csv ---"
cat "$RUN/metrics/validation/fid.csv"
