#!/bin/bash
#SBATCH --job-name=dmi_precompute_val_all
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=90G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

bash /leonardo_work/IscrC_PDR/lcerovaz/diffmechint/scripts/extraction/precompute_val_all_imagenet.sh
