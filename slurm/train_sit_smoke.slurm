#!/bin/bash
#SBATCH --job-name=dmi_sit_smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_YENDRI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

bash /leonardo/home/userexternal/lcerovaz/diffmechint/scripts/train_sit_smoke.sh
