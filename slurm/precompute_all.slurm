#!/bin/bash
#SBATCH --job-name=dmi_precompute_all
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

bash /leonardo/home/userexternal/lcerovaz/diffmechint/scripts/precompute_all_imagenet.sh
