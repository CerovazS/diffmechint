#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_YENDRI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

TOK="${TOK:?must export TOK before sbatch (the tokenizer name)}"
bash /leonardo/home/userexternal/lcerovaz/diffmechint/scripts/precompute_one_imagenet.sh "$TOK"
