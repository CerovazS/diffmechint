#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_YENDRI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Lightning DDP via SLURM: needs ntasks-per-node == devices, and `srun`
# launches the python process once per task. Lightning's auto-SLURM detection
# then sets each rank to its CUDA_VISIBLE_DEVICES slot.
TOK="${TOK:?must export TOK before sbatch (the tokenizer name)}"
MODEL="${MODEL:-sit_b_2}"  # default sit_b_2; export MODEL=sit_b_1 for DC-AE patch=1
# Optional: RESUME_FROM=<path/to/step_NNNNNNNN.safetensors>
# Pair with EMA_RESUME_FROM=<path/to/step_NNNNNNNN_ema.safetensors>
srun --export=ALL bash /leonardo/home/userexternal/lcerovaz/diffmechint/scripts/train_sit_full.sh "$TOK" "$MODEL"
