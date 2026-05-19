#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
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

# Extend NCCL collective-op watchdog timeout from 30 min → 1h. Required for
# CALLBACKS=full because rank 0 spends ~5-8 min inside MiniFIDCallback while
# rank > 0 sits in dist.barrier(); too tight a timeout aborts the run.
export NCCL_TIMEOUT_MS="${NCCL_TIMEOUT_MS:-3600000}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"

srun --export=ALL bash /leonardo_work/IscrC_PDR/lcerovaz/diffmechint/scripts/training/train_sit_full.sh "$TOK" "$MODEL"
