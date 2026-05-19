#!/bin/bash
# Round-trip PSNR for the 4 VAE adapters on the smoke-precomputed ImageNet latents.
# Closes CHECKLIST 1.13. Run from inside an `srun --gres=gpu:1 ...` allocation.
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint

module load cuda/12.2 || true

cd "$REPO"
uv run python scripts/round_trip_psnr_imagenet.py
