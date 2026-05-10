#!/bin/bash
# Smoke training: SiT-B/2 × sd_vae × 5k step on real precomputed latents.
# Validates the Phase 1 → Phase 2 path on real data before committing to the
# full matched-compute sweep. Run from inside an `srun --gres=gpu:1` allocation.
set -euo pipefail

REPO=/leonardo/home/userexternal/lcerovaz/diffmechint
SCRATCH_BASE=/leonardo_scratch/large/userexternal/lcerovaz
FAST_BASE=/leonardo_scratch/fast/IscrC_YENDRI
LATENTS="${SCRATCH_BASE}/diffmechint/latents/sd_vae"
RUN_ID="sit_b2_sd_vae_smoke_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SCRATCH_BASE}/diffmechint/runs/${RUN_ID}"
CKPT_DIR="${OUT_DIR}/checkpoints"

module load cuda/12.2 || true

# HF cache (the precompute step doesn't need it now, but the tokenizer adapter is
# still instantiated by train.py for its spec; force offline so a missing proxy
# doesn't surface as a network error).
export HF_HOME="${FAST_BASE}/lcerovaz/hf_cache"
export HF_HUB_CACHE="${FAST_BASE}/lcerovaz/hf_cache/hub"
export TRANSFORMERS_CACHE="${FAST_BASE}/lcerovaz/hf_cache/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
mkdir -p "$OUT_DIR" "$CKPT_DIR"

echo "=== smoke training start: $(date) ==="
echo "REPO=$REPO"
echo "LATENTS=$LATENTS"
echo "OUT_DIR=$OUT_DIR"
echo "JOB_ID=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

uv run python -m diffmechint.training.train \
    tokenizer=sd_vae \
    model=sit_b_2 \
    transport=fm_ot \
    trainer=local_3090 \
    +data._target_=diffmechint.training.data.CachedLatentDataModule \
    +data.shard_dir="${LATENTS}" \
    +data.batch_size=64 \
    +data.num_workers=8 \
    +data.normalize=true \
    trainer.max_steps=5000 \
    +ckpt_dir="${CKPT_DIR}"

echo
echo "=== smoke training complete: $(date) ==="
echo "checkpoints written:"
ls -lh "${CKPT_DIR}"/*.safetensors 2>/dev/null | head -20 || echo "(none)"
