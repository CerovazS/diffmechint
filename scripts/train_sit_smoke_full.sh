#!/bin/bash
# Smoke training "full": SiT-B/2 × sd_vae × 5k step on real precomputed latents
# WITH validation_step (5% holdout) + SampleCallback (16 imgs every 1k step).
# Validates end-to-end Phase 1 → Phase 2 (incl. val + sampling) before the
# full matched-compute sweep.
set -euo pipefail

REPO=/leonardo/home/userexternal/lcerovaz/diffmechint
SCRATCH_BASE=/leonardo_scratch/large/userexternal/lcerovaz
FAST_BASE=/leonardo_scratch/fast/IscrC_YENDRI
LATENTS="${SCRATCH_BASE}/diffmechint/latents/sd_vae"
RUN_ID="sit_b2_sd_vae_smokefull_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SCRATCH_BASE}/diffmechint/runs/${RUN_ID}"
CKPT_DIR="${OUT_DIR}/checkpoints"

module load cuda/12.2 || true

export HF_HOME="${FAST_BASE}/lcerovaz/hf_cache"
export HF_HUB_CACHE="${FAST_BASE}/lcerovaz/hf_cache/hub"
export TRANSFORMERS_CACHE="${FAST_BASE}/lcerovaz/hf_cache/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
mkdir -p "$OUT_DIR" "$CKPT_DIR"

echo "=== smoke-full training start: $(date) ==="
echo "REPO=$REPO LATENTS=$LATENTS OUT_DIR=$OUT_DIR JOB_ID=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

uv run python -m diffmechint.training.train \
    tokenizer=sd_vae \
    model=sit_b_2 \
    transport=fm_ot \
    trainer=local_3090 \
    callbacks=sample_only \
    callbacks.sample.adapter_name=sd_vae \
    callbacks.sample.stats_path="${LATENTS}/stats.json" \
    callbacks.sample.every_n_steps=1000 \
    callbacks.sample.n_samples=16 \
    callbacks.sample.cfg_scale=4.0 \
    +data._target_=diffmechint.training.data.CachedLatentDataModule \
    +data.shard_dir="${LATENTS}" \
    +data.batch_size=64 \
    +data.num_workers=8 \
    +data.normalize=true \
    +data.holdout_fraction=0.05 \
    +data.holdout_seed=42 \
    +data.val_batch_size=128 \
    trainer.max_steps=5000 \
    +ckpt_dir="${CKPT_DIR}"

echo
echo "=== smoke-full training complete: $(date) ==="
echo "ckpt count:"
ls "$CKPT_DIR"/*.safetensors 2>/dev/null | wc -l
echo "sample count:"
ls "$OUT_DIR"/samples/*.png 2>/dev/null | wc -l
echo "samples written:"
ls -lh "$OUT_DIR"/samples/*.png 2>/dev/null
