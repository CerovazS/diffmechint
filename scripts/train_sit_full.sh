#!/bin/bash
# Full Phase 2 SiT training, parametric on tokenizer. Run via sbatch
# slurm/train_sit_full.slurm with TOK exported.
#
# Setup: 2× A100 (DDP), batch 128/GPU (= 256 global), 200k step, val every 1k,
# sample PNGs (CFG=1, CFG=4) every 5k, mini-FID 5k vs ImageNet val every 25k.
set -euo pipefail

TOK="${1:?usage: $0 <tokenizer> [model_name]}"
MODEL="${2:-sit_b_2}"   # Hydra model config (e.g. sit_b_2 / sit_b_1)
CALLBACKS="${CALLBACKS:-sample_only}"  # default: no live FID (use post_hoc_fid.sh after)
# NORMALIZE=true (default): per-feature z-score from stats.json. NORMALIZE=false:
# raw latents (only the AE's own scaling_factor). Disable when reproducing
# tokenizers whose recipe forbids extra normalization (e.g. EQ-VAE).
NORMALIZE="${NORMALIZE:-true}"
# Optional run-name suffix (e.g. "noz" for no-zscore ablations) to avoid collisions.
RUN_SUFFIX="${RUN_SUFFIX:-}"

REPO=/leonardo/home/userexternal/lcerovaz/diffmechint
SCRATCH_BASE=/leonardo_scratch/large/userexternal/lcerovaz
FAST_BASE=/leonardo_scratch/fast/IscrC_YENDRI
LATENTS="${SCRATCH_BASE}/diffmechint/latents/${TOK}"
# Tag run with model variant + optional suffix so variants don't collide on disk.
SUFFIX_TAG=""
[[ -n "$RUN_SUFFIX" ]] && SUFFIX_TAG="_${RUN_SUFFIX}"
RUN_ID="${MODEL}_${TOK}${SUFFIX_TAG}_full_$(date +%Y%m%d_%H%M%S)"
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

echo "=== train_sit_full $TOK start: $(date) ==="
echo "REPO=$REPO LATENTS=$LATENTS OUT_DIR=$OUT_DIR JOB_ID=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    RESUME_ARGS+=("+resume_from=${RESUME_FROM}")
    if [[ -n "${EMA_RESUME_FROM:-}" ]]; then
        RESUME_ARGS+=("+ema_resume_from=${EMA_RESUME_FROM}")
    fi
    # When resuming from already-trained weights, skip the LR warmup so we
    # don't drop the learning rate to ~0 on top of a tuned model.
    RESUME_ARGS+=("model.warmup_steps=0")
    echo "RESUME_FROM=${RESUME_FROM}"
    echo "EMA_RESUME_FROM=${EMA_RESUME_FROM:-(unset)}"
fi

# When normalization is disabled at the dataset level, callbacks must also
# skip denormalization (otherwise samples are pushed through an inverse
# z-score that doesn't apply). null = skip transform, dataset stays raw-scale.
SAMPLE_STATS="${LATENTS}/stats.json"
if [[ "$NORMALIZE" == "false" ]]; then
    SAMPLE_STATS=null
    echo "NORMALIZE=false → callbacks.sample.stats_path=null, +data.normalize=false"
fi

uv run python -m diffmechint.training.train \
    tokenizer="${TOK}" \
    model="${MODEL}" \
    transport=fm_ot \
    trainer=cineca_2xa100 \
    callbacks="${CALLBACKS}" \
    callbacks.sample.adapter_name="${TOK}" \
    callbacks.sample.stats_path="${SAMPLE_STATS}" \
    callbacks.sample.every_n_steps=5000 \
    callbacks.sample.n_samples=16 \
    +data._target_=diffmechint.training.data.CachedLatentDataModule \
    +data.shard_dir="${LATENTS}" \
    +data.batch_size=128 \
    +data.num_workers=8 \
    +data.normalize="${NORMALIZE}" \
    +data.holdout_fraction=0.05 \
    +data.holdout_seed=42 \
    +data.val_batch_size=256 \
    +ckpt_dir="${CKPT_DIR}" \
    "${RESUME_ARGS[@]}"

echo
echo "=== train_sit_full $TOK complete: $(date) ==="
echo "ckpt count:"
ls "$CKPT_DIR"/*.safetensors 2>/dev/null | wc -l
echo "samples count:"
ls "$OUT_DIR"/samples/*.png 2>/dev/null | wc -l
