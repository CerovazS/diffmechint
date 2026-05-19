#!/bin/bash
# ImageNet val precompute through the 3 trained-DiT tokenizers, 3 GPUs in
# parallel. Mirror of precompute_all_imagenet.sh but points at the val
# ImageFolder layout (50k images, ~4 min wall per tokenizer, I/O-bound).
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
DATA=/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/imagenet_val_imagefolder
OUT_BASE="${SCRATCH}/diffmechint/latents_val"
LOG_BASE="${REPO}/slurm/precompute_val-${SLURM_JOB_ID:-local}"

module load cuda/12.2 || true

export HF_HOME="${FAST}/lcerovaz/hf_cache"
export HF_HUB_CACHE="${FAST}/lcerovaz/hf_cache/hub"
export TRANSFORMERS_CACHE="${FAST}/lcerovaz/hf_cache/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
mkdir -p "$OUT_BASE" "$LOG_BASE"

declare -a TOKENIZERS=(sd_vae eq_vae repa_e)

echo "=== precompute_val_all start: $(date) ==="
echo "REPO=$REPO"
echo "DATA=$DATA"
echo "OUT_BASE=$OUT_BASE"
echo "LOG_BASE=$LOG_BASE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

declare -a PIDS=()
for i in 0 1 2; do
    tok="${TOKENIZERS[$i]}"
    log="${LOG_BASE}/${tok}.log"
    out="${OUT_BASE}/${tok}"
    rm -rf "$out"
    echo "  [GPU $i] $tok → $out (log: $log)"
    CUDA_VISIBLE_DEVICES=$i uv run python -m diffmechint.training.precompute_latents \
        tokenizer="$tok" \
        +data_dir="$DATA" \
        +output_dir="$out" \
        +shard_size=10000 +batch_size=32 +num_workers=8 \
        > "$log" 2>&1 &
    PIDS+=($!)
done

wait "${PIDS[@]}"

echo
echo "=== precompute_val_all complete: $(date) ==="
for tok in "${TOKENIZERS[@]}"; do
    out="${OUT_BASE}/${tok}"
    n_shards=$(ls "$out"/*.h5 2>/dev/null | wc -l)
    size=$(du -sh "$out" 2>/dev/null | cut -f1)
    echo "  $tok: shards=$n_shards size=$size"
done
