#!/bin/bash
# Run from inside an `srun --gres=gpu:4 ...` allocation.
# Encodes 256 ImageNet train images through each of the 4 working VAE adapters,
# one per GPU, in parallel. Validates the precompute pipeline + stats.json schema
# on real data before committing to a full sweep.
set -euo pipefail

REPO=/leonardo/home/userexternal/lcerovaz/diffmechint
DATA=/leonardo_scratch/fast/IscrC_YENDRI/imagenet/train
OUT_BASE=$FAST/lcerovaz/diffmechint/latents

module load cuda/12.2 || true

# HF cache is already warm from the login-node prefetch; force offline so a missing
# squid proxy on the compute node doesn't surface as a noisy network failure.
export HF_HOME=$FAST/lcerovaz/hf_cache
export HF_HUB_CACHE=$FAST/lcerovaz/hf_cache/hub
export TRANSFORMERS_CACHE=$FAST/lcerovaz/hf_cache/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"

declare -a TOKENIZERS=(sd_vae eq_vae repa_e dc_ae_1_0)
mkdir -p "$OUT_BASE" /tmp/smoke_logs

echo "Launching 4 parallel encodes — 256 imgs each — at $(date)"

for i in 0 1 2 3; do
    tok="${TOKENIZERS[$i]}"
    log="/tmp/smoke_logs/${tok}.log"
    out="$OUT_BASE/${tok}_smoke"
    rm -rf "$out"
    echo "  [GPU $i] $tok → $out (log: $log)"
    CUDA_VISIBLE_DEVICES=$i uv run python -m diffmechint.training.precompute_latents \
        tokenizer="$tok" \
        +data_dir="$DATA" \
        +output_dir="$out" \
        +max_images=256 +shard_size=256 +batch_size=32 +num_workers=4 \
        > "$log" 2>&1 &
done

wait
echo
echo "=== All 4 encodes finished at $(date) ==="
echo
for tok in "${TOKENIZERS[@]}"; do
    out="$OUT_BASE/${tok}_smoke"
    log="/tmp/smoke_logs/${tok}.log"
    if [[ -f "$out/stats.json" ]]; then
        size=$(du -sh "$out" | cut -f1)
        wall=$(python3 -c "import json; s=json.load(open('$out/stats.json')); print(f\"{s['wall_seconds']:.1f}s, mean={s['global_mean']:.3f}, std={s['global_std']:.3f}\")")
        echo "  ✓ $tok: $size — $wall"
    else
        echo "  ✗ $tok: stats.json MISSING — see $log"
        tail -10 "$log" | sed 's/^/    /'
    fi
done
