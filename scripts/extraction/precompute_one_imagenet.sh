#!/bin/bash
# Precompute a single tokenizer's latents over the full ImageNet train set.
# Designed for an `srun --gres=gpu:1` allocation. Tokenizer name must be the
# first argument; it must match a config under `conf/tokenizer/<tok>.yaml`.
set -euo pipefail

TOK="${1:?usage: $0 <tokenizer>}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
DATA=/leonardo_scratch/fast/IscrC_YENDRI/imagenet/train
SCRATCH_BASE=/leonardo_scratch/large/userexternal/lcerovaz
FAST_BASE=/leonardo_scratch/fast/IscrC_YENDRI
OUT="${SCRATCH_BASE}/diffmechint/latents/${TOK}"
LOG_DIR="${REPO}/slurm/precompute-${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

module load cuda/12.2 || true

# HF cache populated on the login node; force offline so a missing proxy on the
# compute node doesn't trigger a network call.
export HF_HOME="${FAST_BASE}/lcerovaz/hf_cache"
export HF_HUB_CACHE="${FAST_BASE}/lcerovaz/hf_cache/hub"
export TRANSFORMERS_CACHE="${FAST_BASE}/lcerovaz/hf_cache/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"

echo "=== precompute_one $TOK start: $(date) ==="
echo "OUT=$OUT"
echo "JOB_ID=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

rm -rf "$OUT"
uv run python -m diffmechint.training.precompute_latents \
    tokenizer="$TOK" \
    +data_dir="$DATA" \
    +output_dir="$OUT" \
    +shard_size=10000 +batch_size=64 +num_workers=12

echo
echo "=== precompute_one $TOK complete: $(date) ==="
n_shards=$(ls "$OUT"/*.h5 2>/dev/null | wc -l)
size=$(du -sh "$OUT" | cut -f1)
echo "  shards=$n_shards size=$size"
uv run python -c "
import json
s = json.load(open('${OUT}/stats.json'))
print(f\"  images={s['images_written']} wall={s['wall_seconds']/60:.1f}min mean={s['global_mean']:.3f} std={s['global_std']:.3f}\")
"
