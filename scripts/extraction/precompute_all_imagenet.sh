#!/bin/bash
# Full ImageNet train precompute through the 4 working VAE adapters, 4 GPUs in
# parallel. Output → $SCRATCH/diffmechint/latents/<tok>/. Run via sbatch
# slurm/precompute_all.slurm; the SLURM job ID is appended to each per-VAE log.
set -euo pipefail

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
DATA=/leonardo_scratch/fast/IscrC_YENDRI/imagenet/train
OUT_BASE="${SCRATCH}/diffmechint/latents"
LOG_BASE="${REPO}/slurm/precompute-${SLURM_JOB_ID:-local}"

module load cuda/12.2 || true

# HF cache populated on the login node; force offline so a missing proxy on the
# compute node doesn't trigger a network call.
export HF_HOME="${FAST}/lcerovaz/hf_cache"
export HF_HUB_CACHE="${FAST}/lcerovaz/hf_cache/hub"
export TRANSFORMERS_CACHE="${FAST}/lcerovaz/hf_cache/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
mkdir -p "$OUT_BASE" "$LOG_BASE"

declare -a TOKENIZERS=(sd_vae eq_vae repa_e dc_ae_1_0)

echo "=== precompute_all start: $(date) ==="
echo "REPO=$REPO"
echo "DATA=$DATA"
echo "OUT_BASE=$OUT_BASE"
echo "LOG_BASE=$LOG_BASE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for i in 0 1 2 3; do
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
done

# Periodic progress while we wait — one heartbeat per minute, listing throughput.
(
    while kill -0 $$ 2>/dev/null; do
        sleep 300
        echo "--- heartbeat $(date) ---"
        for tok in "${TOKENIZERS[@]}"; do
            log="${LOG_BASE}/${tok}.log"
            if [[ -f "$log" ]]; then
                last=$(grep -oP 'written \d+' "$log" | tail -1 || true)
                rate=$(grep -oP '\d+\.\d+ img/s' "$log" | tail -1 || true)
                echo "  $tok: $last  $rate"
            fi
        done
    done
) &
HEARTBEAT_PID=$!

wait %1 %2 %3 %4
kill "$HEARTBEAT_PID" 2>/dev/null || true

echo
echo "=== precompute_all complete: $(date) ==="
echo
for tok in "${TOKENIZERS[@]}"; do
    out="${OUT_BASE}/${tok}"
    if [[ -f "${out}/stats.json" ]]; then
        n_shards=$(ls "${out}"/*.h5 2>/dev/null | wc -l)
        size=$(du -sh "$out" | cut -f1)
        wall=$(uv run python -c "import json; s=json.load(open('${out}/stats.json')); print(f\"images={s['images_written']} wall={s['wall_seconds']/60:.1f}min mean={s['global_mean']:.3f} std={s['global_std']:.3f}\")")
        echo "  ✓ $tok: $n_shards shards, $size — $wall"
    else
        echo "  ✗ $tok: stats.json MISSING — see ${LOG_BASE}/${tok}.log"
        tail -15 "${LOG_BASE}/${tok}.log" | sed 's/^/    /'
    fi
done
