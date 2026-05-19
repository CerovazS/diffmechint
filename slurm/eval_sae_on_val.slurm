#!/bin/bash
#SBATCH --job-name=dmi_eval_sae_val
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Inputs:
#   SAE_ROOT    (required) — root under which sae_weights.safetensors live
#   OUTPUT_DIR  (required) — where to write metrics/aggregate/per-SAE JSON
# Optional:
#   VAL_ROOT    (default: $SCRATCH/.../activations_val)
#   INCLUDE_MID (true|false, default false)
#   BATCH_SIZE  (default 4096)
#   MAX_TOKENS  (default: all)
#   LIMIT       (default: all)
SAE_ROOT="${SAE_ROOT:?must export SAE_ROOT before sbatch}"
OUTPUT_DIR="${OUTPUT_DIR:?must export OUTPUT_DIR before sbatch}"
VAL_ROOT="${VAL_ROOT:-/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_val}"
INCLUDE_MID="${INCLUDE_MID:-false}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_TOKENS="${MAX_TOKENS:-}"
LIMIT="${LIMIT:-}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

cd "$REPO"
echo "=== eval_sae_on_val start: $(date) ==="
echo "host: $(hostname) job: ${SLURM_JOB_ID:-local}"
echo "SAE_ROOT=$SAE_ROOT"
echo "VAL_ROOT=$VAL_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

EXTRA_ARGS=()
if [[ "$INCLUDE_MID" == "true" ]]; then
    EXTRA_ARGS+=("--include_mid")
fi
if [[ -n "$MAX_TOKENS" ]]; then
    EXTRA_ARGS+=("--max_tokens" "$MAX_TOKENS")
fi
if [[ -n "$LIMIT" ]]; then
    EXTRA_ARGS+=("--limit" "$LIMIT")
fi

uv run python scripts/eval/eval_sae_on_val.py \
    --sae_root  "$SAE_ROOT" \
    --val_root  "$VAL_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"

echo
echo "=== eval_sae_on_val done: $(date) ==="
echo "--- aggregate.jsonl (first/last) ---"
head -2 "$OUTPUT_DIR/aggregate.jsonl" 2>/dev/null || true
tail -2 "$OUTPUT_DIR/aggregate.jsonl" 2>/dev/null || true
