#!/bin/bash
#SBATCH --job-name=dmi_extract_activations
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Inputs (must export before sbatch):
#   RUN=<absolute path to a Phase 2 run dir>
#   CONDITION=<tokenizer name: sd_vae | repa_e | eq_vae>
# Optional:
#   N_SAMPLES (default 2000)
#   BATCH_SIZE (default 64)
#   NORMALIZE=true|false (default true). Set false to match runs trained with
#     +data.normalize=false (e.g. eq_vae_noz).
#   MODEL_NAME (default SiT-B/2)
RUN="${RUN:?must export RUN before sbatch}"
CONDITION="${CONDITION:?must export CONDITION before sbatch}"
N_SAMPLES="${N_SAMPLES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NORMALIZE="${NORMALIZE:-true}"
MODEL_NAME="${MODEL_NAME:-SiT-B/2}"
# When set, overrides N_SAMPLES with stratified sampling (M per class).
STRATIFIED="${STRATIFIED:-}"
# Output root (default is the canonical path; override to write to a parallel
# tree e.g. `activations_s20` without overwriting an existing extraction).
OUT_ROOT="${OUT_ROOT:-/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations}"
# Latents root (default is the canonical train-encoded path; override to
# `latents_val` to source activations from val-encoded latents).
LATENTS_ROOT="${LATENTS_ROOT:-/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents}"
# When LATENTS_ROOT points at val-encoded latents, set NO_HOLDOUT=true to use
# the full set (the entire val set is already OOD for the SiT).
NO_HOLDOUT="${NO_HOLDOUT:-false}"
# When Y_NULL=true, replace the class label with the null token in forward
# (removes y_embedder contribution; manifest labels remain the true labels).
Y_NULL="${Y_NULL:-false}"
# Optional: ONLY_STEP=<int> to extract only that single checkpoint; BINS to
# restrict to a subset of diffusion-time bins (space-separated floats).
ONLY_STEP="${ONLY_STEP:-}"
BINS="${BINS:-}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint

module load cuda/12.2 || true

cd "$REPO"
echo "=== extract_activations CONDITION=$CONDITION RUN=$(basename "$RUN") start: $(date) ==="
echo "host: $(hostname) job: ${SLURM_JOB_ID:-local}"

# Pre-flight: CINECA boost nodes occasionally come up with no visible GPU
# (kernel/driver glitch); fail loudly so the log makes the cause obvious.
if ! nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | grep -q "MiB"; then
    echo "ERROR: nvidia-smi sees no GPU on $(hostname). Re-submit this job; node is unhealthy." >&2
    exit 6
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv

EXTRA_ARGS=()
if [[ "$NORMALIZE" == "false" ]]; then
    EXTRA_ARGS+=("--no_normalize")
    echo "NORMALIZE=false → passing --no_normalize"
fi
if [[ -n "$STRATIFIED" ]]; then
    EXTRA_ARGS+=("--stratified" "$STRATIFIED")
    echo "STRATIFIED=$STRATIFIED → passing --stratified $STRATIFIED (overrides N_SAMPLES)"
fi
if [[ "$NO_HOLDOUT" == "true" ]]; then
    EXTRA_ARGS+=("--no_holdout")
    echo "NO_HOLDOUT=true → passing --no_holdout (use entire shard set)"
fi
if [[ "$Y_NULL" == "true" ]]; then
    EXTRA_ARGS+=("--y_null")
    echo "Y_NULL=true → passing --y_null (null-token forward, residual without y_embedder)"
fi
if [[ -n "$ONLY_STEP" ]]; then
    EXTRA_ARGS+=("--only_step" "$ONLY_STEP")
    echo "ONLY_STEP=$ONLY_STEP → restricting extraction to single checkpoint"
fi
if [[ -n "$BINS" ]]; then
    # BINS is a space-separated list of floats; pass as multiple args.
    read -ra BIN_ARR <<< "$BINS"
    EXTRA_ARGS+=("--bins" "${BIN_ARR[@]}")
    echo "BINS=$BINS → restricting diffusion-time bins"
fi

uv run python scripts/extraction/extract_activations.py \
    "$RUN" "$CONDITION" \
    --model_name "$MODEL_NAME" \
    --n_samples "$N_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --out_root "$OUT_ROOT" \
    --latents_root "$LATENTS_ROOT" \
    "${EXTRA_ARGS[@]}"

echo
echo "=== extract_activations done: $(date) ==="
echo "--- output tree ($OUT_ROOT/$CONDITION) ---"
ls -la "$OUT_ROOT/$CONDITION/"
