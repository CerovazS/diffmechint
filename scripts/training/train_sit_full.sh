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
TRAINER_CONFIG="${TRAINER_CONFIG:-cineca_2xa100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAMPLE_EVERY="${SAMPLE_EVERY:-5000}"
SAMPLE_N="${SAMPLE_N:-16}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-}"
# NORMALIZE=true (default): per-feature z-score from stats.json. NORMALIZE=false:
# raw latents (only the AE's own scaling_factor). Disable when reproducing
# tokenizers whose recipe forbids extra normalization (e.g. EQ-VAE).
NORMALIZE="${NORMALIZE:-true}"
# Optional run-name suffix (e.g. "noz" for no-zscore ablations) to avoid collisions.
RUN_SUFFIX="${RUN_SUFFIX:-}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
SCRATCH_BASE=/leonardo_scratch/large/userexternal/lcerovaz
FAST_BASE=/leonardo_scratch/fast/IscrC_YENDRI
LATENTS="${SCRATCH_BASE}/diffmechint/latents/${TOK}"
# Tag run with model variant + optional suffix so variants don't collide on disk.
SUFFIX_TAG=""
[[ -n "$RUN_SUFFIX" ]] && SUFFIX_TAG="_${RUN_SUFFIX}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${MODEL}_${TOK}${SUFFIX_TAG}_full_${RUN_TIMESTAMP}}"
MODEL_VARIANT="${MODEL//\//_}"
case "$MODEL_VARIANT" in
    sit_s_*) MODEL_NAME="SiT-S/${MODEL_VARIANT##*_}" ;;
    sit_b_*) MODEL_NAME="SiT-B/${MODEL_VARIANT##*_}" ;;
    sit_l_*) MODEL_NAME="SiT-L/${MODEL_VARIANT##*_}" ;;
    sit_xl_*) MODEL_NAME="SiT-XL/${MODEL_VARIANT##*_}" ;;
    *) MODEL_NAME="$MODEL" ;;
esac
OUT_DIR="${SCRATCH_BASE}/diffmechint/by_model/${MODEL_VARIANT}/runs/${RUN_ID}"
CKPT_DIR="${OUT_DIR}/checkpoints"
RANK="${SLURM_PROCID:-0}"

module load cuda/12.2 || true

export HF_HOME="${FAST_BASE}/lcerovaz/hf_cache"
export HF_HUB_CACHE="${FAST_BASE}/lcerovaz/hf_cache/hub"
export TRANSFORMERS_CACHE="${FAST_BASE}/lcerovaz/hf_cache/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
if [[ "$RANK" == "0" ]]; then
    if [[ -e "$OUT_DIR" ]]; then
        echo "ERROR: refusing to reuse existing OUT_DIR=$OUT_DIR" >&2
        exit 2
    fi
    mkdir -p "$OUT_DIR" "$CKPT_DIR"
    GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    cat > "${OUT_DIR}/run_metadata.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "model_variant": "${MODEL_VARIANT}",
  "model_name": "${MODEL_NAME}",
  "model_config": "${MODEL}",
  "tokenizer": "${TOK}",
  "normalize": "${NORMALIZE}",
  "trainer_config": "${TRAINER_CONFIG}",
  "batch_size_per_device": "${BATCH_SIZE}",
  "val_batch_size": "${VAL_BATCH_SIZE}",
  "num_workers": "${NUM_WORKERS}",
  "max_steps_override": "${MAX_STEPS:-}",
  "val_check_interval_override": "${VAL_CHECK_INTERVAL}",
  "latents": "${LATENTS}",
  "out_dir": "${OUT_DIR}",
  "git_sha": "${GIT_SHA}",
  "git_branch": "${GIT_BRANCH}",
  "slurm_job_id": "${SLURM_JOB_ID:-local}"
}
EOF
    cat > "${OUT_DIR}/reproducibility.md" <<EOF
# Reproducibility

- Repo: ${REPO}
- Branch: ${GIT_BRANCH}
- Commit: ${GIT_SHA}
- SLURM job id: ${SLURM_JOB_ID:-local}
- Model: ${MODEL_NAME} (${MODEL})
- Tokenizer: ${TOK}
- Latents: ${LATENTS}
- Output directory: ${OUT_DIR}

## Environment

\`\`\`bash
module load cuda/12.2
source .venv/bin/activate
export HF_HOME=${FAST_BASE}/lcerovaz/hf_cache
export HF_HUB_CACHE=${FAST_BASE}/lcerovaz/hf_cache/hub
export TRANSFORMERS_CACHE=${FAST_BASE}/lcerovaz/hf_cache/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
\`\`\`

## Command

\`\`\`bash
sbatch --export=ALL,TOK=${TOK},MODEL=${MODEL},NORMALIZE=${NORMALIZE},RUN_SUFFIX=${RUN_SUFFIX},TRAINER_CONFIG=${TRAINER_CONFIG},BATCH_SIZE=${BATCH_SIZE},VAL_BATCH_SIZE=${VAL_BATCH_SIZE},NUM_WORKERS=${NUM_WORKERS},MAX_STEPS=${MAX_STEPS:-},CALLBACKS=${CALLBACKS},SAMPLE_EVERY=${SAMPLE_EVERY},VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL} slurm/train_sit_full.slurm
\`\`\`
EOF
    cat > "${OUT_DIR}/commit.txt" <<EOF
repo=${REPO}
branch=${GIT_BRANCH}
commit=${GIT_SHA}
job_id=${SLURM_JOB_ID:-local}
EOF
else
    for ((i = 0; i < 120; i++)); do
        [[ -f "${OUT_DIR}/run_metadata.json" ]] && break
        sleep 1
    done
    if [[ ! -f "${OUT_DIR}/run_metadata.json" ]]; then
        echo "ERROR: rank $RANK timed out waiting for rank 0 to create $OUT_DIR" >&2
        exit 2
    fi
fi

echo "=== train_sit_full $TOK start: $(date) ==="
echo "REPO=$REPO LATENTS=$LATENTS OUT_DIR=$OUT_DIR MODEL_VARIANT=$MODEL_VARIANT JOB_ID=${SLURM_JOB_ID:-local}"
echo "TRAINER_CONFIG=$TRAINER_CONFIG BATCH_SIZE=$BATCH_SIZE VAL_BATCH_SIZE=$VAL_BATCH_SIZE MAX_STEPS=${MAX_STEPS:-config-default}"
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

TRAINER_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
    TRAINER_ARGS+=("trainer.max_steps=${MAX_STEPS}")
fi
if [[ -n "${VAL_CHECK_INTERVAL}" ]]; then
    TRAINER_ARGS+=("trainer.val_check_interval=${VAL_CHECK_INTERVAL}")
fi

uv run python -m diffmechint.training.train \
    tokenizer="${TOK}" \
    model="${MODEL}" \
    transport=fm_ot \
    trainer="${TRAINER_CONFIG}" \
    callbacks="${CALLBACKS}" \
    callbacks.sample.adapter_name="${TOK}" \
    callbacks.sample.stats_path="${SAMPLE_STATS}" \
    callbacks.sample.every_n_steps="${SAMPLE_EVERY}" \
    callbacks.sample.n_samples="${SAMPLE_N}" \
    +data._target_=diffmechint.training.data.CachedLatentDataModule \
    +data.shard_dir="${LATENTS}" \
    +data.batch_size="${BATCH_SIZE}" \
    +data.num_workers="${NUM_WORKERS}" \
    +data.normalize="${NORMALIZE}" \
    +data.holdout_fraction=0.05 \
    +data.holdout_seed=42 \
    +data.val_batch_size="${VAL_BATCH_SIZE}" \
    +ckpt_dir="${CKPT_DIR}" \
    "${TRAINER_ARGS[@]}" \
    "${RESUME_ARGS[@]}"

echo
if [[ "$RANK" == "0" ]]; then
    echo "=== train_sit_full $TOK complete: $(date) ==="
    echo "ckpt count:"
    find "$CKPT_DIR" -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null | wc -l
    echo "samples count:"
    if [[ -d "${OUT_DIR}/samples" ]]; then
        find "${OUT_DIR}/samples" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l
    else
        echo 0
    fi
fi
