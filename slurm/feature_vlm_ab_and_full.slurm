#!/bin/bash
#SBATCH --job-name=dmi_vlm_ab
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Phase 4.5b — VLM prompt A/B test + full monosemantic sweep in one job.
#
# Pass 1: caption a fixed feature set (FEATURES) with each prompt in
#         PROMPT_VERSIONS, writing rows to AB_LOG (not to per-feature JSONs).
# Pass 2: if RUN_FULL=1, caption every monosemantic feature with the
#         chosen WINNER prompt, writing back to per-feature JSONs and
#         appending to <cell>/vlm_log.jsonl.
#
# Env:
#   CELL_DIR=outputs/phase4_5b_feature_viz_ynull/<cond>_L<L>_T<T>  (required)
#   FEATURES=1002,10141,10347,10530,1059                           (A/B set)
#   PROMPT_VERSIONS=v1,v2,v3
#   AB_LOG=/tmp/vlm_prompt_ab.jsonl
#   WINNER=v3        # prompt version used for the full run
#   RUN_FULL=1       # 1 = also run on all 218 mono features
#   FORCE=1          # 1 = re-caption mono features that already have one

CELL_DIR="${CELL_DIR:?must export CELL_DIR before sbatch}"
FEATURES="${FEATURES:-1002,10141,10347,10530,1059}"
PROMPT_VERSIONS="${PROMPT_VERSIONS:-v1,v2,v3}"
AB_LOG="${AB_LOG:-/tmp/vlm_prompt_ab.jsonl}"
WINNER="${WINNER:-v3}"
RUN_FULL="${RUN_FULL:-1}"
FORCE="${FORCE:-1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

export HF_HOME="/leonardo_scratch/fast/IscrC_YENDRI/lcerovaz/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
echo "=== feature_vlm_ab CELL_DIR=$CELL_DIR ==="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}"
echo "FEATURES=$FEATURES  PROMPT_VERSIONS=$PROMPT_VERSIONS"
echo "AB_LOG=$AB_LOG  WINNER=$WINNER  RUN_FULL=$RUN_FULL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Pass 1 — A/B over fixed feature set, one prompt at a time. The script keeps
# the VLM weights loaded across features within a single invocation, but each
# Python launch reloads — accept the warm-up cost (3× ~30s) for code simplicity.
: > "$AB_LOG"   # fresh log for this A/B
IFS=',' read -ra VERS <<< "$PROMPT_VERSIONS"
for PV in "${VERS[@]}"; do
  echo
  echo ">>> A/B pass: prompt_version=$PV"
  uv run python scripts/analysis/feature_vlm_interp.py \
      --cell_dir "$CELL_DIR" \
      --model_id "$MODEL_ID" \
      --prompt_version "$PV" \
      --features "$FEATURES" \
      --log_path "$AB_LOG"
done

echo
echo "=== A/B summary ==="
cat "$AB_LOG"
echo "==================="

if [[ "$RUN_FULL" == "1" ]]; then
  echo
  echo ">>> Full pass: every monosemantic feature with prompt_version=$WINNER"
  # Reset the cell-level VLM log so the row count after this run equals the
  # number of monosemantic features captioned in this pass (expected 218).
  : > "$CELL_DIR/vlm_log.jsonl"
  EXTRA=()
  if [[ "$FORCE" == "1" ]]; then EXTRA+=(--force); fi
  uv run python scripts/analysis/feature_vlm_interp.py \
      --cell_dir "$CELL_DIR" \
      --model_id "$MODEL_ID" \
      --prompt_version "$WINNER" \
      "${EXTRA[@]}"
fi

echo
echo "=== feature_vlm_ab done: $(date) ==="
