#!/bin/bash
#SBATCH --job-name=dmi_cell_stability
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Cross-seed cell-level stability analysis.
#
# Required:
#   COMMAND=pooled-probes | subspace
#   OUT_DIR=outputs/phase4_9_seed_stability/<unique_run_id>
#
# Optional:
#   CELLS="eq_vae_L9_T0 sd_vae_L3_T1"
#   MAX_IMAGES=1000
#   MAX_PROBE_FEATURES=4096
#   MAX_TOKENS=65536
#   CONCEPTS="animal_binary broad_8 vehicle_binary food_binary instrument_binary"

COMMAND="${COMMAND:?must export COMMAND before sbatch}"
OUT_DIR="${OUT_DIR:?must export OUT_DIR before sbatch}"
DIT_STEP="${DIT_STEP:-200000}"
MAX_IMAGES="${MAX_IMAGES:-1000}"
MAX_PROBE_FEATURES="${MAX_PROBE_FEATURES:-4096}"
BATCH_IMAGES="${BATCH_IMAGES:-8}"
BATCH_TOKENS="${BATCH_TOKENS:-4096}"
MAX_TOKENS="${MAX_TOKENS:-65536}"
TOP_DIMS="${TOP_DIMS:-128}"
POOL="${POOL:-max}"
CONCEPTS="${CONCEPTS:-animal_binary broad_8 vehicle_binary food_binary instrument_binary}"
CELLS="${CELLS:-}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

export HF_HOME="/leonardo_scratch/fast/IscrC_YENDRI/lcerovaz/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
echo "=== cell_stability start: $(date) ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "COMMAND=$COMMAND"
echo "OUT_DIR=$OUT_DIR"
echo "MAX_IMAGES=$MAX_IMAGES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

EXTRA_ARGS=()
if [[ -n "$CELLS" ]]; then
    read -r -a CELL_ARRAY <<< "$CELLS"
    EXTRA_ARGS+=("--cells" "${CELL_ARRAY[@]}")
fi

if [[ "$COMMAND" == "pooled-probes" ]]; then
    read -r -a CONCEPT_ARRAY <<< "$CONCEPTS"
    uv run python scripts/analysis/cell_stability.py pooled-probes \
        --out_dir "$OUT_DIR" \
        --dit_step "$DIT_STEP" \
        --max_images "$MAX_IMAGES" \
        --max_probe_features "$MAX_PROBE_FEATURES" \
        --batch_images "$BATCH_IMAGES" \
        --pool "$POOL" \
        --concepts "${CONCEPT_ARRAY[@]}" \
        "${EXTRA_ARGS[@]}"
elif [[ "$COMMAND" == "subspace" ]]; then
    uv run python scripts/analysis/cell_stability.py subspace \
        --out_dir "$OUT_DIR" \
        --dit_step "$DIT_STEP" \
        --max_images "$MAX_IMAGES" \
        --max_tokens "$MAX_TOKENS" \
        --batch_tokens "$BATCH_TOKENS" \
        --top_dims "$TOP_DIMS" \
        "${EXTRA_ARGS[@]}"
else
    echo "Unknown COMMAND=$COMMAND" >&2
    exit 2
fi

echo
echo "=== cell_stability done: $(date) ==="
find "$OUT_DIR" -maxdepth 1 -type f -print | sort
