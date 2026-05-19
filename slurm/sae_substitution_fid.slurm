#!/bin/bash
#SBATCH --job-name=dmi_sub_fid
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_PDR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
set -euo pipefail

# Phase 4.5a — substitution-FID for one (condition, layer, t_bin) cell, or a
# per-condition baseline if MODE=baseline.
#
# Inputs (export before sbatch):
#   RUN=<SiT run dir>
#   ADAPTER=<sd_vae | repa_e | eq_vae>
#   CONDITION=<sd_vae | repa_e | eq_vae>
#   MODE=<sub | baseline>
#   For MODE=sub also: LAYER=<3|6|9> T_BIN=<0|1|2>
# Optional:
#   DIT_STEP=200000
#   N_SAMPLES=5000
#   NORMALIZE=true|false  (set false for eq_vae_noz adapter latents)
#   OUT_ROOT=outputs/phase4_5a_subst_fid
#   SAE_ROOT=/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k
#   SEED=0

RUN="${RUN:?must export RUN before sbatch}"
ADAPTER="${ADAPTER:?must export ADAPTER before sbatch}"
CONDITION="${CONDITION:?must export CONDITION before sbatch}"
MODE="${MODE:?must export MODE (sub | baseline) before sbatch}"
DIT_STEP="${DIT_STEP:-200000}"
N_SAMPLES="${N_SAMPLES:-5000}"
NORMALIZE="${NORMALIZE:-true}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-outputs/phase4_5a_subst_fid}"
SAE_ROOT="${SAE_ROOT:-/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k}"

REPO=/leonardo_work/IscrC_PDR/lcerovaz/diffmechint
module load cuda/12.2 || true

# Cleanfid inception symlink — no internet on compute nodes.
INCEPTION_HOME="${HOME}/.cache/cleanfid_models/inception-2015-12-05.pt"
ln -sf "${INCEPTION_HOME}" /tmp/inception-2015-12-05.pt
echo "Inception symlinked: $(ls -lh /tmp/inception-2015-12-05.pt)"

cd "$REPO"
echo "=== sae_substitution_fid CONDITION=$CONDITION MODE=$MODE ==="
echo "RUN=$RUN  ADAPTER=$ADAPTER  DIT_STEP=$DIT_STEP  N=$N_SAMPLES  NORMALIZE=$NORMALIZE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

EXTRA_ARGS=()
if [[ "$NORMALIZE" == "false" ]]; then
    EXTRA_ARGS+=("--no_normalize")
    echo "NORMALIZE=false → passing --no_normalize"
fi

if [[ "$MODE" == "baseline" ]]; then
    EXTRA_ARGS+=("--no_substitute")
    echo "MODE=baseline → no SAE substitution"
elif [[ "$MODE" == "sub" ]]; then
    LAYER="${LAYER:?must export LAYER for MODE=sub}"
    T_BIN="${T_BIN:?must export T_BIN for MODE=sub}"
    EXTRA_ARGS+=("--layer" "$LAYER" "--t_bin" "$T_BIN")
    echo "MODE=sub  LAYER=$LAYER  T_BIN=$T_BIN"
else
    echo "ERROR: unknown MODE=$MODE (expected 'sub' or 'baseline')" >&2
    exit 2
fi

uv run python scripts/eval/sae_substitution_fid.py \
    --run "$RUN" \
    --adapter "$ADAPTER" \
    --condition "$CONDITION" \
    --sae_root "$SAE_ROOT" \
    --dit_step "$DIT_STEP" \
    --n_samples "$N_SAMPLES" \
    --seed "$SEED" \
    --out_root "$OUT_ROOT" \
    "${EXTRA_ARGS[@]}"

echo
echo "=== sae_substitution_fid done: $(date) ==="
