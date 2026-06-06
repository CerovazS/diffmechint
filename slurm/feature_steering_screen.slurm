#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_PDR/lcerovaz/diffmechint
#SBATCH --job-name=feature_steering_screen
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_PDR

set -euo pipefail

module load cuda/12.2

TASK_FILE="${TASK_FILE:?TASK_FILE is required}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

export HF_TOKEN="${HF_TOKEN:-$(grep HF_TOKEN ~/.bashrc | grep -oP '(?<=")hf_[^"]+(?=")' | head -n 1)}"
export HF_HOME=/leonardo_scratch/fast/IscrC_YENDRI/lcerovaz/hf_cache
export HF_HUB_CACHE=/leonardo_scratch/fast/IscrC_YENDRI/lcerovaz/hf_cache/hub

COMMAND="$(
  uv run python - "$TASK_FILE" "$TASK_INDEX" <<'PY'
import csv
import sys
from pathlib import Path

task_file = Path(sys.argv[1])
task_index = int(sys.argv[2])
with task_file.open(newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh, delimiter="\t"))
if task_index < 0 or task_index >= len(rows):
    raise SystemExit(f"task index {task_index} out of range for {len(rows)} rows")
print(rows[task_index]["command"])
PY
)"

echo "TASK_FILE=${TASK_FILE}"
echo "TASK_INDEX=${TASK_INDEX}"
echo "COMMAND=${COMMAND}"
eval "${COMMAND}"
