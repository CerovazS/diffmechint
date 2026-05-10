#!/bin/bash
# One-shot prefetch: download cleanfid's Inception-v3 weights + build reference
# stats on ImageNet val 50k. After this, the MiniFIDCallback finds everything
# in ~/.cache/cleanfid/ and runs offline. ~5 min on 1 A100.
set -euo pipefail

REPO=/leonardo/home/userexternal/lcerovaz/diffmechint
FAST_BASE=/leonardo_scratch/fast/IscrC_YENDRI
REF_DIR=/leonardo_scratch/fast/IscrC_YENDRI/imagenet/val
REF_NAME=imagenet_val_50k

module load cuda/12.2 || true

# cleanfid hardcodes `/tmp/inception-2015-12-05.pt` and would re-download the
# weights from NVIDIA CDN on every run — fails on compute nodes (no internet).
# Workaround: symlink our pre-downloaded HOME copy into /tmp before any
# cleanfid call, so download=True is a no-op (file already exists).
INCEPTION_HOME="${HOME}/.cache/cleanfid_models/inception-2015-12-05.pt"
ln -sf "${INCEPTION_HOME}" /tmp/inception-2015-12-05.pt
echo "Inception symlinked: $(ls -lh /tmp/inception-2015-12-05.pt)"

# HF cache (irrelevant here but keeps the env consistent).
export HF_HOME="${FAST_BASE}/lcerovaz/hf_cache"

cd "$REPO"

echo "=== prefetch cleanfid start: $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

uv run python - <<EOF
from cleanfid import fid
print("Building cleanfid reference stats for '$REF_NAME' from $REF_DIR ...")
fid.make_custom_stats("$REF_NAME", "$REF_DIR", mode="clean")
print("Verifying cache:")
print("  test_stats_exists:", fid.test_stats_exists("$REF_NAME", mode="clean"))
EOF

echo
echo "=== cleanfid cache contents ==="
ls -lh ~/.cache/cleanfid/ 2>/dev/null || echo "(no ~/.cache/cleanfid)"
ls -lh ~/.cache/cleanfid/stats/ 2>/dev/null || echo "(no stats subdir)"

echo
echo "=== prefetch cleanfid done: $(date) ==="
