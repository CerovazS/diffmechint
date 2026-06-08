"""Phase 4.5b — post-hoc per-feature activation histograms (Neuronpedia style).

For one already-rendered dashboard cell, re-run the SAE forward pass over the
matching activation shard, accumulate per-feature activation distributions
(max-per-image across the 256 tokens), and write a 30-bin histogram into the
existing `features/feature_<j>.json` under the new key
`activation_histogram`.

Why max-per-image (not per-token)? Neuronpedia-style histograms show how
strongly the feature fires across **examples**, not tokens. The dashboard
already uses per-image max (`max_act`) to pick the top-9 — using the same
quantity here keeps the histogram and the top-9 visually consistent.

Bins:
  - 30 linear bins from 0.0 to feature's per-image max activation (>0 only).
  - Zero count stored separately as `zero_count` so the visualisation can
    show the "feature did not fire" mass without dominating the x-axis.

Usage::

    uv run python scripts/analysis/recompute_histograms.py \\
        --condition repa_e --layer 3 --t_bin 2 \\
        --cell_dir outputs/phase4_5b_feature_viz_ynull/repa_e_L3_T2 \\
        --act_root /leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_ynull
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_FAST = "/leonardo_scratch/fast/IscrC_YENDRI"
os.environ.setdefault("HF_HOME", f"{_FAST}/lcerovaz/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", f"{_FAST}/lcerovaz/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_FAST}/lcerovaz/hf_cache/transformers")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from diffmechint.sae import load_matryoshka_sae, resolve_sae_ckpt  # noqa: E402
from diffmechint.utils import error, info, model_subdir, model_variant_spec, ok, warn  # noqa: E402

SAE_ROOT_DEFAULT = Path(
    "/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k"
)
SCRATCH_PROJECT_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint")
ACTIVATIONS_VAL_ROOT = SCRATCH_PROJECT_ROOT / "activations_val"
FAST_PROJECT_ROOT = SAE_ROOT_DEFAULT.parent
N_BINS = 30


@torch.no_grad()
def _compute_max_act(sae, shard_path: Path, *, batch_images: int,
                     device: torch.device) -> np.ndarray:
    """Per-image max SAE activation across the 256 tokens → (N, d_sae) fp16."""
    sae_dtype = next(sae.parameters()).dtype
    d_sae = int(sae.cfg.d_sae)
    with h5py.File(str(shard_path), "r") as f:
        acts = f["activations"]
        n, t, d = acts.shape
        info(f"shard {shard_path.name}: N={n} T={t} D={d}")
        max_act = np.zeros((n, d_sae), dtype=np.float16)
        n_batches = math.ceil(n / batch_images)
        t0 = time.perf_counter()
        for bi in range(n_batches):
            lo = bi * batch_images
            hi = min(lo + batch_images, n)
            chunk = acts[lo:hi]
            x = torch.from_numpy(chunk).to(device=device, dtype=sae_dtype, non_blocking=True)
            b = x.shape[0]
            z = sae.encode(x.reshape(b * t, d))
            z_b = z.float().view(b, t, d_sae)
            mx = z_b.max(dim=1).values
            max_act[lo:hi] = mx.to(torch.float16).cpu().numpy()
            if bi % 16 == 0 or bi == n_batches - 1:
                rate = (bi + 1) * batch_images / max(time.perf_counter() - t0, 1e-6)
                info(f"  batch {bi+1}/{n_batches}  ~{rate:.0f} img/s")
        ok(f"SAE pass done in {(time.perf_counter()-t0)/60:.2f} min")
    return max_act


def _build_histogram(col: np.ndarray, n_bins: int = N_BINS,
                     threshold: float = 1e-6) -> dict:
    """30-bin histogram over the strictly-positive activations in `col`.

    Returns:
        {
          "bins":   [b0, b1, ..., bN]   (N+1 bin edges, fp32 rounded to 6 dp)
          "counts": [c0, c1, ..., cN-1] (N non-negative ints)
          "zero_count": int,            # number of images where feature did not fire
          "max":   float,               # max activation
          "n":     int,                 # total samples (= zero_count + sum(counts))
        }
    """
    col = col.astype(np.float32)
    n = int(col.size)
    pos_mask = col > threshold
    pos = col[pos_mask]
    zero_count = int(n - pos_mask.sum())
    if pos.size == 0:
        return {
            "bins": [0.0, 0.0],
            "counts": [0],
            "zero_count": zero_count,
            "max": 0.0,
            "n": n,
        }
    mx = float(pos.max())
    edges = np.linspace(0.0, mx, n_bins + 1, dtype=np.float64)
    counts, _ = np.histogram(pos, bins=edges)
    return {
        "bins": [round(float(e), 6) for e in edges],
        "counts": [int(c) for c in counts],
        "zero_count": zero_count,
        "max": round(mx, 6),
        "n": n,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell_dir", type=Path, required=True)
    p.add_argument("--condition", required=True, choices=["sd_vae", "eq_vae", "repa_e"])
    p.add_argument("--model_variant", type=str, default="sit_b_2",
                   help="SiT variant id or model name; used for namespaced default roots.")
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--t_bin", required=True, type=int, choices=[0, 1, 2])
    p.add_argument("--dit_step", type=int, default=200_000)
    p.add_argument("--sae_root", type=Path, default=None)
    p.add_argument("--act_root", type=Path, default=None,
                   help="e.g. .../activations_val or .../activations_ynull")
    p.add_argument("--batch_images", type=int, default=64)
    p.add_argument("--n_bins", type=int, default=N_BINS)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N features (smoke testing).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing activation_histogram entries.")
    args = p.parse_args()
    spec = model_variant_spec(args.model_variant)
    if args.sae_root is None:
        args.sae_root = (
            SAE_ROOT_DEFAULT if spec.variant_id == "sit_b_2"
            else model_subdir(FAST_PROJECT_ROOT, spec.variant_id, "sae")
        )
    if args.act_root is None:
        namespaced = model_subdir(SCRATCH_PROJECT_ROOT, spec.variant_id, "activations_val")
        args.act_root = namespaced if namespaced.exists() or spec.variant_id != "sit_b_2" else ACTIVATIONS_VAL_ROOT

    cell_dir: Path = args.cell_dir
    feat_dir = cell_dir / "features"
    if not feat_dir.is_dir():
        error(f"feature dir missing: {feat_dir}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available — will be slow.")

    sae_ckpt = resolve_sae_ckpt(args.sae_root, args.condition, args.layer,
                                args.t_bin, args.dit_step)
    info(f"Loading SAE from {sae_ckpt}")
    sae = load_matryoshka_sae(sae_ckpt, device)
    d_sae = int(sae.cfg.d_sae)
    info(f"  d_in={sae.cfg.d_in} d_sae={d_sae} k={sae.cfg.k}")

    shard_path = (args.act_root / args.condition /
                  f"step_{args.dit_step:06d}" / f"{args.layer}_{args.t_bin}.h5")
    if not shard_path.exists():
        error(f"activation shard not found: {shard_path}")
        return 1

    info(f"Streaming SAE encode (batch_images={args.batch_images})…")
    max_act = _compute_max_act(sae, shard_path,
                               batch_images=args.batch_images, device=device)
    info(f"max_act shape={max_act.shape} dtype={max_act.dtype}")

    # Enumerate live features by reading existing JSONs.
    info(f"Scanning {feat_dir}…")
    feat_files = sorted(feat_dir.glob("feature_*.json"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if args.limit is not None:
        feat_files = feat_files[: args.limit]
        info(f"  --limit={args.limit} → {len(feat_files)} features")

    n_written = 0
    n_skipped = 0
    t0 = time.perf_counter()
    for fp in feat_files:
        try:
            rec = json.loads(fp.read_text())
        except Exception as exc:
            warn(f"unparseable {fp.name}: {exc}")
            continue
        if not args.force and "activation_histogram" in rec:
            n_skipped += 1
            continue
        j = int(rec["feature_id"])
        if j >= d_sae:
            warn(f"feature_id {j} >= d_sae {d_sae}, skipping")
            continue
        rec["activation_histogram"] = _build_histogram(max_act[:, j], n_bins=args.n_bins)
        fp.write_text(json.dumps(rec))
        n_written += 1
    ok(f"wrote {n_written} histograms (skipped {n_skipped} pre-existing) "
       f"in {(time.perf_counter()-t0)/60:.2f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
