"""E31: monosemanticity atlas of latent-token SAE features (E07/E11 protocol).

Encodes the full ImageNet-val latent precompute (50 080 images x 256 tokens)
through the selected latent SAE per condition, then scores every feature:
token-level density, top-9 activating images (per-image max activation), and
class entropy of the top-9 labels. Monosemantic = density in [1e-4, 1e-1]
AND entropy < 2.5 bits — identical thresholds to the E11 DiT atlas.

Usage:
    uv run python scripts/analysis/latent_feature_atlas.py \
        --run_id latent_sae_ef_20260606 --cell d256_k8 \
        --out_dir outputs/<run_root>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import torch

from diffmechint.analysis.latent_atlas import score_features
from diffmechint.sae import patchify_latents
from diffmechint.utils import info, ok

SCRATCH_PROJECT_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint")


def load_latent_sae(cell_dir: Path, device: str):
    """Load the final SAE checkpoint of one latent cell, variant-aware.

    Mirrors scripts/eval/reeval_latent_sae_holdout.py (scripts are not a
    package, so the small loader is duplicated rather than imported).
    """
    from sae_lens import (
        BatchTopKTrainingSAE,
        MatryoshkaBatchTopKTrainingSAE,
        TopKTrainingSAE,
    )

    finals = sorted(cell_dir.glob("final_*"))
    if not finals:
        raise FileNotFoundError(f"no final_* checkpoint under {cell_dir}")
    ckpt = finals[-1]
    cfg = json.loads((ckpt / "cfg.json").read_text())
    arch = cfg.get("architecture", "")
    cls = (
        MatryoshkaBatchTopKTrainingSAE
        if "matryoshka" in arch
        else BatchTopKTrainingSAE
        if "batch" in arch
        else TopKTrainingSAE
    )
    return cls.load_from_disk(str(ckpt), device=device), ckpt


@torch.no_grad()
def scan_condition(
    condition: str,
    sae,
    val_root: Path,
    patch_size: int,
    device: str,
    batch_size: int = 4096,
    act_threshold: float = 1e-6,
) -> dict:
    """Stream all val shards; return per-feature stats + per-image max matrix."""
    d_sae = int(sae.cfg.d_sae)
    sae_dtype = next(sae.parameters()).dtype
    img_max_chunks, img_labels = [], []
    active_tokens = torch.zeros(d_sae, dtype=torch.float64, device=device)
    total_tokens = 0

    for shard in sorted((val_root / condition).glob("*.h5")):
        with h5py.File(shard, "r") as f:
            latents = torch.from_numpy(f["latents"][()]).float()
            labels = torch.from_numpy(f["labels"][()]).long()
        tokens = patchify_latents(latents, patch_size)  # (N, T, D)
        n, t, d = tokens.shape
        flat = tokens.reshape(-1, d)
        maxes = []
        for start in range(0, flat.shape[0], batch_size):
            x = flat[start : start + batch_size].to(device=device, dtype=sae_dtype)
            z = sae.encode(x).float()  # (B, d_sae)
            active_tokens += (z.abs() > act_threshold).sum(dim=0).to(torch.float64)
            maxes.append(z.cpu())
        z_all = torch.cat(maxes).reshape(n, t, d_sae)
        img_max_chunks.append(z_all.max(dim=1).values)  # (N, d_sae)
        img_labels.append(labels)
        total_tokens += n * t
        info(f"  {condition}: scanned {shard.name} ({n} images)")

    img_max = torch.cat(img_max_chunks)  # (N_total, d_sae)
    labels = torch.cat(img_labels)  # (N_total,)
    density = (active_tokens / max(total_tokens, 1)).cpu()
    return {
        "img_max": img_max,
        "labels": labels,
        "density": density,
        "total_tokens": total_tokens,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_id", type=str, default="latent_sae_ef_20260606")
    p.add_argument("--cell", type=str, default="d256_k8")
    p.add_argument("--conditions", nargs="+", default=["sd_vae", "eq_vae", "repa_e"])
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--sae_base", type=str, default=str(SCRATCH_PROJECT_ROOT / "sae_latents"))
    p.add_argument("--val_root", type=str, default=str(SCRATCH_PROJECT_ROOT / "latents_val"))
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--density_band", type=float, nargs=2, default=[1e-4, 1e-1])
    p.add_argument("--entropy_threshold", type=float, default=2.5)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    (args.out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    summary = {}
    for condition in args.conditions:
        cell_dir = Path(args.sae_base) / args.run_id / condition / args.cell
        sae, ckpt = load_latent_sae(cell_dir, args.device)
        info(f"{condition}: loaded {ckpt}")
        scan = scan_condition(
            condition, sae, Path(args.val_root), args.patch_size, args.device
        )
        rows = score_features(
            scan,
            density_lo=args.density_band[0],
            density_hi=args.density_band[1],
            entropy_threshold=args.entropy_threshold,
        )
        out_csv = args.out_dir / "metrics" / f"latent_atlas_{condition}.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        live = sum(r["live"] for r in rows)
        mono = sum(r["monosemantic"] for r in rows)
        in_band = sum(r["in_density_band"] for r in rows)
        summary[condition] = {
            "checkpoint": str(ckpt),
            "total_features": len(rows),
            "live_features": live,
            "in_density_band": in_band,
            "monosemantic": mono,
            "total_tokens": scan["total_tokens"],
        }
        ok(
            f"{condition}: live={live} in_band={in_band} "
            f"monosemantic={mono} → {out_csv}"
        )

    out_json = args.out_dir / "metrics" / "latent_atlas_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    ok(f"summary → {out_json}")


if __name__ == "__main__":
    main()
