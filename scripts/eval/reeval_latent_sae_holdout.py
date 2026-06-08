"""Re-evaluate trained latent SAEs at increasing holdout sizes (E30b, dead% vs data).

E25 showed dead-feature rates drop 10-18pp when the evaluation holdout grows.
This script tests that measurement hypothesis on the Phase 4.19 latent SAEs
without retraining: load each requested cell's final checkpoint and evaluate
EV / dead% / live features on nested token prefixes of the full ImageNet-val
latent precompute (all shards, ~12.8M tokens per condition).

Usage:
    uv run python scripts/eval/reeval_latent_sae_holdout.py \
        --run_id latent_sae_ef_20260606 --cells d256_k8 \
        --sizes 200000 500000 1000000 2000000 5000000 12800000 \
        --out_dir outputs/<run_root>/holdout_reeval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from diffmechint.sae import evaluate_sae_on_tokens, load_latent_val_tokens
from diffmechint.utils import info, ok, warn

SCRATCH_PROJECT_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint")


def load_latent_sae(cell_dir: Path, device: str):
    """Load the final SAE checkpoint of one latent cell, variant-aware."""
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
    return cls.load_from_disk(str(ckpt), device=device), ckpt, cfg


def val_tokens_all_shards(condition: str, val_root: Path, patch_size: int, max_tokens: int):
    """Concatenate patchified tokens across all val shards up to `max_tokens`."""
    chunks, total = [], 0
    for shard in sorted((val_root / condition).glob("*.h5")):
        t = load_latent_val_tokens(shard, patch_size=patch_size)
        chunks.append(t)
        total += t.shape[0]
        if total >= max_tokens:
            break
    return torch.cat(chunks)[:max_tokens].contiguous()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_id", type=str, required=True)
    p.add_argument("--cells", nargs="+", default=["d256_k8"])
    p.add_argument("--conditions", nargs="+", default=["sd_vae", "eq_vae", "repa_e"])
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 12_800_000])
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--sae_base", type=str, default=str(SCRATCH_PROJECT_ROOT / "sae_latents"))
    p.add_argument("--val_root", type=str, default=str(SCRATCH_PROJECT_ROOT / "latents_val"))
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in args.conditions:
        max_size = max(args.sizes)
        tokens = val_tokens_all_shards(
            condition, Path(args.val_root), args.patch_size, max_size
        )
        info(f"{condition}: loaded {tokens.shape[0]} val tokens (requested {max_size})")
        for cell in args.cells:
            cell_dir = Path(args.sae_base) / args.run_id / condition / cell
            if not cell_dir.exists():
                warn(f"missing cell {cell_dir}, skipping")
                continue
            sae, ckpt, _ = load_latent_sae(cell_dir, args.device)
            for size in args.sizes:
                n = min(size, tokens.shape[0])
                m = evaluate_sae_on_tokens(sae, tokens[:n], batch_size=4096)
                rows.append({
                    "condition": condition,
                    "cell": cell,
                    "checkpoint": ckpt.name,
                    "holdout_tokens": n,
                    "ev": m["ev"],
                    "dead_pct": m["dead_pct"],
                    "live_features": m["live_features"],
                    "l0_mean": m["l0_mean"],
                })
                info(
                    f"  {condition}/{cell} @ {n:>9} tokens: "
                    f"EV={m['ev']:.4f} dead%={m['dead_pct'] * 100:.1f} live={m['live_features']}"
                )

    out = args.out_dir / "holdout_reeval.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    ok(f"{len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
