"""Phase 4.19 driver: Matryoshka SAEs on tokenizer latents (EF ablation, E30).

Trains one Matryoshka BatchTopK SAE per (condition x d_sae x k) cell directly
on the Phase 1 precomputed latents, patchified to SiT x_embedder token order
(patch 2 -> d_in = C*4 = 16 for the 4-channel VAEs). Recipe constants follow
the E04/E23 DiT SAE grid: batch 4096, 81.92M tokens, lr 3e-4, warmup 200,
inline held-out validation every 1500 steps, seed 0.

Usage (single cell, as one SLURM array task):
    uv run python scripts/training/train_latent_sae.py \
        --conditions sd_vae --d_sae_list 768 --k_list 8 \
        --run_id latent_sae_ef_20260606_120000

Usage (full local grid, e.g. interactive smoke):
    uv run python scripts/training/train_latent_sae.py \
        --conditions sd_vae eq_vae repa_e \
        --d_sae_list 256 768 2048 8192 --k_list 8 16 --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("WANDB_ENTITY", "ar_spectra")
os.environ.setdefault("WANDB_PROJECT", "diffmechint")

import torch

from diffmechint.sae import (
    build_sae,
    evaluate_sae_on_tokens,
    first_latent_d_in,
    latent_hdf5_provider,
    load_latent_val_tokens,
    train_sae,
)
from diffmechint.utils import error, info, ok, warn

SCRATCH_PROJECT_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint")
DEFAULT_LATENTS_ROOT = SCRATCH_PROJECT_ROOT / "latents"
DEFAULT_VAL_ROOT = SCRATCH_PROJECT_ROOT / "latents_val"
DEFAULT_OUT_BASE = SCRATCH_PROJECT_ROOT / "sae_latents"


def matryoshka_widths_for(d_sae: int) -> tuple[int, ...]:
    """E04 nested-prefix ratios (1/8, 1/4, 1/2, 1) scaled to `d_sae`."""
    if d_sae % 8:
        raise ValueError(f"d_sae={d_sae} must be divisible by 8 for 1/8..1 widths.")
    return (d_sae // 8, d_sae // 4, d_sae // 2, d_sae)


def train_one_cell(
    *,
    condition: str,
    d_sae: int,
    k: int,
    d_in: int,
    variant: str,
    args: argparse.Namespace,
    out_root: Path,
) -> dict:
    """Train + final-eval one (condition, d_sae, k, variant) latent SAE; return summary row."""
    cell = f"d{d_sae}_k{k}" if variant == "matryoshka" else f"d{d_sae}_k{k}_{variant}"
    out_dir = out_root / condition / cell
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty cell directory: {out_dir}")

    train_shards = sorted((Path(args.latents_root) / condition).glob("*.h5"))
    val_shards = sorted((Path(args.val_root) / condition).glob("*.h5"))
    if not train_shards:
        raise FileNotFoundError(f"no train latent shards under {args.latents_root}/{condition}")
    if not val_shards:
        raise FileNotFoundError(f"no val latent shards under {args.val_root}/{condition}")

    torch.manual_seed(args.seed)
    widths = matryoshka_widths_for(d_sae) if variant == "matryoshka" else None
    sae = build_sae(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        variant=variant,
        device=args.device,
        matryoshka_widths=widths,
        metadata={
            "condition": condition,
            "source": "latent_clean",
            "patch_size": int(args.patch_size),
            "d_in": int(d_in),
            "d_sae": int(d_sae),
            "k": int(k),
            "variant": variant,
            "expansion_factor": d_sae / d_in,
            "matryoshka_widths": list(widths) if widths else None,
            "seed": int(args.seed),
        },
    )
    provider = latent_hdf5_provider(
        train_shards,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        device=args.device,
        seed=args.seed,
    )
    val_tokens = load_latent_val_tokens(
        val_shards[0], patch_size=args.patch_size, max_tokens=args.val_max_tokens
    )
    info(
        f"cell {condition}/{cell}: variant={variant} d_in={d_in} widths={widths} "
        f"train_shards={len(train_shards)} val_tokens={tuple(val_tokens.shape)}"
    )

    group = f"latent_sae_{args.run_id}"
    t0 = time.time()
    train_sae(
        sae,
        provider,
        out_dir=out_dir,
        total_training_samples=args.total_samples,
        train_batch_size_samples=args.batch_size,
        lr=args.lr,
        lr_warm_up_steps=args.lr_warm_up_steps,
        device=args.device,
        log_to_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=os.environ.get("WANDB_ENTITY"),
        wandb_group=group,
        wandb_run_name=f"{group}_{condition}_{cell}",
        val_tokens=val_tokens,
        val_every_n_steps=args.val_every_n_steps,
        val_batch_size=args.batch_size,
    )
    wall_s = time.time() - t0

    # Final deterministic eval on the full held-out token set.
    final = evaluate_sae_on_tokens(sae, val_tokens, batch_size=args.batch_size)
    row = {
        "condition": condition,
        "d_in": d_in,
        "d_sae": d_sae,
        "k": k,
        "variant": variant,
        "expansion_factor": round(d_sae / d_in, 2),
        "patch_size": args.patch_size,
        "total_samples": args.total_samples,
        "seed": args.seed,
        "wall_s": round(wall_s, 1),
        **{f"val_{key}": val for key, val in final.items()},
    }
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics" / "final_eval.json").write_text(json.dumps(row, indent=2))
    ok(
        f"cell {condition}/{cell} done in {wall_s / 60:.1f} min — "
        f"val EV={final['ev']:.4f} dead%={final['dead_pct'] * 100:.1f} l0={final['l0_mean']:.1f}"
    )
    return row


def write_reproducibility(out_root: Path, args: argparse.Namespace, grid: list[tuple]) -> None:
    sha = branch = url = "unknown"
    try:
        repo = Path(__file__).resolve().parents[2]
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, text=True
        ).strip()
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=repo, text=True
        ).strip()
    except Exception as e:
        warn(f"git metadata unavailable: {e}")
    (out_root / "commit.txt").write_text(f"sha: {sha}\nbranch: {branch}\nrepo: {url}\n")
    cells = "\n".join(f"  - {c}/d{d}_k{k} ({v})" for c, d, k, v in grid)
    (out_root / "reproducibility.md").write_text(
        f"""# Reproducibility — Phase 4.19 latent SAE EF ablation (E30)

- Repo: {url} · branch `{branch}` · commit `{sha}`
- Command (per cell):

```bash
uv run python scripts/training/train_latent_sae.py \\
    --conditions <cond> --d_sae_list <d_sae> --k_list <k> \\
    --run_id {args.run_id} --total_samples {args.total_samples} \\
    --batch_size {args.batch_size} --lr {args.lr} --seed {args.seed} \\
    --patch_size {args.patch_size}
```

- Environment: CINECA Leonardo, 1x A100 64GB (`boost_usr_prod`,
  account `IscrC_PDR`), `module load cuda/12.2`, `uv run` from the repo venv.
- Train data: `{args.latents_root}/<cond>/*.h5` (fp16 `latents` (N,C,H,W),
  patchified p={args.patch_size} to SiT x_embedder token order, d_in = C*p^2).
- Val data: first shard of `{args.val_root}/<cond>/`, first
  {args.val_max_tokens} tokens, deterministic; inline eval every
  {args.val_every_n_steps} optimiser steps.
- SAE: Matryoshka BatchTopK, widths (d/8, d/4, d/2, d),
  `normalize_activations=expected_average_only_in`, seed {args.seed}.
- Grid trained under this root:
{cells}
"""
    )


def aggregate(out_root: Path) -> Path:
    """Scan all cells' final_eval.json into one aggregate.csv."""
    rows = [
        json.loads(p.read_text())
        for p in sorted(out_root.glob("*/d*_k*/metrics/final_eval.json"))
    ]
    out = out_root / "aggregate.csv"
    if rows:
        keys = list(rows[0].keys())
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        ok(f"aggregate: {len(rows)} cells → {out}")
    else:
        warn("aggregate: no final_eval.json found")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conditions", nargs="+", default=["sd_vae", "eq_vae", "repa_e"])
    p.add_argument("--d_sae_list", type=int, nargs="+", default=[256, 768, 2048, 8192])
    p.add_argument("--k_list", type=int, nargs="+", default=[8, 16])
    p.add_argument("--variants", nargs="+", default=["matryoshka"],
                   choices=["matryoshka", "batch_topk", "topk"],
                   help="SAE family axis; batch_topk isolates the matryoshka "
                        "dead-by-design effect (E04).")
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--total_samples", type=int, default=81_920_000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_warm_up_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latents_root", type=str, default=str(DEFAULT_LATENTS_ROOT))
    p.add_argument("--val_root", type=str, default=str(DEFAULT_VAL_ROOT))
    p.add_argument("--val_every_n_steps", type=int, default=1500)
    p.add_argument("--val_max_tokens", type=int, default=200_000)
    p.add_argument("--run_id", type=str, default=None,
                   help="Shared id so SLURM array tasks write into one root. "
                        "Default: latent_sae_ef_<timestamp>.")
    p.add_argument("--out_base", type=str, default=str(DEFAULT_OUT_BASE))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wandb_project", type=str,
                   default=os.environ.get("WANDB_PROJECT", "diffmechint"))
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="2.05M tokens (500 steps), val every 100 steps.")
    p.add_argument("--aggregate_only", action="store_true",
                   help="Only rebuild aggregate.csv for --run_id, no training.")
    args = p.parse_args()

    if args.run_id is None:
        args.run_id = f"latent_sae_ef_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.smoke:
        args.total_samples = 2_048_000
        args.val_every_n_steps = 100
    out_root = Path(args.out_base) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(out_root)
        return

    grid = [
        (c, d, k, v)
        for c in args.conditions
        for d in args.d_sae_list
        for k in args.k_list
        for v in args.variants
    ]
    info(
        f"run_id={args.run_id} grid={len(grid)} cells "
        f"({len(args.conditions)} cond x {len(args.d_sae_list)} d_sae x {len(args.k_list)} k "
        f"x {len(args.variants)} variant) "
        f"budget={args.total_samples} tokens/cell → {out_root}"
    )
    write_reproducibility(out_root, args, grid)

    rows, failures = [], []
    for condition, d_sae, k, variant in grid:
        d_in = first_latent_d_in(
            sorted((Path(args.latents_root) / condition).glob("*.h5")), args.patch_size
        )
        try:
            rows.append(
                train_one_cell(
                    condition=condition, d_sae=d_sae, k=k, d_in=d_in, variant=variant,
                    args=args, out_root=out_root,
                )
            )
        except Exception as e:
            error(f"cell {condition}/d{d_sae}_k{k}_{variant} FAILED: {e}")
            failures.append((condition, d_sae, k, variant, str(e)))

    aggregate(out_root)
    if failures:
        error(f"{len(failures)}/{len(grid)} cells failed: {failures}")
        raise SystemExit(1)
    ok(f"all {len(rows)} cells complete → {out_root}")


if __name__ == "__main__":
    main()
