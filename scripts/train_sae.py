"""Phase 4 driver: train SAELens TopK SAEs per (condition × layer × t_bin) chain.

For each of the 27 chains (3 conditions × 3 layers × 3 t_bins), iterate the 7
fractional DiT EMA checkpoints in order, training one SAE per checkpoint and
warm-starting from the previous one (Xu et al. 2412.17626).

Each chain is one W&B run; per-checkpoint sub-stages log under the same run
via SAELens' built-in `n_checkpoints` schedule.

Output layout (per chain):

    $FAST/diffmechint/sae/<condition>/L<layer>_T<tbin>/step_<NNNNNN>/
        ├── final/sae_weights.safetensors   ← canonical end-of-stage weights
        ├── final/cfg.json
        └── trainer_state.json

Usage:
    uv run python scripts/train_sae.py \
        --conditions sd_vae repa_e eq_vae \
        --layers 3 6 9 --t_bins 0 1 2 \
        --base_total_samples 20000000 --warm_total_samples 5000000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make sure wandb defaults to the user's entity/project even if env is unset.
os.environ.setdefault("WANDB_ENTITY", "ar_spectra")
os.environ.setdefault("WANDB_PROJECT", "diffmechint")

import torch  # noqa: E402

from diffmechint.sae import build_sae, hdf5_provider  # noqa: E402
from diffmechint.sae.trainer import warm_started_sweep  # noqa: E402
from diffmechint.utils import error, info, ok, warn  # noqa: E402

ACTIVATIONS_BASE = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations")
SAE_BASE = Path("/leonardo_scratch/fast/IscrC_YENDRI/lcerovaz/diffmechint/sae")
# Default DiT fractional steps (set by FractionalCheckpoint callback in training).
DEFAULT_DIT_STEPS = [4000, 10000, 20000, 50000, 100000, 150000, 200000]


def _cell_shards(condition: str, dit_step: int, layer: int, t_bin: int) -> list[Path]:
    """Return the single Phase 3 HDF5 shard for one (cond × ckpt × layer × t_bin) cell."""
    p = ACTIVATIONS_BASE / condition / f"step_{dit_step:06d}" / f"{layer}_{t_bin}.h5"
    if not p.is_file():
        raise FileNotFoundError(f"cell shard not found: {p}")
    return [p]


def train_one_chain(
    *,
    condition: str,
    layer: int,
    t_bin: int,
    dit_steps: list[int],
    d_in: int,
    d_sae: int,
    k: int,
    variant: str,
    base_total_samples: int,
    warm_total_samples: int,
    batch_size: int,
    lr: float,
    lr_warm_up_steps: int,
    warm_lr_warm_up_steps: int,
    warm_mode: str,
    device: str,
    out_root: Path,
    wandb_project: str,
    variant_tag: str = "",
) -> list[Path]:
    """Train the 7-ckpt warm-started SAE chain for a single (cond, layer, t_bin) cell."""
    out_dir = out_root / condition / f"L{layer}_T{t_bin}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the (label, shards) sequence in DiT-checkpoint order.
    chain: list[tuple[str, list[Path]]] = [
        (f"step_{s:06d}", _cell_shards(condition, s, layer, t_bin)) for s in dit_steps
    ]

    def sae_factory():
        return build_sae(
            d_in=d_in,
            d_sae=d_sae,
            k=k,
            variant=variant,
            device=device,
            metadata={
                "condition": condition,
                "layer": layer,
                "t_bin": t_bin,
                "d_in": d_in,
                "d_sae": d_sae,
                "k": k,
                "variant": variant,
            },
        )

    def provider_factory(shards, batch_size, device):
        return hdf5_provider(
            shards, batch_size=batch_size, device=device,
            flatten_tokens=True, loop_forever=True, shuffle=True,
        )

    info(f"=== chain {condition} / L{layer} / T{t_bin} (tag={variant_tag or '-'}) ===")
    group = f"{condition}_L{layer}_T{t_bin}" + (f"_{variant_tag}" if variant_tag else "")
    finals = warm_started_sweep(
        sae_factory=sae_factory,
        activation_shards_per_dit=chain,
        out_root=out_dir,
        base_total_samples=base_total_samples,
        warm_total_samples=warm_total_samples,
        batch_size=batch_size,
        lr=lr,
        lr_warm_up_steps=lr_warm_up_steps,
        warm_lr_warm_up_steps=warm_lr_warm_up_steps,
        warm_mode=warm_mode,
        device=device,
        provider_factory=provider_factory,
        log_to_wandb=True,
        wandb_project=wandb_project,
        wandb_entity=os.environ.get("WANDB_ENTITY"),
        wandb_group=group,
    )
    return finals


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=["sd_vae", "repa_e", "eq_vae"])
    p.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9])
    p.add_argument("--t_bins", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--dit_steps", type=int, nargs="+", default=DEFAULT_DIT_STEPS)
    p.add_argument("--d_in", type=int, default=768,
                   help="SiT-B/2 hidden size; override for other backbones.")
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--variant", type=str, default="topk",
                   choices=["topk", "batch_topk", "matryoshka"])
    p.add_argument("--base_total_samples", type=int, default=20_000_000,
                   help="Tokens (post-flatten) seen on first/cold-start ckpt.")
    p.add_argument("--warm_total_samples", type=int, default=5_000_000,
                   help="Tokens per warm-started ckpt (≥2nd in chain).")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out_root", type=Path, default=SAE_BASE)
    p.add_argument("--wandb_project", type=str,
                   default=os.environ.get("WANDB_PROJECT", "diffmechint"))
    p.add_argument("--only", type=str, default=None,
                   help="Restrict to one chain, format '<cond>/L<layer>/T<tbin>' (smoke).")
    p.add_argument("--variant_tag", type=str, default="",
                   help="Suffix added to wandb group and run names to disambiguate "
                        "ablation runs on the same cell (e.g. 'd7680', 'k64', 'long').")
    p.add_argument("--lr_warm_up_steps", type=int, default=200,
                   help="LR warmup steps at the FIRST stage (always cold) and at every "
                        "stage when warm_mode=cold.")
    p.add_argument("--warm_lr_warm_up_steps", type=int, default=0,
                   help="LR warmup steps at warm-started stages (default 0 — don't "
                        "ramp LR from 0 on weights that are already pre-tuned).")
    p.add_argument("--warm_mode", type=str, default="cold",
                   choices=["cold", "weights_only"],
                   help="Strategy across the 7 DiT-ckpt stages: "
                        "cold = each stage cold-start with full budget (default, canonical); "
                        "weights_only = load weights only (diagnostic baseline — replicates "
                        "Xu et al. 2412.17626 and our 94%% dead-feature failure mode).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warn("CUDA not available — SAE training will be slow.")

    args.out_root.mkdir(parents=True, exist_ok=True)
    info(f"out_root = {args.out_root}")
    info(f"wandb entity/project = {os.environ.get('WANDB_ENTITY', '?')}/{args.wandb_project}")

    # Build the (cond, layer, t_bin) chain list.
    chains = []
    if args.only is not None:
        try:
            cond, lay, tb = args.only.split("/")
            chains = [(cond, int(lay.removeprefix("L")), int(tb.removeprefix("T")))]
        except Exception as e:  # noqa: BLE001
            error(f"--only must look like 'sd_vae/L6/T1', got {args.only!r}: {e}")
            return 1
    else:
        for c in args.conditions:
            for lay in args.layers:
                for tb in args.t_bins:
                    chains.append((c, lay, tb))
    info(f"{len(chains)} chains × {len(args.dit_steps)} DiT ckpts = "
         f"{len(chains) * len(args.dit_steps)} SAE trainings")

    t0 = time.perf_counter()
    for i, (cond, lay, tb) in enumerate(chains, start=1):
        info(f"[{i}/{len(chains)}] chain {cond} L{lay} T{tb}")
        try:
            train_one_chain(
                condition=cond, layer=lay, t_bin=tb,
                dit_steps=args.dit_steps,
                d_in=args.d_in, d_sae=args.d_sae, k=args.k,
                variant=args.variant,
                base_total_samples=args.base_total_samples,
                warm_total_samples=args.warm_total_samples,
                batch_size=args.batch_size, lr=args.lr,
                lr_warm_up_steps=args.lr_warm_up_steps,
                warm_lr_warm_up_steps=args.warm_lr_warm_up_steps,
                warm_mode=args.warm_mode,
                device=device,
                out_root=args.out_root,
                wandb_project=args.wandb_project,
                variant_tag=args.variant_tag,
            )
        except Exception as e:  # noqa: BLE001
            error(f"chain {cond}/L{lay}/T{tb} failed: {type(e).__name__}: {e}")
            if args.only:
                raise
            continue
    ok(f"Done. {len(chains)} chains in {(time.perf_counter() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
