"""Train a SAELens TrainingSAE; coordinate warm-start across DiT checkpoints."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Sequence

import torch
from sae_lens import SAETrainer, TrainingSAE
from sae_lens.config import LoggingConfig, SAETrainerConfig
from safetensors.torch import load_file

from diffmechint.utils import info, ok, warn


def train_sae(
    sae: TrainingSAE,
    data_provider: Iterator[torch.Tensor],
    *,
    out_dir: Path | str,
    total_training_samples: int,
    train_batch_size_samples: int = 4096,
    lr: float = 3e-4,
    lr_end: float | None = None,
    lr_scheduler_name: str = "constant",
    lr_warm_up_steps: int = 200,
    n_checkpoints: int = 1,
    device: str = "cuda",
    autocast: bool = False,
    log_to_wandb: bool = False,
    wandb_project: str = "diffmechint-sae",
    save_final_checkpoint: bool = True,
) -> Path:
    """Run SAELens `SAETrainer.fit` end-to-end and return the final checkpoint dir.

    Returns the directory under `out_dir` where the final + intermediate
    safetensors checkpoints are written (`<out_dir>/final/` is canonical).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SAETrainerConfig(
        total_training_samples=int(total_training_samples),
        train_batch_size_samples=int(train_batch_size_samples),
        lr=float(lr),
        lr_end=float(lr_end) if lr_end is not None else float(lr),
        lr_scheduler_name=lr_scheduler_name,
        lr_warm_up_steps=int(lr_warm_up_steps),
        n_checkpoints=int(n_checkpoints),
        checkpoint_path=str(out_dir),
        save_final_checkpoint=bool(save_final_checkpoint),
        device=str(device),
        autocast=bool(autocast),
        logger=LoggingConfig(log_to_wandb=bool(log_to_wandb), wandb_project=wandb_project),
    )
    trainer = SAETrainer(cfg=cfg, sae=sae, data_provider=data_provider)
    info(
        f"SAETrainer: total_samples={total_training_samples} "
        f"batch={train_batch_size_samples} lr={lr} dev={device}"
    )
    trainer.fit()
    ok(f"SAE training done → {out_dir}")
    return out_dir


def warm_start_from(sae: TrainingSAE, prev_safetensors_path: Path | str) -> TrainingSAE:
    """Initialize `sae` weights from a previously-trained checkpoint.

    Used by the orchestrator across DiT fractional checkpoints (Xu et al.
    2412.17626): re-using encoder + decoder + bias from checkpoint i means
    SAE training on checkpoint i+1 converges in ~1/3 the steps.
    """
    prev = Path(prev_safetensors_path)
    if not prev.is_file():
        warn(f"warm_start_from: {prev} not found, skipping.")
        return sae
    state = load_file(str(prev))
    missing, unexpected = sae.load_state_dict(state, strict=False)
    info(
        f"warm-start from {prev.name}: "
        f"loaded {len(state) - len(unexpected)} keys, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    return sae


def warm_started_sweep(
    sae_factory,
    activation_shards_per_dit: Sequence[tuple[str, list[Path] | Path | str]],
    *,
    out_root: Path | str,
    base_total_samples: int,
    warm_total_samples: int,
    batch_size: int = 4096,
    lr: float = 3e-4,
    device: str = "cuda",
    provider_factory=None,
) -> list[Path]:
    """Train one SAE per (DiT-checkpoint) pair, warm-starting each from the prior.

    Args:
      sae_factory: `() -> TrainingSAE` — fresh SAE constructor.
      activation_shards_per_dit: sequence of `(label, shard_paths)`. Order
        determines warm-start chain.
      out_root: parent dir; each step lands in `out_root/<label>/`.
      base_total_samples: training samples for the FIRST step (cold-start).
      warm_total_samples: training samples for warm-started subsequent steps.
      provider_factory: callable `(shard_paths) -> Iterator[Tensor]`. Defaults
        to `diffmechint.sae.data_provider.hdf5_provider`.

    Returns the list of final checkpoint directories in order.
    """
    if provider_factory is None:
        from .data_provider import hdf5_provider as provider_factory  # noqa: F811
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    prev_final: Path | None = None
    finals: list[Path] = []
    for i, (label, shards) in enumerate(activation_shards_per_dit):
        info(f"--- warm-started SAE step {i + 1}/{len(activation_shards_per_dit)}: {label} ---")
        sae = sae_factory()
        if prev_final is not None:
            sae = warm_start_from(sae, prev_final / "sae_weights.safetensors")
        provider = provider_factory(shards, batch_size=batch_size, device=device)
        total = warm_total_samples if i > 0 else base_total_samples
        out_dir = train_sae(
            sae,
            provider,
            out_dir=out_root / label,
            total_training_samples=total,
            train_batch_size_samples=batch_size,
            lr=lr,
            device=device,
        )
        # SAELens writes a `final/` subdirectory per the SAETrainer contract.
        prev_final = next((d for d in out_dir.iterdir() if d.is_dir() and d.name == "final"), None)
        finals.append(out_dir)
    return finals
