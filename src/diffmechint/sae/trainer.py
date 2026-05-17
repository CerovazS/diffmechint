"""Train a SAELens TrainingSAE; coordinate warm-start across DiT checkpoints."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Sequence

import torch
from sae_lens import SAETrainer, TrainingSAE
from sae_lens.config import LoggingConfig, SAETrainerConfig
from safetensors.torch import load_file

from diffmechint.utils import info, ok, warn


def _is_rank0() -> bool:
    """True on rank 0 of a distributed run; True everywhere if not distributed.

    SAE training is single-GPU today, but we enforce the rank-0-only-writes
    rule per swe-stack.md so the same code keeps a single JSONL even if we
    later move to DDP.
    """
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:  # noqa: BLE001
        pass
    return True


def _coerce_for_json(v):
    """Best-effort scalar coercion: tensors → float, numpy → float, others → skip.

    `wandb.log` accepts histograms/images/plots; we only mirror scalars to the
    JSONL since they're the plot-ready signal a downstream pandas reader wants.
    """
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, torch.Tensor):
        try:
            return float(v.detach().cpu().item())
        except Exception:  # noqa: BLE001
            return None
    if hasattr(v, "item") and callable(v.item):
        try:
            return float(v.item())
        except Exception:  # noqa: BLE001
            return None
    return None  # drop histograms, dicts, etc.


class _WandbJsonlMirror:
    """Monkey-patch `wandb.log` so every call also appends one line to a JSONL.

    Reverts on exit. Only rank-0 actually writes (no-op on other ranks).
    SAELens 6.x logs only via `wandb.log`, so this captures everything the
    trainer emits without needing a SAELens-internal hook.
    """

    def __init__(self, jsonl_path: Path):
        self.path = Path(jsonl_path)
        self._original = None
        self._is_rank0 = _is_rank0()

    def __enter__(self):
        import wandb
        self._original = wandb.log
        original = self._original
        path = self.path
        is_rank0 = self._is_rank0
        if is_rank0:
            path.parent.mkdir(parents=True, exist_ok=True)

        def wrapped_log(data, step=None, commit=None, sync=None):  # noqa: ARG001
            try:
                original(data, step=step, commit=commit, sync=sync)
            except TypeError:
                # Older wandb signatures missed `sync`; retry without it.
                original(data, step=step, commit=commit)
            if not is_rank0:
                return
            rec = {"step": step}
            for k, v in (data or {}).items():
                coerced = _coerce_for_json(v)
                if coerced is not None:
                    rec[k] = coerced
            try:
                with path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:  # noqa: BLE001
                warn(f"_WandbJsonlMirror: failed to append {path}: {e}")

        wandb.log = wrapped_log
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        if self._original is not None:
            import wandb
            wandb.log = self._original


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
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
    save_final_checkpoint: bool = True,
) -> Path:
    """Run SAELens `SAETrainer.fit` end-to-end and return the output dir.

    History: an earlier "Fix B" variant (transfer Adam optimizer state across
    DiT-checkpoint warm-starts) was implemented to mimic Xu et al. 2412.17626,
    but a careful audit of their code (`Superposition09m/SAE-Track`) showed
    they transfer ONLY SAE weights — no optimizer state, no normalization stats.
    Empirically Fix B mitigated but did not fix our dead-feature explosion
    (94% → 91%). Cold restart per stage (warm_mode='cold' in warm_started_sweep)
    drops dead to 0.16%, so Fix B was removed as a dead path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # LoggingConfig accepts only a subset of fields across SAELens versions;
    # build it with whichever ones are supported in the installed wheel.
    logger_kwargs: dict = {"log_to_wandb": bool(log_to_wandb), "wandb_project": wandb_project}
    for key, val in (
        ("wandb_entity", wandb_entity),
        ("wandb_group", wandb_group),
        ("run_name", wandb_run_name),
    ):
        if val is not None and key in LoggingConfig.__dataclass_fields__:
            logger_kwargs[key] = val
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
        logger=LoggingConfig(**logger_kwargs),
    )
    trainer = SAETrainer(cfg=cfg, sae=sae, data_provider=data_provider)
    info(
        f"SAETrainer: total_samples={total_training_samples} "
        f"batch={train_batch_size_samples} lr={lr} dev={device} "
        f"warm_up={lr_warm_up_steps}"
    )
    # SAELens 6.x calls wandb.log() inside its trainer but does NOT call
    # wandb.init() itself — responsibility falls to us. Wrap fit() with
    # init/finish so each train_sae call is its own self-contained run.
    if log_to_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            name=wandb_run_name,
            reinit=True,
            config={
                "d_sae": getattr(sae.cfg, "d_sae", None),
                "d_in": getattr(sae.cfg, "d_in", None),
                "k": getattr(sae.cfg, "k", None),
                "total_training_samples": int(total_training_samples),
                "batch_size": int(train_batch_size_samples),
                "lr": float(lr),
            },
        )
    # Local JSONL mirror (rank-0 only) per swe-stack.md "Training Logging
    # Standard": loss/metric history must live on disk in a plot-friendly
    # format even if wandb is unavailable. Path is co-located with the
    # safetensors checkpoint.
    metrics_path = out_dir / "metrics" / "train.jsonl"
    try:
        with _WandbJsonlMirror(metrics_path):
            trainer.fit()
    finally:
        if log_to_wandb:
            import wandb
            wandb.finish()
    ok(f"SAE training done → {out_dir}")
    if _is_rank0() and metrics_path.exists():
        info(f"  local metrics → {metrics_path}")
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
    lr_warm_up_steps: int = 200,
    warm_lr_warm_up_steps: int = 0,
    device: str = "cuda",
    provider_factory=None,
    log_to_wandb: bool = False,
    wandb_project: str = "diffmechint-sae",
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    warm_mode: str = "cold",
) -> list[Path]:
    """Train one SAE per (DiT-checkpoint) pair, chaining stages by `warm_mode`.

    Two modes after the Xu et al. 2412.17626 audit (see trainer docstring):
      - `cold` (default, canonical): every stage is a fresh cold-start with
        full `base_total_samples` budget. Drops dead-feature rate from 94% to
        0.16% in our setup. This is the production choice.
      - `weights_only`: load SAE weights from previous stage, reset optimizer
        and lr scheduler. Mimics Xu et al.'s actual implementation. Buggy in
        our setup (94% dead, EV plateaus at 0.83) because our DiT-ckpt gaps
        are larger than Xu's LLM-ckpt gaps and our cold budget is 15× smaller.
        Kept ONLY to reproduce the failure mode as a diagnostic baseline.

    The earlier `weights_and_optimizer` mode (Fix B — carry Adam state) was
    removed: Xu et al. don't do this, and empirically it only marginally
    improved dead-feature rate (94% → 91%), not a real fix.

    Args:
      sae_factory: `() -> TrainingSAE` — fresh SAE constructor.
      activation_shards_per_dit: sequence of `(label, shard_paths)`. Order
        determines stage order; only matters in `weights_only` mode.
      out_root: parent dir; each stage lands in `out_root/<label>/`.
      base_total_samples: budget for stage 0 and every stage when warm_mode='cold'.
      warm_total_samples: budget for stage 1+ when warm_mode='weights_only'.
      lr_warm_up_steps: LR warmup at stage 0 and every `cold` stage.
      warm_lr_warm_up_steps: LR warmup at `weights_only` warm stages.
      warm_mode: 'cold' (default) | 'weights_only'.

    Returns the list of final checkpoint directories in order.
    """
    if warm_mode not in {"cold", "weights_only"}:
        raise ValueError(
            f"warm_mode must be 'cold' | 'weights_only', got {warm_mode!r}. "
            "The previous 'weights_and_optimizer' mode (Fix B) was removed."
        )
    if provider_factory is None:
        from .data_provider import hdf5_provider as provider_factory  # noqa: F811
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    info(f"warm_started_sweep: mode={warm_mode} stages={len(activation_shards_per_dit)} "
         f"cold_budget={base_total_samples} warm_budget={warm_total_samples}")

    prev_final: Path | None = None
    finals: list[Path] = []
    for i, (label, shards) in enumerate(activation_shards_per_dit):
        info(f"--- SAE stage {i + 1}/{len(activation_shards_per_dit)}: {label} "
             f"(mode={warm_mode}) ---")
        sae = sae_factory()
        # Warm-start weights only in `weights_only` mode and only past stage 0.
        if i > 0 and warm_mode == "weights_only" and prev_final is not None:
            sae = warm_start_from(sae, prev_final / "sae_weights.safetensors")
        provider = provider_factory(shards, batch_size=batch_size, device=device)
        # Budget per stage: full cold budget on stage 0 and every stage in
        # `cold` mode; warm budget on warm-started stages otherwise.
        total = base_total_samples if (warm_mode == "cold" or i == 0) else warm_total_samples
        # LR warmup per stage: full warmup on cold-started stages, optional
        # smaller warmup on warm-started stages (we don't ramp from 0 on
        # weights that are already pre-tuned).
        stage_lr_warmup = (
            warm_lr_warm_up_steps
            if (warm_mode == "weights_only" and i > 0)
            else lr_warm_up_steps
        )
        out_dir = train_sae(
            sae,
            provider,
            out_dir=out_root / label,
            total_training_samples=total,
            train_batch_size_samples=batch_size,
            lr=lr,
            lr_warm_up_steps=stage_lr_warmup,
            device=device,
            log_to_wandb=log_to_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            wandb_run_name=f"{wandb_group}_{label}" if wandb_group else label,
        )
        # SAELens writes a `final/` subdirectory per the SAETrainer contract.
        prev_final = next((d for d in out_dir.iterdir() if d.is_dir() and d.name == "final"), None)
        finals.append(out_dir)
    return finals
