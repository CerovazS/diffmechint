"""Hydra entry-point for SiT training (FM-OT).

Single-GPU smoke (synthetic latents, SD-VAE shape):
    uv run python -m diffmechint.training.train \\
        +trainer.max_steps=1000 \\
        +data.batch_size=32 \\
        ckpt_dir=outputs/smoke_sd_vae

Real run (cached latents):
    uv run python -m diffmechint.training.train \\
        tokenizer=eq_vae model=sit_b_2 transport=fm_ot trainer=local_3090 \\
        +data._target_=diffmechint.training.data.CachedLatentDataModule \\
        +data.shard_dir=$FAST/diffmechint/latents/eq_vae
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from diffmechint.training.checkpointing import FractionalCheckpoint
from diffmechint.training.data import SyntheticLatentDataModule
from diffmechint.training.sit_module import SiTLightningModule
from diffmechint.utils import info, ok


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    L.seed_everything(int(cfg.get("seed", 42)), workers=True)
    info(f"Seed: {cfg.get('seed', 42)}")

    # Tokenizer adapter — used here only for its spec (latent_shape, in_channels).
    # Weights are not loaded for SiT training; latents come from the datamodule.
    adapter = hydra.utils.instantiate(cfg.tokenizer)
    spec = adapter.spec
    info(f"Tokenizer: {adapter} latent_shape={spec.latent_shape}")

    # Model.
    ema_resume = cfg.get("ema_resume_from")
    sit_module = hydra.utils.instantiate(
        cfg.model,
        input_size=spec.latent_shape[1],
        in_channels=spec.in_channels,
        transport_cfg=OmegaConf.to_container(cfg.transport, resolve=True),
        ema_resume_path=ema_resume,
    )
    info(f"SiT params: {sit_module.n_parameters() / 1e6:.1f}M")

    # Optional: load live model weights from a prior run's safetensors. Note
    # that we do NOT recover optimizer/scheduler/RNG state — this is
    # "weights resume", not full Lightning resume. Pair with model.warmup_steps=0
    # so the LR schedule doesn't reset to zero on top of an already-trained model.
    resume_from = cfg.get("resume_from")
    if resume_from:
        from safetensors.torch import load_file
        info(f"Resuming live model weights from {resume_from}")
        live_sd = load_file(resume_from)
        missing, unexpected = sit_module.model.load_state_dict(live_sd, strict=False)
        if missing or unexpected:
            info(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    # Data — default to synthetic if no datamodule explicit.
    if "data" in cfg and "_target_" in cfg.data:
        datamodule = hydra.utils.instantiate(cfg.data)
    else:
        data_cfg = cfg.get("data", {})
        datamodule = SyntheticLatentDataModule(
            latent_shape=spec.latent_shape,
            num_classes=cfg.model.get("num_classes", 1000),
            n_samples=int(data_cfg.get("n_samples", 8192)),
            batch_size=int(data_cfg.get("batch_size", 32)),
            num_workers=int(data_cfg.get("num_workers", 2)),
            seed=int(cfg.get("seed", 42)),
        )
        info(f"Using SyntheticLatentDataModule (latent_shape={spec.latent_shape})")

    # Trainer + fractional checkpoint callback.
    trainer_cfg = OmegaConf.to_container(cfg.get("trainer", {}), resolve=True) or {}
    max_steps = int(trainer_cfg.get("max_steps", 1000))
    ckpt_dir = Path(cfg.get("ckpt_dir", "outputs/run/checkpoints"))
    ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else Path.cwd() / ckpt_dir
    callbacks: list = [FractionalCheckpoint(out_dir=str(ckpt_dir), max_steps=max_steps)]
    cb_cfg = cfg.get("callbacks") or {}
    for name in ("sample", "fid"):
        sub = cb_cfg.get(name)
        if sub is not None:
            cb = hydra.utils.instantiate(sub)
            callbacks.append(cb)
            info(f"Callback enabled: {name} ({type(cb).__name__})")

    trainer_kwargs: dict = {
        "max_steps": max_steps,
        "max_epochs": -1,
        "log_every_n_steps": 50,
        "callbacks": callbacks,
        "enable_checkpointing": False,  # the fractional callback owns saving
        "default_root_dir": str(ckpt_dir.parent),  # SampleCallback writes under here
    }
    if torch.cuda.is_available():
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = trainer_cfg.get("devices", 1)
        trainer_kwargs["precision"] = trainer_cfg.get("precision", "bf16-mixed")
    else:
        trainer_kwargs["accelerator"] = "cpu"

    # Allow a few specific overrides from the trainer config.
    for k in (
        "gradient_clip_val",
        "accumulate_grad_batches",
        "val_check_interval",
        "limit_val_batches",
        "check_val_every_n_epoch",
    ):
        if k in trainer_cfg:
            trainer_kwargs[k] = trainer_cfg[k]

    trainer = L.Trainer(**trainer_kwargs)
    info(
        f"Trainer: max_steps={max_steps} accel={trainer_kwargs['accelerator']} "
        f"devices={trainer_kwargs.get('devices', 1)} "
        f"precision={trainer_kwargs.get('precision', 'fp32')}"
    )

    trainer.fit(sit_module, datamodule=datamodule)
    ok(f"Training done. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
