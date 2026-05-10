"""SampleCallback — periodic class-conditional ODE sampling + VAE decode + grid PNG.

Saves a 4×4 (or n_samples-shaped) RGB image grid every `every_n_steps` steps to
the run's output dir. Uses the EMA shadow weights when available, since the EMA
is the canonical analysis target for SAE / probes / circuits downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from diffmechint.sit.transport import Sampler
from diffmechint.tokenizers import build
from diffmechint.utils import info, ok, warn


class SampleCallback(Callback):
    """Class-conditional ODE sampling on a fixed prompt set, every N steps.

    Args:
      every_n_steps: cadence of sampling (e.g. 5000).
      n_samples: number of generations per evaluation (must be divisible by 2 for CFG).
      cfg_scale: classifier-free guidance scale; 1.0 = no CFG, 4.0 = paper default.
      sample_steps: ODE solver steps (dopri5 adaptive: a hint, not strict).
      sample_method: dopri5 (adaptive) | euler | heun (fixed-step).
      out_subdir: relative dir under run output to write PNGs into.
      adapter_name: tokenizer name for VAE decode. Must match the training condition.
      latent_shape: (C, H, W) of the latent. Inferred from adapter spec at runtime.
      num_classes: ImageNet-1k → 1000.
      stats_path: path to stats.json from precompute; required to denormalize the
                  sampled latent before VAE decode.
      seed: deterministic class-id selection across runs.
    """

    def __init__(
        self,
        every_n_steps: int = 5000,
        n_samples: int = 16,
        cfg_scales: tuple[float, ...] | list[float] | float = (1.0, 4.0),
        sample_steps: int = 50,
        sample_method: str = "dopri5",
        out_subdir: str = "samples",
        adapter_name: str = "sd_vae",
        num_classes: int = 1000,
        stats_path: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if n_samples % 2 != 0:
            raise ValueError(f"n_samples must be even (CFG doubles the batch), got {n_samples}")
        # Accept a single float for backwards compatibility with older configs.
        if isinstance(cfg_scales, (int, float)):
            cfg_scales = (float(cfg_scales),)
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self.cfg_scales = tuple(float(c) for c in cfg_scales)
        self.sample_steps = sample_steps
        self.sample_method = sample_method
        self.out_subdir = out_subdir
        self.adapter_name = adapter_name
        self.num_classes = num_classes
        self.stats_path = Path(stats_path) if stats_path else None
        self.seed = seed

        self._adapter = None  # lazy
        self._mean: Tensor | None = None
        self._inv_std: Tensor | None = None
        self._stats_loaded = False
        # Pre-pick deterministic class ids — one diverse selection per run, fixed
        # across all evaluations so we can compare visual progression cleanly.
        gen = torch.Generator().manual_seed(seed)
        self._class_ids = torch.randint(0, num_classes, (n_samples,), generator=gen)

    def _ensure_stats(self) -> None:
        if self._stats_loaded or self.stats_path is None:
            return
        if not self.stats_path.exists():
            warn(f"SampleCallback: stats.json not at {self.stats_path}; skipping denormalize")
            self._stats_loaded = True
            return
        s = json.loads(self.stats_path.read_text())
        mean = np.asarray(s["per_feature_mean"], dtype=np.float32)
        std = np.asarray(s["per_feature_std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        # Reshape for broadcast on (B, C, H, W).
        feat_axis = s["feature_axis"]  # batched-tensor axis (1 for spatial)
        n_dims = len(s["latent_shape"])
        broadcast = [1] * (n_dims + 1)  # +1 for batch
        broadcast[feat_axis] = -1
        self._mean = torch.from_numpy(mean).view(*broadcast)
        self._inv_std = torch.from_numpy(1.0 / std).view(*broadcast)
        self._stats_loaded = True

    def _ensure_adapter(self, device: torch.device) -> None:
        if self._adapter is None:
            info(f"SampleCallback: loading VAE adapter '{self.adapter_name}'")
            adapter = build(self.adapter_name)
            adapter.load()
            adapter.to(device)
            self._adapter = adapter
        elif next(self._adapter.parameters(), torch.tensor([])).device != device:
            self._adapter.to(device)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if trainer.global_rank != 0:
            return
        # +1 so that step==N triggers at end of step N (consistent with the
        # ckpt callback's "first N steps complete" semantics).
        step = trainer.global_step + 1
        if step % self.every_n_steps != 0:
            return
        try:
            self._sample(trainer, pl_module, step)
        except Exception as e:  # noqa: BLE001
            warn(f"SampleCallback @ step {step}: {type(e).__name__}: {e}")

    @torch.no_grad()
    def _sample(self, trainer: L.Trainer, pl_module: L.LightningModule, step: int) -> None:
        device = pl_module.device
        self._ensure_stats()
        self._ensure_adapter(device)

        # Pick the model: prefer EMA shadow (analysis target), fallback to live.
        if getattr(pl_module, "ema", None) is not None:
            model = pl_module.ema.shadow
        else:
            model = pl_module.model
        model.eval()

        # Latent shape from the model's instantiation parameters.
        in_channels = pl_module.hparams.in_channels
        input_size = pl_module.hparams.input_size
        latent_shape = (in_channels, input_size, input_size)

        B = self.n_samples
        # Initial standard-normal noise in z-scored latent space.
        noise = torch.randn(B, *latent_shape, device=device)

        # CFG: double the batch with null class (= num_classes per SiT convention).
        labels = self._class_ids.to(device)
        null_label = torch.full_like(labels, self.num_classes)
        noise_full = torch.cat([noise, noise], dim=0)
        labels_full = torch.cat([labels, null_label], dim=0)

        sampler = Sampler(pl_module.transport)
        sample_fn = sampler.sample_ode(
            sampling_method=self.sample_method,
            num_steps=self.sample_steps,
        )
        n_ch = latent_shape[0]

        # Save grid PNG.
        from torchvision.utils import save_image  # local import keeps callback light
        nrow = max(1, int(B ** 0.5))
        out_dir = Path(trainer.default_root_dir) / self.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        for cfg_scale in self.cfg_scales:
            def model_fn(x, t, y, cfg=cfg_scale):  # bind cfg into closure
                out = model.forward_with_cfg(x, t, y, cfg)
                return out[:, :n_ch]

            samples = sample_fn(noise_full, model_fn, y=labels_full)[-1]  # (2B, C, H, W)
            samples = samples[:B]  # conditional half only

            # Denormalize z-scored latent → raw VAE-space latent.
            if self._mean is not None and self._inv_std is not None:
                samples = samples / self._inv_std.to(device) + self._mean.to(device)

            # Decode through frozen VAE.
            images = self._adapter.decode(samples).clamp_(-1, 1)  # [-1, 1]
            images = images.add(1).div(2).clamp_(0, 1)            # → [0, 1]

            cfg_tag = f"cfg{cfg_scale:.1f}".replace(".", "p")
            out_path = out_dir / f"step_{step:08d}_{cfg_tag}.png"
            save_image(images, out_path, nrow=nrow)
            ok(f"SampleCallback @ step {step} cfg={cfg_scale}: {B} samples → {out_path}")
