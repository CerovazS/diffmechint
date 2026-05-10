"""MiniFIDCallback — periodic FID-vs-ImageNet-val evaluation on N generated images.

Uses Clean-FID (Parmar et al. 2022) which removes resize artifacts that affect
the classical FID-50k score. Caches the reference dataset statistics on first
use under the user's ~/.cache/cleanfid; subsequent calls only score the
generated set.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from diffmechint.sit.transport import Sampler
from diffmechint.tokenizers import build
from diffmechint.utils import error, info, ok, warn


class MiniFIDCallback(Callback):
    """Generate N images, compute Clean-FID against ImageNet val, log scalar.

    Args:
      every_n_steps: cadence (e.g. 25000).
      n_samples: total generations per evaluation (5000 → reasonable mini-FID).
      sample_batch_size: per-step batch (capped by GPU mem; CFG doubles it).
      cfg_scale: classifier-free guidance scale.
      sample_steps: ODE solver steps.
      sample_method: dopri5 | euler | heun.
      reference_dir: real-image reference for FID (e.g. imagenet/val/).
      reference_name: stat-cache key under ~/.cache/cleanfid.
      adapter_name: tokenizer name for VAE decode.
      num_classes: 1000 for ImageNet-1k.
      stats_path: stats.json from precompute (for latent denormalize).
      seed: deterministic class-id sampling.
    """

    def __init__(
        self,
        every_n_steps: int = 25_000,
        # 5000 samples for parity with post_hoc_fid.py — both go into the
        # same metrics/validation/fid.csv across runs, so the live and
        # post-hoc curves must use identical n_samples to be comparable.
        # Per call ~5-8 min on 1× A100; pair with NCCL_TIMEOUT_MS=3600000
        # in the sbatch wrapper so the rank > 0 barrier doesn't time out.
        n_samples: int = 5000,
        sample_batch_size: int = 32,
        cfg_scale: float = 4.0,
        sample_steps: int = 50,
        sample_method: str = "dopri5",
        reference_dir: str = "/leonardo_scratch/fast/IscrC_YENDRI/imagenet/val",
        reference_name: str = "imagenet_val_50k",
        adapter_name: str = "sd_vae",
        num_classes: int = 1000,
        stats_path: str | None = None,
        seed: int = 0,
        require_cache_at_init: bool = True,         # fail fast if stats missing
    ) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self.sample_batch_size = sample_batch_size
        self.cfg_scale = cfg_scale
        self.sample_steps = sample_steps
        self.sample_method = sample_method
        self.reference_dir = Path(reference_dir)
        self.reference_name = reference_name
        self.adapter_name = adapter_name
        self.num_classes = num_classes
        self.stats_path = Path(stats_path) if stats_path else None
        self.seed = seed
        self.require_cache_at_init = require_cache_at_init

        self._adapter = None
        self._mean: Tensor | None = None
        self._inv_std: Tensor | None = None
        self._stats_loaded = False
        self._ref_stats_built = False

    def _ensure_stats(self) -> None:
        if self._stats_loaded or self.stats_path is None:
            return
        if not self.stats_path.exists():
            warn(f"MiniFIDCallback: stats.json not at {self.stats_path}; latents NOT denormalized")
            self._stats_loaded = True
            return
        s = json.loads(self.stats_path.read_text())
        mean = np.asarray(s["per_feature_mean"], dtype=np.float32)
        std = np.asarray(s["per_feature_std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        feat_axis = s["feature_axis"]
        n_dims = len(s["latent_shape"])
        broadcast = [1] * (n_dims + 1)
        broadcast[feat_axis] = -1
        self._mean = torch.from_numpy(mean).view(*broadcast)
        self._inv_std = torch.from_numpy(1.0 / std).view(*broadcast)
        self._stats_loaded = True

    def _ensure_adapter(self, device: torch.device) -> None:
        if self._adapter is None:
            adapter = build(self.adapter_name)
            adapter.load()
            adapter.to(device)
            self._adapter = adapter
        elif next(self._adapter.parameters(), torch.tensor([])).device != device:
            self._adapter.to(device)

    def _ensure_reference_stats(self) -> None:
        """Build Inception statistics on the reference dir once (cached on disk)."""
        if self._ref_stats_built:
            return
        from cleanfid import fid

        try:
            if fid.test_stats_exists(self.reference_name, mode="clean"):
                self._ref_stats_built = True
                return
        except Exception:
            pass

        if not self.reference_dir.exists():
            error(f"MiniFIDCallback: reference dir missing: {self.reference_dir}")
            raise FileNotFoundError(self.reference_dir)
        info(
            f"MiniFIDCallback: building Clean-FID reference stats '{self.reference_name}' "
            f"from {self.reference_dir} (one-time, ~5 min on 50k images)…"
        )
        fid.make_custom_stats(
            self.reference_name, str(self.reference_dir), mode="clean"
        )
        self._ref_stats_built = True

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Verify the Clean-FID reference stats exist before training starts.

        Failing here (loud, immediate) is far better than failing 25k steps in,
        when each rank-0 retry causes a 30-min NCCL allreduce stall on rank > 0.
        Disable by passing require_cache_at_init=False.
        """
        if not self.require_cache_at_init:
            return
        if trainer.global_rank != 0:
            return
        from cleanfid import fid as _fid
        if not _fid.test_stats_exists(self.reference_name, mode="clean"):
            raise FileNotFoundError(
                f"MiniFIDCallback: clean-fid reference stats '{self.reference_name}' "
                f"are missing. Run `scripts/prefetch_cleanfid.sh` first to build them."
            )
        # Pre-symlink the Inception weights into /tmp here so the first FID call
        # never blocks on a download attempt mid-training.
        inc_home = Path.home() / ".cache" / "cleanfid_models" / "inception-2015-12-05.pt"
        inc_tmp = Path("/tmp/inception-2015-12-05.pt")
        if inc_home.exists() and not inc_tmp.exists():
            try:
                inc_tmp.symlink_to(inc_home)
            except FileExistsError:
                pass

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step + 1
        if step % self.every_n_steps != 0:
            return
        # All ranks enter — only rank 0 does the heavy work — all ranks sync at the
        # end. Without this barrier, rank > 0 returns immediately, runs the next
        # train step, and tries to allreduce against a still-busy rank 0 → NCCL
        # watchdog times out at 30 min and SIGABRTs the whole job.
        if trainer.global_rank == 0:
            try:
                self._compute_and_log(trainer, pl_module, step)
            except Exception as e:  # noqa: BLE001
                warn(f"MiniFIDCallback @ step {step}: {type(e).__name__}: {e}")
        if trainer.world_size > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()

    @torch.no_grad()
    def _compute_and_log(
        self, trainer: L.Trainer, pl_module: L.LightningModule, step: int
    ) -> None:
        from cleanfid import fid
        from torchvision.utils import save_image

        device = pl_module.device
        self._ensure_stats()
        self._ensure_adapter(device)
        self._ensure_reference_stats()

        # EMA model preferred, like SampleCallback.
        if getattr(pl_module, "ema", None) is not None:
            model = pl_module.ema.shadow
        else:
            model = pl_module.model
        model.eval()

        in_channels = pl_module.hparams.in_channels
        input_size = pl_module.hparams.input_size
        latent_shape = (in_channels, input_size, input_size)

        # Output dir (per-eval, deleted after FID computed).
        eval_dir = Path(trainer.default_root_dir) / "fid_tmp" / f"step_{step:08d}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        sampler = Sampler(pl_module.transport)
        sample_fn = sampler.sample_ode(
            sampling_method=self.sample_method, num_steps=self.sample_steps
        )
        n_ch = latent_shape[0]
        cfg_scale = self.cfg_scale

        def model_fn(x, t, y):
            out = model.forward_with_cfg(x, t, y, cfg_scale)
            return out[:, :n_ch]

        # Deterministic class ids (uniform across 1000 classes for diversity).
        gen_seed = torch.Generator().manual_seed(self.seed)
        all_class_ids = torch.randint(
            0, self.num_classes, (self.n_samples,), generator=gen_seed
        )

        info(
            f"MiniFIDCallback @ step {step}: generating {self.n_samples} images "
            f"(batch {self.sample_batch_size}, cfg={cfg_scale}, ode_steps={self.sample_steps})"
        )
        n_done = 0
        bsz = self.sample_batch_size
        while n_done < self.n_samples:
            this_b = min(bsz, self.n_samples - n_done)
            noise = torch.randn(this_b, *latent_shape, device=device)
            labels = all_class_ids[n_done : n_done + this_b].to(device)
            null_label = torch.full_like(labels, self.num_classes)
            noise_full = torch.cat([noise, noise], dim=0)
            labels_full = torch.cat([labels, null_label], dim=0)

            samples = sample_fn(noise_full, model_fn, y=labels_full)[-1][:this_b]
            if self._mean is not None and self._inv_std is not None:
                samples = samples / self._inv_std.to(device) + self._mean.to(device)
            images = self._adapter.decode(samples).clamp_(-1, 1)
            images = images.add(1).div(2).clamp_(0, 1)

            for k, img in enumerate(images):
                save_image(img, eval_dir / f"img_{n_done + k:06d}.png")
            n_done += this_b

        info(f"MiniFIDCallback @ step {step}: scoring with Clean-FID against {self.reference_name}")
        score = fid.compute_fid(
            str(eval_dir),
            dataset_name=self.reference_name,
            dataset_split="custom",
            mode="clean",
        )
        ok(f"MiniFIDCallback @ step {step}: clean-FID-{self.n_samples//1000}k = {score:.3f}")
        pl_module.log(
            f"val/clean_fid_{self.n_samples//1000}k", float(score), rank_zero_only=True
        )

        # Cleanup the per-eval images (keep only the score in metrics).
        try:
            shutil.rmtree(eval_dir)
        except Exception as e:  # noqa: BLE001
            warn(f"MiniFIDCallback: failed to rm {eval_dir}: {e}")
