"""LightningModule wrapping SiT trained with FM-OT (Linear interpolant + velocity)."""

from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from diffmechint.sit import SiT_models, create_transport
from diffmechint.utils import info, ok

from .ema import EMA


class SiTLightningModule(L.LightningModule):
    """SiT + FM-OT training loop.

    Args:
      model_name: key into `SiT_models` (e.g. "SiT-B/2", "SiT-L/2", "SiT-XL/2").
      input_size: latent spatial size (= image_size / vae_compression). 32 for
                  SD-VAE @ 256², 16 for VA-VAE f16, 8 for DC-AE f32, 16 for RAE.
      in_channels: latent channel count from the tokenizer adapter.
      num_classes: number of label classes (1000 for ImageNet-1K).
      class_dropout_prob: CFG dropout (default 0.1 per SiT paper).
      learn_sigma: SiT outputs (eps, sigma) when True. Default True for parity
                  with the upstream paper.
      transport_cfg: dict of kwargs for `create_transport` (path_type, prediction,
                  loss_weight, train_eps, sample_eps).
      lr: AdamW learning rate (default 1e-4 per SiT).
      weight_decay: 0 per SiT.
      betas: AdamW betas.
      warmup_steps: linear warm-up; flat after.
      ema_decay: EMA decay; 0.9999 per SiT/REPA conventions.
    """

    def __init__(
        self,
        model_name: str = "SiT-B/2",
        input_size: int = 32,
        in_channels: int = 4,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = True,
        transport_cfg: dict[str, Any] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 10_000,
        ema_decay: float = 0.9999,
        ema_resume_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if model_name not in SiT_models:
            raise KeyError(f"Unknown model {model_name!r}; choose from {sorted(SiT_models)}.")
        self.model = SiT_models[model_name](
            input_size=input_size,
            in_channels=in_channels,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=learn_sigma,
        )
        tc = dict(transport_cfg or {})
        tc.setdefault("path_type", "Linear")
        tc.setdefault("prediction", "velocity")
        tc.setdefault("loss_weight", None)
        self.transport = create_transport(**tc)
        self.ema: EMA | None = None  # built on setup() to land on the right device
        info(
            f"SiTLightningModule: {model_name} input_size={input_size} in_ch={in_channels} "
            f"path={tc['path_type']} pred={tc['prediction']}"
        )

    def on_fit_start(self) -> None:
        # Build EMA after the live model has been moved to its training device,
        # so the shadow copy lives on the same device.
        if self.ema is None:
            self.ema = EMA(self.model, decay=self.hparams.ema_decay)
            self.ema.shadow.to(self.device)
        # If resuming from a previous run, overwrite EMA shadow weights with the
        # saved EMA. The live model is loaded earlier in train.py before fit().
        ema_path = self.hparams.ema_resume_path
        if ema_path:
            from safetensors.torch import load_file
            ema_sd = load_file(ema_path)
            self.ema.shadow.load_state_dict(ema_sd)
            ok(f"EMA shadow resumed from {ema_path}")

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        z, y = batch["latent"], batch["label"]
        # SiT's training_losses signature: (model, x_1, model_kwargs)
        loss_dict = self.transport.training_losses(self.model, z, dict(y=y))
        loss = loss_dict["loss"].mean()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        # Same loss as training but on held-out latents. Class label still passed:
        # we want val loss under the same conditional regime as train (no CFG drop
        # at val time — class_dropout is applied inside the model under .train()).
        z, y = batch["latent"], batch["label"]
        with torch.no_grad():
            loss_dict = self.transport.training_losses(self.model, z, dict(y=y))
        loss = loss_dict["loss"].mean()
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:  # noqa: ARG002
        # Lightning calls EMA.update *after* the optimizer step via on_train_batch_end.
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:  # noqa: ARG002
        if self.ema is not None:
            self.ema.update(self.model)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
        )
        warmup_steps = max(int(self.hparams.warmup_steps), 1)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ema_sd = checkpoint.get("ema")
        if ema_sd is not None and self.ema is not None:
            self.ema.load_state_dict(ema_sd)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
