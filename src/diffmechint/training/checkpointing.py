"""Fractional-fraction checkpoint callback (PLAN §6.4).

Saves at training fractions {2%, 5%, 10%, 25%, 50%, 75%, 100%} of `max_steps`.
Each save writes:
  step_{step:08d}.safetensors           — live model weights
  step_{step:08d}_ema.safetensors       — EMA shadow weights
  step_{step:08d}_metadata.json         — step, fraction, loss, lr
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback
from safetensors.torch import save_file

from diffmechint.utils import info, ok

DEFAULT_FRACTIONS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00)


class FractionalCheckpoint(Callback):
    def __init__(
        self,
        out_dir: str,
        max_steps: int,
        fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    ) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.fractions = sorted(set(fractions))
        self.targets = sorted({max(1, int(round(f * max_steps))) for f in self.fractions})
        self._next_idx = 0
        info(f"FractionalCheckpoint: targets at steps {self.targets}")

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        # Only save on rank 0.
        if trainer.global_rank != 0:
            return
        if self._next_idx >= len(self.targets):
            return
        target_step = self.targets[self._next_idx]
        if (trainer.global_step + 1) >= target_step:
            self._save(trainer, pl_module, target_step)
            self._next_idx += 1

    def _save(self, trainer: L.Trainer, pl_module: L.LightningModule, target_step: int) -> None:
        step_str = f"{target_step:08d}"
        live_path = self.out_dir / f"step_{step_str}.safetensors"
        ema_path = self.out_dir / f"step_{step_str}_ema.safetensors"
        meta_path = self.out_dir / f"step_{step_str}_metadata.json"

        # Live model.
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in pl_module.model.state_dict().items()},
            str(live_path),
        )

        # EMA shadow (if present).
        if getattr(pl_module, "ema", None) is not None:
            save_file(
                {
                    k: v.detach().cpu().contiguous()
                    for k, v in pl_module.ema.shadow.state_dict().items()
                },
                str(ema_path),
            )

        # Metadata.
        fraction = target_step / max(self.max_steps, 1)
        opt = trainer.optimizers[0] if trainer.optimizers else None
        lr = float(opt.param_groups[0]["lr"]) if opt else float("nan")
        last_loss = trainer.callback_metrics.get("train/loss")
        last_loss_v = float(last_loss.detach().cpu()) if last_loss is not None else float("nan")
        meta = {
            "step": target_step,
            "fraction": fraction,
            "lr": lr,
            "train_loss": last_loss_v,
            "max_steps": self.max_steps,
            "global_step": int(trainer.global_step),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        ok(f"Checkpoint @ step {target_step:08d} (frac {fraction:.0%}) → {self.out_dir}")
