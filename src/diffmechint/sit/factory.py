"""Construct SiT models and enumerate training-run EMA checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from .models import SiT_models

NUM_CLASSES = 1000


def build_sit_model(
    model_name: str,
    in_channels: int,
    input_size: int,
    device: torch.device | str,
    *,
    num_classes: int = NUM_CLASSES,
) -> torch.nn.Module:
    """Instantiate a SiT variant in eval mode with the program-standard config."""
    model = SiT_models[model_name](
        input_size=input_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ).to(device)
    model.eval()
    return model


def list_ema_checkpoints(run_dir: Path, steps: set[int] | None = None) -> list[tuple[int, Path]]:
    """Return (step, path) for every `step_*_ema.safetensors` under a training run."""
    ckpts: list[tuple[int, Path]] = []
    for path in sorted((run_dir / "checkpoints").glob("step_*_ema.safetensors")):
        step = int(path.name.split("_")[1])
        if steps is None or step in steps:
            ckpts.append((step, path))
    return ckpts
