"""Load per-feature latent normalization stats written by precompute_latents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_LATENTS_BASE = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents")


def load_latent_stats(
    adapter_name: str, latents_base: Path = DEFAULT_LATENTS_BASE
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return (mean, std, raw stats dict) shaped (1, C, 1, 1) for de/normalization."""
    stats = json.loads((latents_base / adapter_name / "stats.json").read_text())
    mean = torch.from_numpy(np.asarray(stats["per_feature_mean"], dtype=np.float32)).view(1, -1, 1, 1)
    std = torch.from_numpy(np.asarray(stats["per_feature_std"], dtype=np.float32)).view(1, -1, 1, 1)
    return mean, std, stats
