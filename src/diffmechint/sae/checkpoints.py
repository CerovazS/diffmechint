"""Locate and load production Matryoshka SAE checkpoints (Lustre-tolerant)."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from diffmechint.utils import warn


def resolve_sae_ckpt(sae_root: Path, condition: str, layer: int, t_bin: int, dit_step: int) -> Path:
    """Return the latest complete `final_*` SAE dir for one (cell, DiT-step)."""
    base = sae_root / condition / f"L{layer}_T{t_bin}" / f"step_{dit_step:06d}"
    if not base.exists():
        raise FileNotFoundError(f"SAE step dir missing: {base}")
    finals: list[Path] = []
    last_exc: OSError | None = None
    for attempt in range(5):
        try:
            finals = sorted(
                p
                for p in base.glob("final_*")
                if p.is_dir() and (p / "sae_weights.safetensors").exists()
            )
            break
        except OSError as exc:  # transient Lustre metadata failures
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise FileNotFoundError(
            f"could not inspect SAE checkpoint dir {base}: {last_exc}"
        ) from last_exc
    if not finals:
        raise FileNotFoundError(f"no final_* SAE dir under {base}")
    return finals[-1]


def load_matryoshka_sae(ckpt_dir: Path, device: torch.device | str) -> torch.nn.Module:
    """Load a production Matryoshka SAE via SAELens, retrying transient IO errors."""
    from sae_lens import MatryoshkaBatchTopKTrainingSAE

    last_error: OSError | None = None
    for attempt in range(1, 4):
        try:
            sae = MatryoshkaBatchTopKTrainingSAE.load_from_disk(str(ckpt_dir), device=str(device))
            break
        except OSError as exc:
            last_error = exc
            if attempt == 3:
                raise
            wait_s = 5 * attempt
            warn(f"SAE load failed for {ckpt_dir} with {exc!r}; retrying in {wait_s}s")
            time.sleep(wait_s)
    else:
        raise last_error if last_error is not None else RuntimeError(
            f"failed to load SAE from {ckpt_dir}"
        )
    sae.eval()
    return sae
