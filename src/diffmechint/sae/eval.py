"""Per-SAE quality metrics — recon cosine, density, dead-feature count."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from sae_lens import TrainingSAE


@torch.no_grad()
def evaluate_sae(
    sae: TrainingSAE,
    data_provider: Iterator[torch.Tensor],
    *,
    n_batches: int = 32,
    density_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Compute reconstruction + sparsity statistics on `n_batches` of activations.

    Returns:
      recon_cosine     : mean cosine(x, x_hat) over all tokens
      recon_l2         : mean ||x - x_hat||_2
      density          : fraction of features active per token, averaged
      live_features    : count of features that fired at least once
      dead_features    : d_sae - live_features
      n_features       : d_sae
    """
    sae.eval()
    cos_sum = 0.0
    l2_sum = 0.0
    n_tokens = 0
    feature_active_any = torch.zeros(sae.cfg.d_sae, dtype=torch.bool, device=sae.device)
    feature_density_sum = torch.zeros(sae.cfg.d_sae, device=sae.device)

    for i, batch in enumerate(data_provider):
        if i >= n_batches:
            break
        x = batch.to(device=sae.device, dtype=sae.dtype)
        out = sae(x) if isinstance(sae(x), torch.Tensor) else sae.forward_with_features(x) \
            if hasattr(sae, "forward_with_features") else None  # noqa: F841
        # SAELens TrainingSAE forward returns the reconstructed x by default.
        x_hat = sae(x)
        # Encode to get sparse code for density.
        z = sae.encode(x)

        # cosine per token
        x_n = torch.nn.functional.normalize(x, dim=-1)
        xh_n = torch.nn.functional.normalize(x_hat, dim=-1)
        cos_per = (x_n * xh_n).sum(dim=-1)
        cos_sum += cos_per.sum().item()
        l2_sum += (x - x_hat).norm(dim=-1).sum().item()
        n_tokens += x.shape[0]

        active = z.abs() > density_threshold  # (B, d_sae)
        feature_active_any |= active.any(dim=0)
        feature_density_sum += active.float().sum(dim=0)

    n = max(n_tokens, 1)
    live = int(feature_active_any.sum().item())
    return {
        "recon_cosine": cos_sum / n,
        "recon_l2": l2_sum / n,
        "density": float(feature_density_sum.sum().item()) / (n * sae.cfg.d_sae),
        "live_features": live,
        "dead_features": int(sae.cfg.d_sae) - live,
        "n_features": int(sae.cfg.d_sae),
        "n_tokens_evaluated": int(n),
    }


def write_metrics(metrics: dict[str, Any], out_path: Path | str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
