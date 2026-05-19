"""Per-SAE quality metrics — recon cosine, density, dead-feature count."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import torch
from sae_lens import TrainingSAE


@torch.no_grad()
def evaluate_sae_on_tokens(
    sae: TrainingSAE,
    tokens: torch.Tensor,
    *,
    batch_size: int = 4096,
    density_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Deterministic held-out eval on a fixed `(T, D)` token tensor.

    Used by the inline trainer hook to monitor generalization every N
    optimizer steps. The tensor lives on CPU; batches are shipped to
    `sae.device` one at a time. `sae` is moved to eval mode for the
    forward and restored after.

    Returned dict keys (all scalars, plot-ready): `ev`, `mse`, `l0_mean`,
    `recon_cosine`, `recon_l2`, `density`, `live_features`, `dead_features`,
    `dead_pct`, `n_features`, `n_tokens_evaluated`.
    """
    was_training = sae.training
    sae.eval()
    device = sae.device
    sae_dtype = next(sae.parameters()).dtype
    d_in = int(tokens.shape[-1])
    d_sae = int(sae.cfg.d_sae)

    # Two passes baked into one: pre-compute global mean (EV denominator).
    x_mean = tokens.mean(dim=0).to(device=device, dtype=torch.float32)  # (D,)

    sq_err_sum = 0.0
    sq_var_sum = 0.0
    cos_sum = 0.0
    l2_sum = 0.0
    l0_sum = 0
    n_tokens = 0
    feature_active_any = torch.zeros(d_sae, dtype=torch.bool, device=device)
    feature_density_sum = torch.zeros(d_sae, dtype=torch.float64, device=device)

    try:
        for start in range(0, tokens.shape[0], batch_size):
            end = min(start + batch_size, tokens.shape[0])
            x = tokens[start:end].to(device=device, dtype=sae_dtype, non_blocking=True)
            x_hat = sae(x)
            z = sae.encode(x)

            x_f = x.float()
            x_hat_f = x_hat.float()
            diff = x_f - x_hat_f
            sq_err_sum += float(diff.pow(2).sum().item())
            sq_var_sum += float((x_f - x_mean).pow(2).sum().item())

            x_n = torch.nn.functional.normalize(x_f, dim=-1)
            xh_n = torch.nn.functional.normalize(x_hat_f, dim=-1)
            cos_sum += float((x_n * xh_n).sum(dim=-1).sum().item())
            l2_sum += float(diff.norm(dim=-1).sum().item())

            active = z.abs() > density_threshold
            feature_active_any |= active.any(dim=0)
            feature_density_sum += active.sum(dim=0).to(torch.float64)
            l0_sum += int(active.sum().item())

            n_tokens += int(x.shape[0])
    finally:
        if was_training:
            sae.train()

    n = max(n_tokens, 1)
    live = int(feature_active_any.sum().item())
    return {
        "ev": 1.0 - (sq_err_sum / max(sq_var_sum, 1e-12)),
        "mse": sq_err_sum / (n * d_in),
        "l0_mean": l0_sum / n,
        "recon_cosine": cos_sum / n,
        "recon_l2": l2_sum / n,
        "density": float(feature_density_sum.sum().item()) / (n * d_sae),
        "live_features": live,
        "dead_features": d_sae - live,
        "dead_pct": (d_sae - live) / d_sae,
        "n_features": d_sae,
        "n_tokens_evaluated": int(n),
    }


def load_val_tokens(
    shard_path: Path | str,
    *,
    max_tokens: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load a Phase-3 HDF5 activation shard into a flat CPU `(T, D)` tensor.

    Deterministic (no shuffle): the first `max_tokens` rows are returned in
    shard order so re-loads at different optimizer steps see the same data,
    a precondition for monotonic EV / MSE telemetry over a single SAE run.
    """
    with h5py.File(str(shard_path), "r") as f:
        arr = f["activations"][()]
    t = torch.from_numpy(arr).to(dtype)  # (N, T, D)
    t = t.reshape(-1, t.shape[-1])  # (N*T, D)
    if max_tokens is not None and t.shape[0] > max_tokens:
        t = t[:max_tokens].contiguous()
    return t


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
