"""Factory wrappers around SAELens TrainingSAE variants.

We never reimplement the SAE — we configure it. Picks among:
  - topk          → TopKTrainingSAE
  - batch_topk    → BatchTopKTrainingSAE       (default, Bussmann et al.)
  - matryoshka    → MatryoshkaBatchTopKTrainingSAE
"""

from __future__ import annotations

from typing import Literal

import torch
from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
    TrainingSAE,
)
from sae_lens.saes.sae import SAEMetadata

SAEVariant = Literal["topk", "batch_topk", "matryoshka"]


def build_sae(
    *,
    d_in: int,
    d_sae: int,
    k: int,
    variant: SAEVariant = "batch_topk",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    normalize_activations: str = "expected_average_only_in",
    matryoshka_widths: tuple[int, ...] | None = None,
    metadata: dict | None = None,
) -> TrainingSAE:
    """Instantiate a SAELens `TrainingSAE` of the chosen variant.

    Args:
      d_in : residual-stream width (e.g. 768 for SiT-B, 1152 for SiT-XL).
      d_sae: dictionary size; the proposal sets 16384.
      k    : top-k sparsity.
      variant: which SAE family.
      normalize_activations: SAELens preset; the default subtracts the
        expected mean per channel from inputs only (recommended for diffusion
        residuals where the mean is non-zero).
      matryoshka_widths: cumulative latent prefix widths for nested matryoshka
        levels. E.g. `(4096, 8192, 16384, 32768)` with `k=256` and `d_sae=32768`
        gives 4 readout endpoints with effective K ≈ (32, 64, 128, 256) — features
        are split proportionally across prefixes via BatchTopK selection.
        Used only for variant="matryoshka".
      metadata: free-form dict written into the SAE's saved `cfg.json` —
        we stamp `(condition, dit_ckpt_step, layer, t_bin, k)` here so the
        SAE is self-describing on disk.

    Returns: a SAELens TrainingSAE on `device` with `dtype`.
    """
    # SAELens 6.x expects an SAEMetadata dataclass (not a plain dict). Build it
    # via from_dict so our free-form keys (condition, layer, t_bin, ...) land in
    # the extra_data bucket without colliding with SAEMetadata's reserved fields.
    md = SAEMetadata.from_dict(dict(metadata or {}))
    if variant == "topk":
        cfg = TopKTrainingSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            k=k,
            dtype=str(dtype).split(".")[-1],
            device=str(device),
            normalize_activations=normalize_activations,
            metadata=md,
        )
        return TopKTrainingSAE(cfg)
    if variant == "batch_topk":
        cfg = BatchTopKTrainingSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            k=k,
            dtype=str(dtype).split(".")[-1],
            device=str(device),
            normalize_activations=normalize_activations,
            metadata=md,
        )
        return BatchTopKTrainingSAE(cfg)
    if variant == "matryoshka":
        if matryoshka_widths is None:
            raise ValueError("matryoshka variant requires matryoshka_widths.")
        cfg = MatryoshkaBatchTopKTrainingSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            k=k,
            dtype=str(dtype).split(".")[-1],
            device=str(device),
            normalize_activations=normalize_activations,
            matryoshka_widths=list(matryoshka_widths),
            metadata=md,
        )
        return MatryoshkaBatchTopKTrainingSAE(cfg)
    raise ValueError(f"Unknown SAE variant {variant!r}; choose topk / batch_topk / matryoshka.")
