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
    matryoshka_group_sizes: tuple[int, ...] | None = None,
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
      matryoshka_group_sizes: cumulative sizes for nested dictionaries.
        E.g. `(1024, 4096, 16384)` exposes coarse-to-fine sub-dictionaries.
        Used only for variant="matryoshka".
      metadata: free-form dict written into the SAE's saved `cfg.json` —
        we stamp `(condition, dit_ckpt_step, layer, t_bin, k)` here so the
        SAE is self-describing on disk.

    Returns: a SAELens TrainingSAE on `device` with `dtype`.
    """
    md = dict(metadata or {})
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
        if matryoshka_group_sizes is None:
            raise ValueError("matryoshka variant requires matryoshka_group_sizes.")
        cfg = MatryoshkaBatchTopKTrainingSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            k=k,
            dtype=str(dtype).split(".")[-1],
            device=str(device),
            normalize_activations=normalize_activations,
            matryoshka_group_sizes=tuple(matryoshka_group_sizes),
            metadata=md,
        )
        return MatryoshkaBatchTopKTrainingSAE(cfg)
    raise ValueError(f"Unknown SAE variant {variant!r}; choose topk / batch_topk / matryoshka.")
