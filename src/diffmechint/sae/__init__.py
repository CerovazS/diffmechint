"""SAE training utilities — SAELens-backed."""

from .builder import SAEVariant, build_sae
from .data_provider import hdf5_provider, synthetic_provider
from .eval import (
    evaluate_sae,
    evaluate_sae_on_tokens,
    load_val_tokens,
    write_metrics,
)
from .trainer import train_sae, warm_start_from, warm_started_sweep

__all__ = [
    "SAEVariant",
    "build_sae",
    "hdf5_provider",
    "synthetic_provider",
    "evaluate_sae",
    "evaluate_sae_on_tokens",
    "load_val_tokens",
    "write_metrics",
    "train_sae",
    "warm_start_from",
    "warm_started_sweep",
]
