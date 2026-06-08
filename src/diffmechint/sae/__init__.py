"""SAE training utilities — SAELens-backed."""

from .builder import SAEVariant, build_sae
from .checkpoints import load_matryoshka_sae, resolve_sae_ckpt
from .data_provider import (
    first_latent_d_in,
    hdf5_provider,
    latent_hdf5_provider,
    load_latent_val_tokens,
    patchify_latents,
    synthetic_provider,
)
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
    "evaluate_sae",
    "evaluate_sae_on_tokens",
    "first_latent_d_in",
    "hdf5_provider",
    "latent_hdf5_provider",
    "load_latent_val_tokens",
    "load_matryoshka_sae",
    "load_val_tokens",
    "patchify_latents",
    "resolve_sae_ckpt",
    "synthetic_provider",
    "train_sae",
    "warm_start_from",
    "warm_started_sweep",
    "write_metrics",
]
