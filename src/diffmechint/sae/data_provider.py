"""Data providers feeding SAELens `SAETrainer` from our Phase 3 HDF5 cells.

`SAETrainer` consumes `Iterator[torch.Tensor]` where each yielded tensor has
shape `(batch_size, d_in)`. We provide:
  - `hdf5_provider`  : streams (N, T, D) cells from disk, flattens tokens
  - `latent_hdf5_provider`: streams Phase 1 (N, C, H, W) latent shards,
    patchified to SiT x_embedder token order
  - `synthetic_provider`: structured K-sparse feature mixtures for tests
"""

from __future__ import annotations

import glob
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Sequence

import h5py
import torch


def hdf5_provider(
    shard_paths: Sequence[str | Path] | str | Path,
    batch_size: int,
    *,
    device: str | torch.device = "cuda",
    flatten_tokens: bool = True,
    dtype: torch.dtype = torch.float32,
    loop_forever: bool = True,
    shuffle: bool = True,
    seed: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield `(batch_size, d_in)` tensors from Phase 3 activation shards.

    Each shard is an HDF5 file with dataset `activations` of shape
    `(N, T, D)` fp16 (see `diffmechint.hooks.ActivationBuffer`).

    Args:
      shard_paths: explicit list of `.h5` files, OR a directory glob, OR a
                   single `.h5` path.
      batch_size : SAE training batch size in tokens.
      device     : where to ship tensors (`cuda` recommended).
      flatten_tokens: if True (default) collapse `(N, T, D) → (N*T, D)` so
                   each spatial token is one SAE sample. If False, yield
                   `(N, T, D)` records sample-by-sample (token-aware variants).
      dtype      : output dtype after fp16 → cast.
      loop_forever: if True, restart from the first shard when exhausted.
                    SAELens controls termination via `total_training_samples`.
      shuffle    : shuffle within each shard load.
      seed       : torch generator seed when shuffling.
    """
    paths = _normalize_paths(shard_paths)
    if not paths:
        raise FileNotFoundError(f"No HDF5 shards resolved from {shard_paths!r}.")
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    while True:
        for path in paths:
            with h5py.File(path, "r") as f:
                arr = torch.from_numpy(f["activations"][()]).to(dtype=dtype)  # (N, T, D)
            if flatten_tokens:
                arr = arr.reshape(-1, arr.shape[-1])  # (N*T, D)
            if shuffle:
                perm = torch.randperm(arr.shape[0], generator=gen)
                arr = arr[perm]
            for batch in arr.split(batch_size):
                if batch.shape[0] < batch_size:
                    continue  # drop_last to keep SAE update sizes constant
                yield batch.to(device=device, non_blocking=True)
        if not loop_forever:
            return


def patchify_latents(latents: torch.Tensor, patch_size: int) -> torch.Tensor:
    """`(N, C, H, W)` → `(N, T, C*p*p)` in SiT x_embedder token order.

    Tokens are row-major over the `(H/p, W/p)` grid; each token vector is the
    flattened `(C, p, p)` block, matching timm PatchEmbed's Conv2d weight
    layout so latent tokens align 1:1 with SiT residual-stream tokens.
    """
    n, c, h, w = latents.shape
    p = int(patch_size)
    if h % p or w % p:
        raise ValueError(f"latent spatial dims ({h}, {w}) not divisible by patch_size={p}.")
    th, tw = h // p, w // p
    x = latents.reshape(n, c, th, p, tw, p)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(n, th * tw, c * p * p)


def latent_hdf5_provider(
    shard_paths: Sequence[str | Path] | str | Path,
    batch_size: int,
    *,
    patch_size: int = 2,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    loop_forever: bool = True,
    shuffle: bool = True,
    seed: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield `(batch_size, C*p*p)` tensors from Phase 1 latent shards.

    Each shard is an HDF5 file with dataset `latents` of shape `(N, C, H, W)`
    fp16 (see `diffmechint.training.precompute_latents`). The latent map is
    patchified exactly as the SiT x_embedder sees it (`patchify_latents`),
    then spatial tokens are flattened into SAE samples — the latent analogue
    of `hdf5_provider`.
    """
    paths = _normalize_paths(shard_paths)
    if not paths:
        raise FileNotFoundError(f"No HDF5 shards resolved from {shard_paths!r}.")
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    while True:
        for path in paths:
            with h5py.File(path, "r") as f:
                arr = torch.from_numpy(f["latents"][()]).to(dtype=dtype)  # (N, C, H, W)
            arr = patchify_latents(arr, patch_size)  # (N, T, D)
            arr = arr.reshape(-1, arr.shape[-1])  # (N*T, D)
            if shuffle:
                perm = torch.randperm(arr.shape[0], generator=gen)
                arr = arr[perm]
            for batch in arr.split(batch_size):
                if batch.shape[0] < batch_size:
                    continue  # drop_last to keep SAE update sizes constant
                yield batch.to(device=device, non_blocking=True)
        if not loop_forever:
            return


def load_latent_val_tokens(
    shard_path: Path | str,
    *,
    patch_size: int = 2,
    max_tokens: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load a Phase 1 latent shard into a flat CPU `(T, D)` token tensor.

    Deterministic (no shuffle), mirroring `eval.load_val_tokens`: the first
    `max_tokens` tokens in shard order, so inline val telemetry over one SAE
    run always sees identical data.
    """
    with h5py.File(str(shard_path), "r") as f:
        arr = torch.from_numpy(f["latents"][()]).to(dtype)  # (N, C, H, W)
    t = patchify_latents(arr, patch_size).reshape(-1, arr.shape[1] * patch_size * patch_size)
    if max_tokens is not None and t.shape[0] > max_tokens:
        t = t[:max_tokens].contiguous()
    return t


def first_latent_d_in(shard_paths: Sequence[str | Path] | str | Path, patch_size: int = 2) -> int:
    """Token width `C*p*p` of the first resolved latent shard."""
    paths = _normalize_paths(shard_paths)
    if not paths:
        raise FileNotFoundError(f"No HDF5 shards resolved from {shard_paths!r}.")
    with h5py.File(paths[0], "r") as f:
        c = int(f["latents"].shape[1])
    return c * int(patch_size) ** 2


def synthetic_provider(
    d_in: int,
    *,
    batch_size: int,
    n_features_dict: int,
    k_sparse: int,
    noise_std: float = 0.05,
    device: str | torch.device = "cuda",
    seed: int = 0,
    loop_forever: bool = True,
) -> Iterator[torch.Tensor]:
    """Yield K-sparse feature mixtures + noise (smoke / synthetic-recovery).

    Generates a fixed dictionary `D ∈ R^{d_in × n_features_dict}` once, then
    each batch picks `k_sparse` features per sample with random non-negative
    coefficients and adds isotropic Gaussian noise. A correctly-trained SAE
    with `k = k_sparse` should recover the dictionary atoms.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # unit-norm dictionary atoms
    D = torch.randn(d_in, n_features_dict, generator=gen)
    D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-6)
    D = D.to(device=device)

    while True:
        # pick K active features per sample
        idx = torch.stack(
            [torch.randperm(n_features_dict, generator=gen)[:k_sparse] for _ in range(batch_size)]
        )  # (B, k)
        coeff = torch.rand(batch_size, k_sparse, generator=gen).abs() + 0.5  # >= 0.5
        # build sparse code (B, n_features) and project to (B, d_in)
        code = torch.zeros(batch_size, n_features_dict)
        code.scatter_(1, idx, coeff)
        x = code @ D.T.cpu()  # (B, d_in)
        x = x + noise_std * torch.randn(batch_size, d_in, generator=gen)
        yield x.to(device=device)
        if not loop_forever:
            return


def _normalize_paths(p: Sequence[str | Path] | str | Path) -> list[str]:
    if isinstance(p, (str, Path)):
        path = Path(p)
        if path.is_dir():
            return sorted(glob.glob(str(path / "*.h5")))
        if path.suffix == ".h5":
            return [str(path)]
        # assume glob pattern
        return sorted(glob.glob(str(path)))
    return [str(x) for x in p]


def take(provider: Iterable[torch.Tensor], n: int) -> list[torch.Tensor]:
    """Test helper: pull `n` batches from a provider into a list."""
    out: list[torch.Tensor] = []
    it = iter(provider)
    for _ in range(n):
        out.append(next(it))
    return out
