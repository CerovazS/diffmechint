"""DCT-2D spectral framework (Spectrum Matching, arXiv 2603.14645): radial PSD and octave band decomposition."""

from __future__ import annotations

import torch

OCTAVE_EDGES: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 1.0)
N_OCTAVE_BANDS: int = len(OCTAVE_EDGES)  # B0 (DC) + 4 radial octaves


def dct_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix `C` of shape `(n, n)`: `X = C @ x` along one axis."""
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    c = torch.cos(torch.pi * (2 * i + 1) * k / (2 * n))
    c = c * torch.sqrt(torch.tensor(2.0 / n, device=device, dtype=dtype))
    c[0] = c[0] / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return c


def dct2(x: torch.Tensor) -> torch.Tensor:
    """Separable 2D DCT-II (ortho) over the last two dims. No mean centering (`center='none'`)."""
    h, w = x.shape[-2], x.shape[-1]
    ch = dct_matrix(h, device=x.device, dtype=x.dtype)
    cw = dct_matrix(w, device=x.device, dtype=x.dtype)
    return torch.einsum("uh,...hw,vw->...uv", ch, x, cw)


def idct2(coeffs: torch.Tensor) -> torch.Tensor:
    """Inverse of `dct2` (DCT-III via the transposed orthonormal basis)."""
    h, w = coeffs.shape[-2], coeffs.shape[-1]
    ch = dct_matrix(h, device=coeffs.device, dtype=coeffs.dtype)
    cw = dct_matrix(w, device=coeffs.device, dtype=coeffs.dtype)
    return torch.einsum("uh,...uv,vw->...hw", ch, coeffs, cw)


def radial_bin_map(h: int, w: int, n_bins: int, device=None) -> torch.Tensor:
    """Verbatim Spectrum Matching binning: `floor(sqrt(u²+v²) / rmax * n_bins)`, clamped, DC in bin 0."""
    yy = torch.arange(h, device=device).view(h, 1).float()
    xx = torch.arange(w, device=device).view(1, w).float()
    rr = torch.sqrt(yy**2 + xx**2)
    rmax = rr.max().clamp_min(1.0)
    bin_id = torch.floor(rr / rmax * n_bins).long()
    return bin_id.clamp(0, n_bins - 1)


def octave_band_masks(h: int, w: int, device=None) -> torch.Tensor:
    """Bool masks `(B, h, w)` partitioning the DCT grid: B0 = DC only, then octaves of normalized radius."""
    yy = torch.arange(h, device=device).view(h, 1).float()
    xx = torch.arange(w, device=device).view(1, w).float()
    rr = torch.sqrt(yy**2 + xx**2)
    rnorm = rr / rr.max().clamp_min(1.0)
    masks = torch.zeros(N_OCTAVE_BANDS, h, w, dtype=torch.bool, device=device)
    masks[0, 0, 0] = True
    for b in range(1, N_OCTAVE_BANDS):
        lo, hi = OCTAVE_EDGES[b - 1], OCTAVE_EDGES[b]
        masks[b] = (rnorm > lo) & (rnorm <= hi)
    masks[1, 0, 0] = False  # DC belongs to B0 even though rnorm == 0 <= edge
    return masks


def band_decompose(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """`(..., h, w)` → `(B, ..., h, w)` band-limited components; exact partition (`sum over B == x`)."""
    coeffs = dct2(x)
    parts = [idct2(coeffs * m.to(coeffs.dtype)) for m in masks]
    return torch.stack(parts, dim=0)


def tokens_to_grid(tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """`(N, T, d)` row-major tokens → `(N, d, grid_h, grid_w)` spatial grid (matches patchify_latents order)."""
    n, t, d = tokens.shape
    if t != grid_h * grid_w:
        raise ValueError(f"token count {t} != grid {grid_h}x{grid_w}.")
    return tokens.reshape(n, grid_h, grid_w, d).permute(0, 3, 1, 2)


def grid_to_tokens(grid: torch.Tensor) -> torch.Tensor:
    """`(N, d, h, w)` → `(N, h*w, d)` row-major tokens; inverse of `tokens_to_grid`."""
    n, d, h, w = grid.shape
    return grid.permute(0, 2, 3, 1).reshape(n, h * w, d)


def band_decompose_tokens(tokens: torch.Tensor, grid_h: int, grid_w: int, masks: torch.Tensor) -> torch.Tensor:
    """`(N, T, d)` tokens → `(B, N, T, d)` band-limited token sets on the spatial token grid."""
    grid = tokens_to_grid(tokens, grid_h, grid_w)
    parts = band_decompose(grid, masks)
    return torch.stack([grid_to_tokens(p) for p in parts], dim=0)


def token_dct_coeffs(tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """`(N, T, d)` tokens → `(N, d, grid_h, grid_w)` DCT coefficients, cacheable for per-band reuse."""
    return dct2(tokens_to_grid(tokens, grid_h, grid_w))


def band_tokens_from_coeffs(coeffs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`(N, d, h, w)` cached DCT coeffs + `(h, w)` bool mask → `(N, h*w, d)` band-limited tokens.

    One-band counterpart of `band_decompose_tokens` that avoids re-running the DCT
    and stacking all bands at once (memory-friendly for large activation tensors).
    """
    return grid_to_tokens(idct2(coeffs * mask.to(coeffs.dtype)))


def channel_power(x: torch.Tensor) -> torch.Tensor:
    """`(N, C, h, w)` → `(N, h, w)` DCT power averaged over channels (squared ortho DCT-II coefficients)."""
    return dct2(x).pow(2).mean(dim=1)


def radial_band_energy(power: torch.Tensor, n_bins: int) -> torch.Tensor:
    """`(N, h, w)` power → `(N, n_bins)` mean power per radial bin (Spectrum Matching profile)."""
    h, w = power.shape[-2], power.shape[-1]
    bin_id = radial_bin_map(h, w, n_bins, device=power.device).reshape(-1)
    flat = power.reshape(power.shape[0], -1)
    out = torch.zeros(power.shape[0], n_bins, device=power.device, dtype=power.dtype)
    counts = torch.zeros(n_bins, device=power.device, dtype=power.dtype)
    out.index_add_(1, bin_id, flat)
    counts.index_add_(0, bin_id, torch.ones_like(bin_id, dtype=power.dtype))
    return out / counts.clamp_min(1.0)


def scramble_band_signs(
    tokens: torch.Tensor, grid_h: int, grid_w: int, masks: torch.Tensor, band: int, generator: torch.Generator
) -> torch.Tensor:
    """Null surrogate: flip random per-(sample, u, v) signs of DCT coefficients inside `band`.

    Signs are shared across the feature dim, destroying cross-representation
    correspondence while preserving per-channel band power.
    """
    grid = tokens_to_grid(tokens, grid_h, grid_w)
    coeffs = dct2(grid)
    n = coeffs.shape[0]
    signs = torch.randint(0, 2, (n, 1, grid_h, grid_w), generator=generator, device=coeffs.device) * 2 - 1
    m = masks[band].to(coeffs.dtype)
    coeffs = coeffs * (1 - m) + coeffs * m * signs.to(coeffs.dtype)
    return grid_to_tokens(idct2(coeffs))
