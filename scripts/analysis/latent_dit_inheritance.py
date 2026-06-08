"""E32: inheritance test — do SiT layers host the latent SAE dictionary?

Per (condition x layer x t_bin) cell of the B/2 y-null grid:

A) Activation-profile matching (E12-style): per-image max-activation profiles
   of latent SAE features vs DiT SAE features on the SAME ImageNet-val images
   (paired via the extraction manifest `sample_idx`), feature-feature Pearson
   matrix, per-latent-feature best |corr| + Hungarian assignment.
   Controls: shuffled image pairing; cross-condition latent dictionary.

B) Affine dictionary transfer (E17-style, token level): ridge 768->16, latent
   SAE encode/decode, ridge 16->768; EV vs the ridge-only round trip (the
   16-dim bottleneck baseline) and vs the native DiT SAE.
   Control: shuffled token pairing before the ridge fit.

Preregistered predictions (program.md E32): best match at early layers and
low-noise bins, decaying with depth; controls flat.

E37 extension (band-resolved inheritance): `--bands` band-decomposes BOTH the
latent tokens and the DiT tokens (same octave band b on both spatially-1:1
16x16 token grids) upstream of the profile matching, the ridge fits, and all
controls, so every metric and control is band-matched. Each output row carries
a `band` column (band=-1 is broadband, identical to the E32 path). Note on B0
(DC): per-token variance within an image is zero, so the ridge transfer on B0
effectively maps per-image means -- expected, kept.

Usage:
    uv run python scripts/analysis/latent_dit_inheritance.py \
        --out_dir outputs/<run_root> [--n_images 4096] [--bands all | 0 1 ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from diffmechint.sae import patchify_latents
from diffmechint.spectral import (
    N_OCTAVE_BANDS,
    band_tokens_from_coeffs,
    octave_band_masks,
    token_dct_coeffs,
)
from diffmechint.utils import info, ok

SCRATCH = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint")
DIT_SAE_ROOT = Path("/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k")
LATENT_SAE_BASE = SCRATCH / "sae_latents"
ACT_ROOT = SCRATCH / "activations_ynull"
VAL_ROOT = SCRATCH / "latents_val"
CONDITIONS = ["sd_vae", "eq_vae", "repa_e"]
# Cross-condition control pairing: latent dictionary of CROSS[cond] vs DiT of cond.
CROSS = {"sd_vae": "eq_vae", "eq_vae": "repa_e", "repa_e": "sd_vae"}
DIT_STEP = 200_000


class BandedTokens:
    """Lazy per-band views of `(M, T, d)` tokens: DCT coeffs cached once, one band
    materialized at a time (never the full `(B, M, T, d)` stack).

    `band(-1)` returns the original tensor object untouched (broadband, E32-identical).
    """

    def __init__(self, tokens: torch.Tensor):
        self.tokens = tokens
        self.grid = round(tokens.shape[1] ** 0.5)
        if self.grid * self.grid != tokens.shape[1]:
            raise ValueError(f"token count {tokens.shape[1]} is not a square grid")
        self._coeffs: torch.Tensor | None = None
        self._masks: torch.Tensor | None = None

    def band(self, b: int) -> torch.Tensor:
        if b < 0:
            return self.tokens
        if self._coeffs is None:
            self._coeffs = token_dct_coeffs(self.tokens, self.grid, self.grid)
            self._masks = octave_band_masks(self.grid, self.grid)
        return band_tokens_from_coeffs(self._coeffs, self._masks[b])


def parse_bands(spec: list[str] | None) -> list[int]:
    """`None` -> broadband only; `['all']` -> broadband plus bands 0..4; else explicit indices."""
    if spec is None:
        return [-1]
    if spec == ["all"]:
        return [-1, *range(N_OCTAVE_BANDS)]
    bands = [int(b) for b in spec]
    for b in bands:
        if not -1 <= b < N_OCTAVE_BANDS:
            raise ValueError(f"band {b} outside [-1, {N_OCTAVE_BANDS - 1}]")
    return bands


def load_latent_sae(condition: str, run_id: str, cell: str, device: str):
    from sae_lens import MatryoshkaBatchTopKTrainingSAE

    cell_dir = LATENT_SAE_BASE / run_id / condition / cell
    finals = sorted(cell_dir.glob("final_*"))
    if not finals:
        raise FileNotFoundError(f"no final_* under {cell_dir}")
    sae = MatryoshkaBatchTopKTrainingSAE.load_from_disk(str(finals[-1]), device=device)
    sae.eval()
    return sae


def load_dit_sae(condition: str, layer: int, t_bin: int, device: str):
    from sae_lens import MatryoshkaBatchTopKTrainingSAE

    base = DIT_SAE_ROOT / condition / f"L{layer}_T{t_bin}" / f"step_{DIT_STEP:06d}"
    finals = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("final_"))
    if not finals:
        raise FileNotFoundError(f"no final_* under {base}")
    sae = MatryoshkaBatchTopKTrainingSAE.load_from_disk(str(finals[-1]), device=device)
    sae.eval()
    return sae


def load_latent_rows(condition: str, global_idx: np.ndarray, patch_size: int) -> torch.Tensor:
    """Gather latent maps for arbitrary global val indices → (M, T, 16) tokens."""
    shards = sorted((VAL_ROOT / condition).glob("*.h5"))
    sizes = []
    for s in shards:
        with h5py.File(s, "r") as f:
            sizes.append(f["latents"].shape[0])
    offsets = np.cumsum([0, *sizes])
    out = torch.empty(len(global_idx), 4, 32, 32)
    order = np.argsort(global_idx)
    for si, shard in enumerate(shards):
        lo, hi = offsets[si], offsets[si + 1]
        mask = (global_idx[order] >= lo) & (global_idx[order] < hi)
        if not mask.any():
            continue
        rows = global_idx[order][mask] - lo
        with h5py.File(shard, "r") as f:
            arr = torch.from_numpy(f["latents"][np.sort(rows)]).float()
        # f[sorted rows] is h5py-required; un-sort back to `rows` order.
        unsort = np.argsort(np.argsort(rows))
        out[order[mask]] = arr[unsort]
    return patchify_latents(out, patch_size)  # (M, 256, 16)


@torch.no_grad()
def per_image_max_profiles(sae, tokens: torch.Tensor, device: str, images_per_chunk: int = 64):
    """(M, T, D) tokens → (M, d_sae) per-image max activation profiles.

    Reduces over tokens chunk-by-chunk: storing all token-level codes first
    would need M*T*d_sae floats (~34 GB for 1024 images at d_sae=32768).
    """
    m, t, d = tokens.shape
    sae_dtype = next(sae.parameters()).dtype
    outs = []
    for start in range(0, m, images_per_chunk):
        chunk = tokens[start : start + images_per_chunk]
        flat = chunk.reshape(-1, d).to(device=device, dtype=sae_dtype)
        z = sae.encode(flat).float()  # (n*t, d_sae)
        outs.append(z.reshape(chunk.shape[0], t, -1).max(dim=1).values.cpu())
    return torch.cat(outs)  # (M, d_sae)


def rank_transform(a: torch.Tensor) -> torch.Tensor:
    """Per-column rank transform → Spearman when fed to `corr_matrix`.

    The per-image max profiles are heavy-tailed (E31: ultra-rare features fire
    on 1-2 images), which inflates Pearson best-match tails to ~1 by chance
    over 32k candidate columns. Ranks tame the tails.
    """
    return a.argsort(dim=0).argsort(dim=0).float()


def corr_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Column-standardized correlation: (M,Fa),(M,Fb) → (Fa,Fb)."""
    az = (a - a.mean(0)) / a.std(0).clamp_min(1e-8)
    bz = (b - b.mean(0)) / b.std(0).clamp_min(1e-8)
    return (az.T @ bz) / a.shape[0]


def eligible_columns(prof: torch.Tensor, min_images: int) -> torch.Tensor:
    """Features active on at least `min_images` images (kills 1-2-image tails)."""
    return (prof > 0).sum(dim=0) >= min_images


def match_stats(c: torch.Tensor) -> dict:
    """Best-match and Hungarian statistics; `c` already restricted to
    eligible (latent-row, DiT-column) features."""
    if c.numel() == 0:
        # Band-limited profiles can leave zero eligible features on one side.
        nan = float("nan")
        return {
            "n_lat": int(c.shape[0]), "n_dit": int(c.shape[1]),
            "mean_best_abscorr": nan, "median_best_abscorr": nan,
            "p90_best_abscorr": nan, "hungarian_mean_abscorr": nan,
        }
    c_abs = c.abs()
    best = c_abs.max(dim=1).values
    row, col = linear_sum_assignment(-c_abs.numpy())
    hung = c_abs.numpy()[row, col]
    return {
        "n_lat": int(c.shape[0]),
        "n_dit": int(c.shape[1]),
        "mean_best_abscorr": round(float(best.mean()), 4),
        "median_best_abscorr": round(float(best.median()), 4),
        "p90_best_abscorr": round(float(best.quantile(0.9)), 4),
        "hungarian_mean_abscorr": round(float(hung.mean()), 4),
    }


def ridge_fit(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """Closed-form ridge X→Y with bias column; returns W ((dx+1), dy)."""
    xb = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    a = xb.T @ xb + lam * x.shape[0] * torch.eye(xb.shape[1])
    return torch.linalg.solve(a, xb.T @ y)


def ridge_apply(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1)], dim=1) @ w


def ev(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    num = (x - x_hat).pow(2).sum()
    den = (x - x.mean(0)).pow(2).sum().clamp_min(1e-12)
    return float(1.0 - num / den)


@torch.no_grad()
def transfer_block(
    lat_tokens: torch.Tensor,  # (M, 256, 16)
    act_tokens: torch.Tensor,  # (M, 256, 768) fp32
    lat_sae,
    dit_sae,
    device: str,
    n_tokens: int,
    seed: int,
    shuffled: bool = False,
) -> dict:
    """E17-style affine transfer with the 16-dim bottleneck baseline."""
    g = torch.Generator().manual_seed(seed)
    m, t, _ = lat_tokens.shape
    flat_idx = torch.randperm(m * t, generator=g)[:n_tokens]
    x_lat = lat_tokens.reshape(-1, lat_tokens.shape[-1])[flat_idx]
    x_dit = act_tokens.reshape(-1, act_tokens.shape[-1])[flat_idx]
    if shuffled:
        x_lat = x_lat[torch.randperm(n_tokens, generator=g)]
    n_train = int(0.8 * n_tokens)
    w_down = ridge_fit(x_dit[:n_train], x_lat[:n_train])
    w_up = ridge_fit(x_lat[:n_train], x_dit[:n_train])
    xe_dit, xe_lat = x_dit[n_train:], x_lat[n_train:]

    down = ridge_apply(w_down, xe_dit)  # (n_eval, 16)
    ridge_only = ridge_apply(w_up, down)
    sae_dtype = next(lat_sae.parameters()).dtype
    z = lat_sae.decode(lat_sae.encode(down.to(device=device, dtype=sae_dtype))).float().cpu()
    ridge_sae = ridge_apply(w_up, z)
    dit_dtype = next(dit_sae.parameters()).dtype
    native = dit_sae.decode(dit_sae.encode(xe_dit.to(device=device, dtype=dit_dtype))).float().cpu()

    ev_ridge = ev(xe_dit, ridge_only)
    ev_sae = ev(xe_dit, ridge_sae)
    return {
        "ridge_only_ev": round(ev_ridge, 4),
        "ridge_latentsae_ev": round(ev_sae, 4),
        "native_ditsae_ev": round(ev(xe_dit, native), 4),
        "inheritance_ratio": round(ev_sae / ev_ridge, 4) if ev_ridge > 1e-6 else float("nan"),
        "latent_ev_through_map": round(ev(xe_lat, ridge_apply(w_down, xe_dit)), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent_run_id", type=str, default="latent_sae_ef_20260606")
    p.add_argument("--latent_cell", type=str, default="d256_k8")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9])
    p.add_argument("--t_bins", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_images", type=int, default=4096)
    p.add_argument("--min_images", type=int, default=20,
                   help="Eligibility floor: features must fire on >= this many "
                        "images (kills the 1-2-image correlation tails).")
    p.add_argument("--n_transfer_tokens", type=int, default=100_000)
    p.add_argument("--bands", nargs="+", default=None,
                   help="E37: octave band indices 0-4 (-1 = broadband), or 'all' for "
                        "broadband plus every band. Omit for broadband only "
                        "(E32-identical numbers).")
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    band_list = parse_bands(args.bands)

    (args.out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    rows = []
    for cond in args.conditions:
        manifest = json.loads(
            (ACT_ROOT / cond / f"step_{DIT_STEP:06d}" / "manifest.json").read_text()
        )
        sample_idx = np.asarray(manifest["sample_idx"], dtype=np.int64)
        assert manifest.get("y_null") is True
        m = args.n_images
        img_global = sample_idx[:m]  # extraction order is a stratified shuffle

        lat_sae = load_latent_sae(cond, args.latent_run_id, args.latent_cell, args.device)
        lat_banded = BandedTokens(load_latent_rows(cond, img_global, args.patch_size))

        cross_cond = CROSS[cond]
        cross_sae = load_latent_sae(cross_cond, args.latent_run_id, args.latent_cell, args.device)
        cross_banded = BandedTokens(load_latent_rows(cross_cond, img_global, args.patch_size))

        # Decompose the latent side once per (condition, band); reused across cells.
        lat_side: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        cross_side: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for b in band_list:
            lat_tokens = lat_banded.band(b)
            lat_prof = per_image_max_profiles(lat_sae, lat_tokens, args.device)
            elig_lat = eligible_columns(lat_prof, args.min_images)
            lat_side[b] = (lat_tokens, lat_prof, elig_lat, rank_transform(lat_prof[:, elig_lat]))
            cross_prof = per_image_max_profiles(cross_sae, cross_banded.band(b), args.device)
            elig_cross = eligible_columns(cross_prof, args.min_images)
            cross_side[b] = (elig_cross, rank_transform(cross_prof[:, elig_cross]))
            info(f"{cond} band={b}: latent profiles ready (eligible {int(elig_lat.sum())}, "
                 f"cross={cross_cond} eligible {int(elig_cross.sum())})")

        g = torch.Generator().manual_seed(args.seed)
        shuffle_perm = torch.randperm(m, generator=g)

        for layer in args.layers:
            for t_bin in args.t_bins:
                shard = ACT_ROOT / cond / f"step_{DIT_STEP:06d}" / f"{layer}_{t_bin}.h5"
                with h5py.File(shard, "r") as f:
                    acts = torch.from_numpy(f["activations"][:m]).float()  # (M,256,768)
                dit_sae = load_dit_sae(cond, layer, t_bin, args.device)
                act_banded = BandedTokens(acts)

                for band in band_list:
                    lat_tokens, lat_prof, elig_lat, lat_rank = lat_side[band]
                    elig_cross, cross_rank = cross_side[band]
                    acts_b = act_banded.band(band)
                    dit_prof = per_image_max_profiles(dit_sae, acts_b, args.device)
                    elig_dit = eligible_columns(dit_prof, args.min_images)
                    dit_rank = rank_transform(dit_prof[:, elig_dit])

                    real = match_stats(corr_matrix(lat_rank, dit_rank))
                    shuf = match_stats(
                        corr_matrix(rank_transform(lat_prof[shuffle_perm][:, elig_lat]), dit_rank)
                    )
                    cross = match_stats(corr_matrix(cross_rank, dit_rank))

                    tr = transfer_block(
                        lat_tokens, acts_b, lat_sae, dit_sae, args.device,
                        args.n_transfer_tokens, args.seed,
                    )
                    tr_shuf = transfer_block(
                        lat_tokens, acts_b, lat_sae, dit_sae, args.device,
                        args.n_transfer_tokens, args.seed, shuffled=True,
                    )

                    delta_best = round(real["mean_best_abscorr"] - shuf["mean_best_abscorr"], 4)
                    delta_hung = round(
                        real["hungarian_mean_abscorr"] - shuf["hungarian_mean_abscorr"], 4
                    )
                    row = {
                        "condition": cond, "layer": layer, "t_bin": t_bin,
                        "band": band, "n_images": m,
                        **{f"match_{k}": v for k, v in real.items()},
                        **{f"shuffled_{k}": v for k, v in shuf.items()
                           if k not in ("n_lat", "n_dit")},
                        **{f"cross_{cross_cond}_{k}": v for k, v in cross.items()
                           if k != "n_dit"},
                        "delta_mean_best_abscorr": delta_best,
                        "delta_hungarian_abscorr": delta_hung,
                        **tr,
                        **{f"shuffled_{k}": v for k, v in tr_shuf.items()
                           if k in ("ridge_only_ev", "ridge_latentsae_ev")},
                    }
                    rows.append(row)
                    ok(
                        f"{cond} L{layer} T{t_bin} band={band}: "
                        f"Δbest|r|={delta_best} Δhung={delta_hung} "
                        f"(real {real['mean_best_abscorr']} shuf {shuf['mean_best_abscorr']} "
                        f"cross {cross['mean_best_abscorr']}) "
                        f"ratio={tr['inheritance_ratio']} "
                        f"[ridge {tr['ridge_only_ev']} → +sae {tr['ridge_latentsae_ev']}]"
                    )
                    del acts_b, dit_prof
                del acts, act_banded, dit_sae

    out = args.out_dir / "metrics" / "inheritance_cells.csv"
    keys = list(rows[0].keys())
    with out.open("w", newline="") as f:
        # cross-condition column name varies per condition; union the keys.
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in keys, k))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    ok(f"{len(rows)} cells → {out}")


if __name__ == "__main__":
    main()
