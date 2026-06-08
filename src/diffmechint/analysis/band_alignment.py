"""E36: band-resolved linear CKA alignment spectrum across latents and DiT residual streams."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

from diffmechint.analysis.alignment import cell_path, linear_cka, read_rows
from diffmechint.sae.data_provider import patchify_latents
from diffmechint.spectral import (
    dct2,
    grid_to_tokens,
    idct2,
    octave_band_masks,
    scramble_band_signs,
    tokens_to_grid,
)
from diffmechint.utils import info, ok, warn, write_csv
from diffmechint.utils.plotting import PALETTE_B

SCRATCH_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint")
ACTIVATIONS_YNULL = SCRATCH_ROOT / "activations_ynull"
ACTIVATIONS_VAL = SCRATCH_ROOT / "activations_val"
LATENTS_VAL = SCRATCH_ROOT / "latents_val"
CONDITIONS = ("sd_vae", "eq_vae", "repa_e")
LAYERS = (3, 6, 9)
T_BINS = (0, 1, 2)
LAYER_PAIRS = ((3, 6), (6, 9), (3, 9))
T_PAIRS = ((0, 1), (1, 2), (0, 2))
ANCHOR_STEPS = (150_000, 200_000)
GRID = 16
PATCH = 2
BAND_NAMES = ("B0", "B1", "B2", "B3", "B4")
E16_BROADBAND_REFERENCE = 0.339
CSV_FIELDS = [
    "axis", "cond_a", "cond_b", "layer_a", "t_a", "layer_b", "t_b",
    "band", "cka", "n_tokens", "n_images",
]


def _token_positions(n_total: int, token_cap: int, seed: int) -> np.ndarray:
    """Deterministic sorted token subsample positions, shared by both sides of a comparison."""
    if n_total <= token_cap:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=token_cap, replace=False)).astype(np.int64)


def _gather_tokens(tokens: torch.Tensor, pos: np.ndarray) -> np.ndarray:
    flat = tokens.reshape(-1, tokens.shape[-1])
    idx = torch.as_tensor(pos, device=flat.device)
    return flat[idx].cpu().numpy()


def _band_token_matrix(coeffs: torch.Tensor, mask: torch.Tensor, pos: np.ndarray) -> np.ndarray:
    comp = grid_to_tokens(idct2(coeffs * mask.to(coeffs.dtype)))
    return _gather_tokens(comp, pos)


def band_cka_for_pair(
    tokens_a: torch.Tensor,
    tokens_b: torch.Tensor,
    grid_h: int,
    grid_w: int,
    masks: torch.Tensor,
    token_cap: int,
    seed: int,
    *,
    include_broadband: bool = True,
) -> tuple[dict[str, float], int]:
    """Per-band linear CKA between two paired `(N, T, d)` token sets; band `all` = unfiltered."""
    if tokens_a.shape != tokens_b.shape:
        raise ValueError(f"token shapes differ: {tuple(tokens_a.shape)} vs {tuple(tokens_b.shape)}")
    n, t, _ = tokens_a.shape
    pos = _token_positions(n * t, token_cap, seed)
    coeffs_a = dct2(tokens_to_grid(tokens_a, grid_h, grid_w))
    coeffs_b = dct2(tokens_to_grid(tokens_b, grid_h, grid_w))
    out: dict[str, float] = {}
    for b in range(masks.shape[0]):
        xa = _band_token_matrix(coeffs_a, masks[b], pos)
        xb = _band_token_matrix(coeffs_b, masks[b], pos)
        out[f"B{b}"] = linear_cka(xa, xb)
    del coeffs_a, coeffs_b
    if include_broadband:
        out["all"] = linear_cka(_gather_tokens(tokens_a, pos), _gather_tokens(tokens_b, pos))
    return out, int(pos.shape[0])


def scrambled_band_cka(
    tokens_a: torch.Tensor,
    tokens_b: torch.Tensor,
    grid_h: int,
    grid_w: int,
    masks: torch.Tensor,
    token_cap: int,
    seed: int,
    scramble_seed: int,
) -> tuple[dict[str, float], int]:
    """Lower anchor: per-band CKA of A's band against the sign-scrambled band of B."""
    if tokens_a.shape != tokens_b.shape:
        raise ValueError(f"token shapes differ: {tuple(tokens_a.shape)} vs {tuple(tokens_b.shape)}")
    n, t, _ = tokens_a.shape
    pos = _token_positions(n * t, token_cap, seed)
    coeffs_a = dct2(tokens_to_grid(tokens_a, grid_h, grid_w))
    out: dict[str, float] = {}
    for b in range(masks.shape[0]):
        gen = torch.Generator(device=tokens_b.device.type).manual_seed(scramble_seed + b)
        scrambled = scramble_band_signs(tokens_b, grid_h, grid_w, masks, b, gen)
        coeffs_s = dct2(tokens_to_grid(scrambled, grid_h, grid_w))
        xa = _band_token_matrix(coeffs_a, masks[b], pos)
        xb = _band_token_matrix(coeffs_s, masks[b], pos)
        out[f"B{b}"] = linear_cka(xa, xb)
        del scrambled, coeffs_s
    return out, int(pos.shape[0])


def manifest_sample_idx(root: Path, condition: str, dit_step: int) -> np.ndarray:
    path = root / condition / f"step_{dit_step}" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest missing: {path}")
    return np.asarray(json.loads(path.read_text())["sample_idx"], dtype=np.int64)


def load_ynull_tokens(
    root: Path, condition: str, dit_step: int, layer: int, t_bin: int,
    rows: np.ndarray, device: torch.device,
) -> torch.Tensor:
    arr = read_rows(cell_path(root, condition, dit_step, layer, t_bin), rows)
    return torch.from_numpy(arr).to(device)


def load_latent_tokens(
    latents_root: Path, condition: str, image_idx: np.ndarray, device: torch.device,
) -> torch.Tensor:
    """Gather latents by global image index across sorted val shards, patchified to the token grid."""
    shards = sorted((latents_root / condition).glob("*.h5"))
    if not shards:
        raise FileNotFoundError(f"no latent val shards under {latents_root / condition}")
    order = np.argsort(image_idx)
    sorted_idx = image_idx[order]
    out = None
    offset = 0
    for shard in shards:
        with h5py.File(shard, "r") as f:
            n = f["latents"].shape[0]
            if out is None:
                c, h, w = f["latents"].shape[1:]
                out = np.empty((image_idx.shape[0], c, h, w), dtype=np.float32)
            lo = np.searchsorted(sorted_idx, offset)
            hi = np.searchsorted(sorted_idx, offset + n)
            if hi > lo:
                local = sorted_idx[lo:hi] - offset
                out[order[lo:hi]] = np.asarray(f["latents"][local], dtype=np.float32)
            offset += n
    if int(sorted_idx.max()) >= offset:
        raise IndexError(f"image index {int(sorted_idx.max())} exceeds {offset} concatenated latents")
    tokens = patchify_latents(torch.from_numpy(out), PATCH)
    return tokens.to(device)


def _axis_band_means(rows: list[dict]) -> dict[tuple[str, str], float]:
    acc: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        acc.setdefault((str(r["axis"]), str(r["band"])), []).append(float(r["cka"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _plot_spectra(rows: list[dict], plots_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = _axis_band_means(rows)
    raw_axes = ["inter_latent", "inter_dit", "intra_layer", "intra_time", "anchor_upper", "anchor_lower"]
    x_bands = [*BAND_NAMES, "all"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="white")
    for i, axis in enumerate(raw_axes):
        y = [means.get((axis, b), np.nan) for b in x_bands]
        if all(np.isnan(v) for v in y):
            continue
        ax.plot(range(len(x_bands)), y, marker="o", ms=4,
                color=PALETTE_B[i % len(PALETTE_B)],
                linestyle="-" if i < len(PALETTE_B) else "--", label=axis)
    ax.set_xticks(range(len(x_bands)), x_bands)
    ax.set_xlabel("DCT octave band")
    ax.set_ylabel("mean linear CKA")
    ax.set_facecolor("white")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("E36 - raw band-resolved CKA per axis")
    fig.tight_layout()
    fig.savefig(plots_dir / "band_cka_raw.png", dpi=180)
    plt.close(fig)

    upper = {b: means.get(("anchor_upper", b), np.nan) for b in BAND_NAMES}
    rel_axes = ["inter_latent", "inter_dit", "intra_layer", "intra_time", "anchor_lower"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="white")
    for i, axis in enumerate(rel_axes):
        y = [means.get((axis, b), np.nan) / upper[b] for b in BAND_NAMES]
        if all(np.isnan(v) for v in y):
            continue
        ax.plot(range(len(BAND_NAMES)), y, marker="o", ms=4,
                color=PALETTE_B[i % len(PALETTE_B)], label=axis)
    ax.set_xticks(range(len(BAND_NAMES)), list(BAND_NAMES))
    ax.set_xlabel("DCT octave band")
    ax.set_ylabel("relative alignment: CKA(b) / anchor_upper(b)")
    ax.set_facecolor("white")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("E36 - alignment spectrum (relative to within-model checkpoint anchor)")
    fig.tight_layout()
    fig.savefig(plots_dir / "alignment_spectrum.png", dpi=180)
    plt.close(fig)


def _write_report(out_dir: Path, rows: list[dict], headline: dict) -> None:
    means = _axis_band_means(rows)
    axes = ["inter_latent", "inter_dit", "intra_layer", "intra_time", "anchor_upper", "anchor_lower"]
    bands = [*BAND_NAMES, "all"]
    lines = [
        "# E36 - Band-Resolved Alignment Spectrum",
        "",
        "> [!summary] TL;DR",
        "> Per-band linear CKA across 4 comparison axes plus within-checkpoint upper anchor",
        f"> and sign-scrambled lower anchor. Broadband inter_dit mean CKA = "
        f"=={headline.get('broadband_inter_dit_mean_cka', float('nan')):.3f}== "
        f"(E16 reference {E16_BROADBAND_REFERENCE}).",
        "",
        "## Mean CKA per axis x band",
        "",
        "| axis | " + " | ".join(bands) + " |",
        "|---|" + "---|" * len(bands),
    ]
    for axis in axes:
        cells = []
        for b in bands:
            v = means.get((axis, b))
            cells.append(f"{v:.4f}" if v is not None else "-")
        lines.append(f"| {axis} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"- n_images: {headline['n_images']}, token_cap: {headline['token_cap']}, "
        f"seed: {headline['seed']}",
        f"- conditions: {', '.join(headline['conditions'])}; cells: {headline['cells']}",
        "- Relative alignment plot: `plots/alignment_spectrum.png`; raw: `plots/band_cka_raw.png`.",
        "",
    ]
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_repro(out_dir: Path, args: argparse.Namespace) -> None:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
    (out_dir / "reproducibility.md").write_text(
        "# Reproducibility - E36 band-resolved alignment spectrum\n\n"
        f"- commit: `{sha}` (branch `{branch}`)\n"
        f"- command: `uv run python scripts/analysis/band_alignment.py --out_dir {out_dir} "
        f"--n_images {args.n_images} --token_cap {args.token_cap} --seed {args.seed} "
        f"--device {args.device}{' --smoke' if args.smoke else ''}`\n"
        f"- inputs: ynull activations `{args.activations_root}/<cond>/step_{args.dit_step}/<layer>_<tbin>.h5`, "
        f"y-true val anchors `{args.val_root}/<cond>/step_{{{ANCHOR_STEPS[0]},{ANCHOR_STEPS[1]}}}/`, "
        f"latents `{args.latents_root}/<cond>/*.h5`\n"
        "- env: CINECA Leonardo, `uv sync` venv; DCT band decomposition on torch "
        f"`{args.device}`, CKA in float64 numpy on CPU\n"
        "- framework: DCT-II ortho full-grid, 5 octave bands (Spectrum Matching, arXiv 2603.14645); "
        "image subset = seeded rng choice over ynull rows; token subsample shared across both sides "
        "of every comparison\n"
        f"- seed: {args.seed}; n_images={args.n_images}; token_cap={args.token_cap}\n"
    )
    (out_dir / "commit.txt").write_text(f"{sha} {branch} https://github.com/CerovazS/diffmechint\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_images", type=int, default=4096)
    ap.add_argument("--token_cap", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dit_step", type=int, default=200_000)
    ap.add_argument("--activations_root", type=Path, default=ACTIVATIONS_YNULL)
    ap.add_argument("--val_root", type=Path, default=ACTIVATIONS_VAL)
    ap.add_argument("--latents_root", type=Path, default=LATENTS_VAL)
    ap.add_argument("--smoke", action="store_true", help="256 images, 2 conditions, cell (6,1), token_cap 20000")
    args = ap.parse_args()

    if args.smoke:
        args.n_images = 256
        args.token_cap = 20_000
        conds = ["sd_vae", "eq_vae"]
        cells = [(6, 1)]
        layer_pairs = [(3, 6)]
        t_pairs = [(0, 1)]
        intra_layer_tbins = [1]
        intra_time_layers = [6]
    else:
        conds = list(CONDITIONS)
        cells = [(layer, t) for layer in LAYERS for t in T_BINS]
        layer_pairs = list(LAYER_PAIRS)
        t_pairs = list(T_PAIRS)
        intra_layer_tbins = list(T_BINS)
        intra_time_layers = list(LAYERS)

    device = torch.device(args.device)
    masks = octave_band_masks(GRID, GRID, device=device)
    metrics_dir = args.out_dir / "metrics"
    plots_dir = args.out_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_idx = manifest_sample_idx(args.activations_root, conds[0], args.dit_step)
    for cond in conds[1:]:
        other = manifest_sample_idx(args.activations_root, cond, args.dit_step)
        if not np.array_equal(other, base_idx):
            raise ValueError(f"ynull manifest sample_idx mismatch: {conds[0]} vs {cond}; row pairing invalid")
    rng = np.random.default_rng(args.seed)
    rows_sel = np.sort(rng.choice(base_idx.shape[0], size=min(args.n_images, base_idx.shape[0]), replace=False))
    image_idx = base_idx[rows_sel]
    n_images = int(rows_sel.shape[0])
    ok(f"selected {n_images} ynull rows (seed {args.seed}); device {device}; token_cap {args.token_cap}")

    csv_rows: list[dict] = []

    def add(axis, cond_a, cond_b, layer_a, t_a, layer_b, t_b, ckas, n_tokens, n_imgs):
        for band, cka in ckas.items():
            csv_rows.append({
                "axis": axis, "cond_a": cond_a, "cond_b": cond_b,
                "layer_a": layer_a, "t_a": t_a, "layer_b": layer_b, "t_b": t_b,
                "band": band, "cka": cka, "n_tokens": n_tokens, "n_images": n_imgs,
            })

    def flush():
        write_csv(metrics_dir / "band_cka.csv", csv_rows, fieldnames=CSV_FIELDS)

    info("axis 1: inter_latent")
    latent_tokens = {c: load_latent_tokens(args.latents_root, c, image_idx, device) for c in conds}
    for a, b in itertools.combinations(conds, 2):
        ckas, n_tok = band_cka_for_pair(
            latent_tokens[a], latent_tokens[b], GRID, GRID, masks, args.token_cap, args.seed + 1
        )
        add("inter_latent", a, b, "", "", "", "", ckas, n_tok, n_images)
        info(f"  {a} vs {b}: broadband {ckas['all']:.4f}")
    del latent_tokens
    flush()

    info("axis 2 + 6: inter_dit + anchor_lower")
    for layer, t_bin in cells:
        toks = {
            c: load_ynull_tokens(args.activations_root, c, args.dit_step, layer, t_bin, rows_sel, device)
            for c in conds
        }
        for a, b in itertools.combinations(conds, 2):
            ckas, n_tok = band_cka_for_pair(
                toks[a], toks[b], GRID, GRID, masks, args.token_cap, args.seed + 1
            )
            add("inter_dit", a, b, layer, t_bin, layer, t_bin, ckas, n_tok, n_images)
            info(f"  L{layer}/T{t_bin} {a} vs {b}: broadband {ckas['all']:.4f}")
            sckas, n_tok = scrambled_band_cka(
                toks[a], toks[b], GRID, GRID, masks, args.token_cap, args.seed + 1, args.seed + 97
            )
            add("anchor_lower", a, b, layer, t_bin, layer, t_bin, sckas, n_tok, n_images)
        del toks
        if device.type == "cuda":
            torch.cuda.empty_cache()
        flush()

    info("axis 3: intra_layer")
    for cond in conds:
        for t_bin in intra_layer_tbins:
            needed = sorted({layer for pair in layer_pairs for layer in pair})
            toks = {
                layer: load_ynull_tokens(args.activations_root, cond, args.dit_step, layer, t_bin, rows_sel, device)
                for layer in needed
            }
            for la, lb in layer_pairs:
                ckas, n_tok = band_cka_for_pair(
                    toks[la], toks[lb], GRID, GRID, masks, args.token_cap, args.seed + 1
                )
                add("intra_layer", cond, cond, la, t_bin, lb, t_bin, ckas, n_tok, n_images)
            del toks
            if device.type == "cuda":
                torch.cuda.empty_cache()
            flush()

    info("axis 4: intra_time")
    for cond in conds:
        for layer in intra_time_layers:
            needed = sorted({t for pair in t_pairs for t in pair})
            toks = {
                t: load_ynull_tokens(args.activations_root, cond, args.dit_step, layer, t, rows_sel, device)
                for t in needed
            }
            for ta, tb in t_pairs:
                ckas, n_tok = band_cka_for_pair(
                    toks[ta], toks[tb], GRID, GRID, masks, args.token_cap, args.seed + 1
                )
                add("intra_time", cond, cond, layer, ta, layer, tb, ckas, n_tok, n_images)
            del toks
            if device.type == "cuda":
                torch.cuda.empty_cache()
            flush()

    info("axis 5: anchor_upper (y-true val, adjacent checkpoints)")
    for cond in conds:
        idx_a = manifest_sample_idx(args.val_root, cond, ANCHOR_STEPS[0])
        idx_b = manifest_sample_idx(args.val_root, cond, ANCHOR_STEPS[1])
        shared, pos_a, pos_b = np.intersect1d(idx_a, idx_b, return_indices=True)
        if shared.size == 0:
            raise ValueError(f"no shared val sample_idx between checkpoints for {cond}")
        if shared.size < 256:
            warn(f"anchor_upper {cond}: only {shared.size} shared images; estimator may be noisy")
        if shared.size > n_images:
            sel = np.sort(np.random.default_rng(args.seed).choice(shared.size, size=n_images, replace=False))
            shared, pos_a, pos_b = shared[sel], pos_a[sel], pos_b[sel]
        for layer, t_bin in cells:
            a = read_rows(cell_path(args.val_root, cond, ANCHOR_STEPS[0], layer, t_bin), pos_a)
            b = read_rows(cell_path(args.val_root, cond, ANCHOR_STEPS[1], layer, t_bin), pos_b)
            ckas, n_tok = band_cka_for_pair(
                torch.from_numpy(a).to(device), torch.from_numpy(b).to(device),
                GRID, GRID, masks, args.token_cap, args.seed + 1,
            )
            add("anchor_upper", cond, cond, layer, t_bin, layer, t_bin, ckas, n_tok, int(shared.size))
            del a, b
        if device.type == "cuda":
            torch.cuda.empty_cache()
        flush()

    means = _axis_band_means(csv_rows)
    axis_band_mean = {}
    for (axis, band), v in sorted(means.items()):
        axis_band_mean.setdefault(axis, {})[band] = v
    headline = {
        "axis_band_mean_cka": axis_band_mean,
        "broadband_inter_dit_mean_cka": means.get(("inter_dit", "all")),
        "e16_broadband_reference": E16_BROADBAND_REFERENCE,
        "n_images": n_images,
        "token_cap": args.token_cap,
        "seed": args.seed,
        "conditions": conds,
        "cells": [f"L{layer}_T{t}" for layer, t in cells],
        "smoke": bool(args.smoke),
    }
    (metrics_dir / "headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    _plot_spectra(csv_rows, plots_dir)
    _write_report(args.out_dir, csv_rows, headline)
    write_repro(args.out_dir, args)
    bb = headline["broadband_inter_dit_mean_cka"]
    ok(f"done -> {args.out_dir} (broadband inter_dit mean CKA {bb:.4f}, E16 ref {E16_BROADBAND_REFERENCE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
