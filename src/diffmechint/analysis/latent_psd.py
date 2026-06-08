"""E36-pre: radial DCT power spectra of tokenizer latents + power-law fit (Spectrum Matching profile)."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

from diffmechint.spectral import channel_power, radial_band_energy
from diffmechint.utils import info, ok
from diffmechint.utils.plotting import PALETTE_B

VAL_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents_val")
CONDITIONS = ["sd_vae", "eq_vae", "repa_e"]


def condition_profile(cond: str, n_bins: int, batch: int = 2048, max_images: int | None = None) -> np.ndarray:
    """Mean radial DCT power profile `(n_bins,)` over the condition's val latents (native grid)."""
    shards = sorted(glob.glob(str(VAL_ROOT / cond / "*.h5")))
    if not shards:
        raise FileNotFoundError(f"no val shards under {VAL_ROOT / cond}")
    total = torch.zeros(n_bins, dtype=torch.float64)
    count = 0
    for s in shards:
        with h5py.File(s, "r") as f:
            n = f["latents"].shape[0]
            for lo in range(0, n, batch):
                if max_images is not None and count >= max_images:
                    break
                lat = torch.from_numpy(f["latents"][lo : lo + batch]).float()
                prof = radial_band_energy(channel_power(lat), n_bins=n_bins)
                total += prof.double().sum(dim=0)
                count += prof.shape[0]
        info(f"{cond}: {count} images")
        if max_images is not None and count >= max_images:
            break
    return (total / count).numpy()


def fit_alpha(profile: np.ndarray) -> tuple[float, float]:
    """OLS slope of log power vs log bin index on bins 1..n-1 → (alpha, r2); power ∝ k^-alpha."""
    k = np.arange(1, len(profile))
    x, y = np.log(k), np.log(profile[1:])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - resid.var() / y.var()
    return float(-slope), float(r2)


def write_repro(out_dir: Path, args: argparse.Namespace) -> None:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
    (out_dir / "reproducibility.md").write_text(
        "# Reproducibility — E36-pre latent PSD\n\n"
        f"- commit: `{sha}` (branch `{branch}`)\n"
        f"- command: `uv run python scripts/analysis/latent_psd.py --out_dir {out_dir} "
        f"--n_bins {args.n_bins}`\n"
        f"- inputs: `{VAL_ROOT}/<cond>/*.h5` (ImageNet-val latents, fp16, native (4,32,32))\n"
        "- env: CINECA Leonardo, `uv sync` venv, CPU-only (no GPU required)\n"
        "- framework: DCT-II ortho full-grid, radial binning verbatim Spectrum Matching "
        "(arXiv 2603.14645, `utils_DCT.py` branch DSM); `center='none'`\n"
        f"- seed: deterministic (no sampling); n_bins={args.n_bins}\n"
    )
    (out_dir / "commit.txt").write_text(f"{sha} {branch} https://github.com/CerovazS/diffmechint\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--n_bins", type=int, default=16)
    ap.add_argument("--max_images", type=int, default=None, help="cap per condition (smoke)")
    args = ap.parse_args()

    metrics = args.out_dir / "metrics"
    plots = args.out_dir / "plots"
    metrics.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    rows, fits = [], {}
    for cond in args.conditions:
        prof = condition_profile(cond, args.n_bins, max_images=args.max_images)
        alpha, r2 = fit_alpha(prof)
        fits[cond] = {"alpha": alpha, "r2": r2}
        ok(f"{cond}: alpha={alpha:.3f} (r2={r2:.3f})")
        for k, p in enumerate(prof):
            rows.append({"condition": cond, "bin": k, "power": float(p), "log_power": float(np.log(p + 1e-12 + 1.0))})

    with (metrics / "latent_psd.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (metrics / "alpha_fits.json").write_text(json.dumps(fits, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for ax, scale in zip(axes, ["log", "linear"], strict=True):
        for i, cond in enumerate(args.conditions):
            prof = np.array([r["power"] for r in rows if r["condition"] == cond])
            ax.plot(range(args.n_bins), prof, marker="o", ms=3.5, color=PALETTE_B[i % len(PALETTE_B)],
                    label=f"{cond} (alpha={fits[cond]['alpha']:.2f})")
        ax.set_yscale(scale)
        ax.set_xlabel("radial DCT bin")
        ax.set_ylabel("mean power")
        ax.set_facecolor("white")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("E36-pre — latent radial DCT power spectra (val, native 32x32)")
    fig.tight_layout()
    fig.savefig(plots / "latent_psd_overlay.png", dpi=180)

    write_repro(args.out_dir, args)
    ok(f"done → {args.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
