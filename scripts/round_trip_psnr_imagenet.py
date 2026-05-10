"""Decode the smoke-precomputed latents and compute PSNR vs the original images.

Closes CHECKLIST 1.13 (acceptance: PSNR > 22 dB per adapter on real ImageNet
samples). Run from inside an `srun --gres=gpu:1 ...` allocation:

    bash scripts/round_trip_psnr_imagenet.sh
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# HF cache must be set before any HF/diffusers import.
FAST = Path(os.environ["FAST"])
os.environ["HF_HOME"] = str(FAST / "lcerovaz" / "hf_cache")
os.environ["HF_HUB_CACHE"] = str(FAST / "lcerovaz" / "hf_cache" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(FAST / "lcerovaz" / "hf_cache" / "transformers")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import h5py  # noqa: E402
import torch  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from diffmechint.tokenizers import build  # noqa: E402
from diffmechint.utils import error, info, ok, warn  # noqa: E402

DATA_DIR = "/leonardo_scratch/fast/IscrC_YENDRI/imagenet/train"
LATENT_BASE = FAST / "lcerovaz" / "diffmechint" / "latents"
ADAPTERS = ["sd_vae", "eq_vae", "repa_e", "dc_ae_1_0"]
N_IMAGES = 256
BATCH_SIZE = 32
PSNR_THRESHOLD = 22.0  # PLAN §14 acceptance


def _psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """PSNR for tensors in [-1, 1]; max range = 2.0."""
    mse = torch.mean((x - x_hat).float() ** 2).item()
    return 10.0 * math.log10(4.0 / max(mse, 1e-12))


def _load_originals(n: int) -> torch.Tensor:
    """Load the first `n` ImageNet train images with the same transform used in precompute."""
    transform = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    ds = ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    chunks: list[torch.Tensor] = []
    seen = 0
    for imgs, _ in loader:
        chunks.append(imgs)
        seen += imgs.shape[0]
        if seen >= n:
            break
    return torch.cat(chunks, dim=0)[:n]


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available — PSNR will still compute but slowly.")

    info(f"Loading {N_IMAGES} ImageNet originals from {DATA_DIR}")
    originals = _load_originals(N_IMAGES)
    info(f"originals shape: {tuple(originals.shape)} dtype: {originals.dtype}")

    results: dict[str, dict] = {}
    failed: list[str] = []
    for tok in ADAPTERS:
        latent_dir = LATENT_BASE / f"{tok}_smoke"
        info(f"--- {tok} (latent_dir={latent_dir}) ---")
        if not (latent_dir / "00000.h5").exists():
            error(f"{tok}: shard missing")
            failed.append(tok)
            continue

        adapter = build(tok)
        adapter.load()
        adapter.to(device)

        with h5py.File(latent_dir / "00000.h5") as f:
            z_all = torch.from_numpy(f["latents"][:N_IMAGES]).float()  # fp16 → fp32
        info(f"latents: shape {tuple(z_all.shape)}")

        # Decode in batches; keep on GPU.
        decoded_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, z_all.shape[0], BATCH_SIZE):
                z_batch = z_all[i : i + BATCH_SIZE].to(device)
                x_hat = adapter.decode(z_batch).clamp_(-1, 1).cpu()
                decoded_chunks.append(x_hat)
        decoded = torch.cat(decoded_chunks, dim=0)
        info(f"decoded shape: {tuple(decoded.shape)}")

        psnr = _psnr(originals, decoded)
        # Per-image distribution for diagnostics.
        per_img = torch.tensor(
            [_psnr(originals[i : i + 1], decoded[i : i + 1]) for i in range(N_IMAGES)]
        )
        verdict = "OK" if psnr >= PSNR_THRESHOLD else "FAIL"
        msg = (
            f"{tok}: mean PSNR {psnr:.2f} dB | per-img median {per_img.median():.2f} "
            f"min {per_img.min():.2f} max {per_img.max():.2f}"
        )
        if verdict == "OK":
            ok(msg)
        else:
            error(msg + f" — below threshold {PSNR_THRESHOLD} dB")
            failed.append(tok)
        results[tok] = {
            "mean_psnr_db": psnr,
            "median_psnr_db": float(per_img.median()),
            "min_psnr_db": float(per_img.min()),
            "max_psnr_db": float(per_img.max()),
            "n_images": N_IMAGES,
            "threshold_db": PSNR_THRESHOLD,
            "verdict": verdict,
        }

        # Free GPU memory before next adapter.
        adapter.to("cpu")
        del adapter
        torch.cuda.empty_cache() if device.type == "cuda" else None

    out_path = LATENT_BASE / "round_trip_psnr.json"
    out_path.write_text(json.dumps(results, indent=2))
    info(f"Wrote {out_path}")

    if failed:
        error(f"{len(failed)}/{len(ADAPTERS)} adapters failed: {failed}")
        return 1
    ok(f"All {len(ADAPTERS)} adapters PSNR ≥ {PSNR_THRESHOLD} dB on real ImageNet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
