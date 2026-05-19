"""GPU smoke for the 4 working tokenizer adapters — encode/decode round-trip + PSNR.

Run from repo root:
    uv run python scripts/smoke_adapters_gpu.py
"""

from __future__ import annotations

import math
import time
from io import BytesIO

import torch
import urllib.request

from diffmechint.tokenizers import build
from diffmechint.utils import error, info, ok, warn

# A canonical real-world image (ImageNet-friendly): one of torchvision's sample URLs.
# Falls back to a structured synthetic image if download fails.
IMG_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
)


def fetch_image(size: int = 256) -> torch.Tensor:
    """Return a real RGB image normalized to [-1, 1] of shape (1, 3, size, size)."""
    try:
        from PIL import Image
        from torchvision import transforms

        with urllib.request.urlopen(IMG_URL, timeout=10) as resp:
            data = resp.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(size, antialias=True),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        x = tf(img).unsqueeze(0)
        info(f"Loaded real image from {IMG_URL}: shape {tuple(x.shape)}")
        return x
    except Exception as e:  # noqa: BLE001
        warn(f"Image download failed ({e!r}); falling back to synthetic structured image.")
        # Structured synthetic: gradient + checkerboard + low-freq texture.
        from torch import linspace
        y = linspace(-1, 1, size).view(1, 1, size, 1).expand(1, 3, size, size)
        x = linspace(-1, 1, size).view(1, 1, 1, size).expand(1, 3, size, size)
        grad = (x + y) / 2
        cb = ((torch.arange(size).view(1, 1, size, 1) // 32) +
              (torch.arange(size).view(1, 1, 1, size) // 32)) % 2
        cb = cb.float() * 2 - 1
        texture = torch.sin(linspace(0, 4 * math.pi, size)).view(1, 1, size, 1) \
                  * torch.sin(linspace(0, 4 * math.pi, size)).view(1, 1, 1, size)
        return (0.5 * grad + 0.3 * cb + 0.2 * texture).clamp_(-1, 1)


def psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    mse = torch.mean((x - x_hat) ** 2).item()
    return 10 * math.log10(4.0 / max(mse, 1e-12))  # range 2.0 → max² = 4.0


def main() -> None:
    if not torch.cuda.is_available():
        error("CUDA not available — refusing to run a GPU smoke on CPU.")
        raise SystemExit(2)

    device = torch.device("cuda:0")
    info(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    x = fetch_image(256).to(device)
    info(f"Image batch: {tuple(x.shape)}, range [{x.min():.3f}, {x.max():.3f}]")

    adapters_to_test = ["sd_vae", "eq_vae", "repa_e", "dc_ae_1_0"]
    results: list[tuple[str, float, tuple[int, ...], float]] = []

    for name in adapters_to_test:
        info(f"--- {name} ---")
        adapter = build(name)
        try:
            t0 = time.perf_counter()
            adapter.load()
            adapter.to(device)
            t_load = time.perf_counter() - t0
            n_params = sum(p.numel() for p in adapter.parameters())
            info(f"  loaded in {t_load:.1f}s ({n_params / 1e6:.1f}M params)")

            t0 = time.perf_counter()
            z = adapter.encode(x)
            x_hat = adapter.decode(z)
            t_rt = time.perf_counter() - t0
            db = psnr(x, x_hat)
            results.append((name, db, tuple(z.shape), t_rt))
            ok(f"  z={tuple(z.shape)} dtype={z.dtype} | round-trip {t_rt*1000:.1f}ms | PSNR {db:.2f} dB")
        except Exception as e:  # noqa: BLE001
            error(f"  {name} failed: {type(e).__name__}: {e}")
            results.append((name, float("nan"), (), -1.0))
        finally:
            del adapter
            torch.cuda.empty_cache()

    info("\n=== Summary ===")
    for name, db, shape, t_rt in results:
        info(f"{name:>12}: PSNR={db:6.2f} dB  z={shape}  rt={t_rt*1000:.0f}ms")
    bad = [n for n, db, _, _ in results if not (db > 25.0)]
    if bad:
        warn(f"PSNR < 25 dB on: {bad}")
    else:
        ok("All adapters cleared the 25 dB acceptance threshold.")


if __name__ == "__main__":
    main()
