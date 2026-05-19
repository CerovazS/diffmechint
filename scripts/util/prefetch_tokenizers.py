"""Prefetch the 4 working tokenizer/VAE checkpoints into the HF cache on $FAST.

Run from the login node (compute nodes have no internet by default).
Sets HF_HOME to the project's $FAST cache dir, then instantiates and `.load()`s
each adapter; this triggers the from_pretrained() download path and warms the
cache so compute-node jobs can load offline.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

CACHE_DIR = Path(os.environ["FAST"]) / "lcerovaz" / "hf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")

# Import after env is set so HF respects it.
from diffmechint.tokenizers import build  # noqa: E402
from diffmechint.utils import error, info, ok  # noqa: E402

ADAPTERS = ["sd_vae", "eq_vae", "repa_e", "dc_ae_1_0"]


def main() -> int:
    info(f"HF cache: {CACHE_DIR}")
    failures: list[tuple[str, str]] = []
    for name in ADAPTERS:
        info(f"--- {name} ---")
        t0 = time.perf_counter()
        try:
            adapter = build(name)
            adapter.load()
            spec = adapter.spec
            n_params = sum(p.numel() for p in adapter.parameters())
            dt = time.perf_counter() - t0
            ok(
                f"{name}: loaded in {dt:.1f}s | "
                f"latent_shape={spec.latent_shape} scale={spec.scaling_factor} "
                f"params={n_params/1e6:.1f}M license={spec.license}"
            )
        except Exception as e:  # noqa: BLE001
            error(f"{name} FAILED: {type(e).__name__}: {e}")
            failures.append((name, repr(e)))
    if failures:
        error(f"{len(failures)}/{len(ADAPTERS)} adapter prefetch failures: {failures}")
        return 1
    ok(f"All {len(ADAPTERS)} adapters cached under {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
