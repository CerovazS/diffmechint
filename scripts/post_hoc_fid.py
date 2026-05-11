"""Compute Mini-FID (5k samples, Clean-FID vs ImageNet val 50k) for every saved
EMA checkpoint of a single run. Appends a row per ckpt to <run>/metrics/validation/fid.csv.

Usage (inside an `srun --gres=gpu:1`):
    uv run python scripts/post_hoc_fid.py <run_dir> <adapter_name> [--n_samples 5000] [--cfg 4.0]

Requires:
  * cleanfid Inception cache: ~/.cache/cleanfid_models/inception-2015-12-05.pt
  * Reference stats already built: cleanfid/stats/imagenet_val_50k_clean_*.npz
  * stats.json in <run>/../latents/<adapter>/stats.json (for latent denormalize)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Symlink Inception into /tmp before any cleanfid import (cleanfid hardcodes /tmp).
_INC_HOME = Path.home() / ".cache" / "cleanfid_models" / "inception-2015-12-05.pt"
_INC_TMP = Path("/tmp/inception-2015-12-05.pt")
if _INC_HOME.exists() and not _INC_TMP.exists():
    try:
        _INC_TMP.symlink_to(_INC_HOME)
    except FileExistsError:
        pass

# HF cache offline for the VAE adapter.
_FAST = "/leonardo_scratch/fast/IscrC_YENDRI"
os.environ.setdefault("HF_HOME", f"{_FAST}/lcerovaz/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", f"{_FAST}/lcerovaz/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_FAST}/lcerovaz/hf_cache/transformers")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from diffmechint.sit import SiT_models, create_transport  # noqa: E402
from diffmechint.sit.transport import Sampler  # noqa: E402
from diffmechint.tokenizers import build  # noqa: E402
from diffmechint.utils import error, info, ok, warn  # noqa: E402

LATENTS_BASE = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents")
REF_NAME = "imagenet_val_50k"
NUM_CLASSES = 1000


def _load_stats(adapter_name: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    stats_path = LATENTS_BASE / adapter_name / "stats.json"
    s = json.loads(stats_path.read_text())
    mean = torch.from_numpy(np.asarray(s["per_feature_mean"], dtype=np.float32)).view(1, -1, 1, 1)
    std = torch.from_numpy(np.asarray(s["per_feature_std"], dtype=np.float32)).view(1, -1, 1, 1)
    return mean, std, s


def _list_ema_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in sorted((run_dir / "checkpoints").glob("step_*_ema.safetensors")):
        step = int(p.name.split("_")[1])
        out.append((step, p))
    return out


def _build_model(model_name: str, in_channels: int, input_size: int, device: torch.device) -> torch.nn.Module:
    model = SiT_models[model_name](
        input_size=input_size,
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ).to(device).eval()
    return model


@torch.no_grad()
def _sample_ckpt(
    model: torch.nn.Module,
    transport,
    adapter,
    mean_d: torch.Tensor,
    std_d: torch.Tensor,
    n_samples: int,
    batch_size: int,
    cfg_scale: float,
    sample_steps: int,
    out_dir: Path,
    seed: int,
    device: torch.device,
    input_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method="dopri5", num_steps=sample_steps)

    in_ch = int(mean_d.shape[1])
    H = W = int(input_size)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_classes = torch.randint(0, NUM_CLASSES, (n_samples,), generator=gen)

    n_done = 0
    while n_done < n_samples:
        bsz = min(batch_size, n_samples - n_done)
        noise = torch.randn(bsz, in_ch, H, W, device=device)
        labels = all_classes[n_done : n_done + bsz].to(device)
        null_label = torch.full_like(labels, NUM_CLASSES)
        noise_full = torch.cat([noise, noise], dim=0)
        labels_full = torch.cat([labels, null_label], dim=0)

        def model_fn(x, t, y, cfg=cfg_scale):
            out = model.forward_with_cfg(x, t, y, cfg)
            return out[:, :in_ch]

        samples = sample_fn(noise_full, model_fn, y=labels_full)[-1][:bsz]
        samples = samples / std_d + mean_d
        imgs = adapter.decode(samples).clamp_(-1, 1).add(1).div(2).clamp_(0, 1)
        for k, img in enumerate(imgs):
            save_image(img, out_dir / f"img_{n_done + k:06d}.png")
        n_done += bsz


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("adapter", type=str)
    p.add_argument("--model_name", type=str, default="SiT-B/2",
                   help="SiT variant: 'SiT-B/2' (default) or 'SiT-B/1' for DC-AE patch=1")
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available — this will be very slow")

    from cleanfid import fid

    if not fid.test_stats_exists(REF_NAME, mode="clean"):
        error(f"Reference stats '{REF_NAME}' not found in cleanfid cache. Run prefetch_cleanfid.sh first.")
        return 1

    mean_d, std_d, stats = _load_stats(args.adapter)
    mean_d = mean_d.to(device)
    std_d = std_d.to(device)
    info(f"Adapter={args.adapter} latent_shape={stats['latent_shape']} feature_dim={stats['feature_dim']}")

    in_channels = stats["feature_dim"]
    input_size = stats["input_size"]

    info("Loading frozen VAE adapter for decode")
    adapter = build(args.adapter)
    adapter.load()
    adapter.to(device)

    transport = create_transport(path_type="Linear", prediction="velocity", loss_weight=None)
    info(f"Building model {args.model_name} (in_ch={in_channels}, input_size={input_size})")
    model = _build_model(args.model_name, in_channels, input_size, device)

    fid_csv = args.run_dir / "metrics" / "validation" / "fid.csv"
    fid_csv.parent.mkdir(parents=True, exist_ok=True)
    if not fid_csv.exists():
        fid_csv.write_text("step,n_samples,cfg_scale,fid\n")

    # Load already-computed steps so we can resume.
    done_steps: set[int] = set()
    with fid_csv.open() as f:
        next(f)  # header
        for row in f:
            parts = row.strip().split(",")
            if len(parts) >= 1 and parts[0].isdigit():
                done_steps.add(int(parts[0]))

    ckpts = _list_ema_checkpoints(args.run_dir)
    info(f"Found {len(ckpts)} EMA checkpoints; {len(done_steps)} already computed.")
    tmp_root = args.run_dir / "fid_post_hoc_tmp"
    tmp_root.mkdir(exist_ok=True)

    for step, ema_path in ckpts:
        if step in done_steps:
            info(f"  [skip] step {step} already in fid.csv")
            continue
        t0 = time.perf_counter()
        info(f"--- step {step}: loading EMA from {ema_path.name} ---")
        sd = load_file(str(ema_path))
        # Strip Lightning EMA shadow prefix if present.
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            warn(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

        out_imgs = tmp_root / f"step_{step:08d}"
        if out_imgs.exists():
            shutil.rmtree(out_imgs)
        info(f"  generating {args.n_samples} samples (cfg={args.cfg}, ode_steps={args.sample_steps})…")
        _sample_ckpt(
            model, transport, adapter, mean_d, std_d,
            n_samples=args.n_samples, batch_size=args.batch_size,
            cfg_scale=args.cfg, sample_steps=args.sample_steps,
            out_dir=out_imgs, seed=args.seed + step, device=device,
            input_size=input_size,
        )
        info(f"  scoring with Clean-FID against {REF_NAME}…")
        score = fid.compute_fid(
            str(out_imgs),
            dataset_name=REF_NAME,
            dataset_split="custom",
            mode="clean",
        )
        dt = time.perf_counter() - t0
        ok(f"  step={step}: FID-{args.n_samples//1000}k = {score:.3f}  ({dt/60:.1f} min)")

        with fid_csv.open("a") as f:
            csv.writer(f).writerow([step, args.n_samples, args.cfg, f"{score:.6f}"])

        shutil.rmtree(out_imgs, ignore_errors=True)

    shutil.rmtree(tmp_root, ignore_errors=True)
    ok(f"Done. FID curve in {fid_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
