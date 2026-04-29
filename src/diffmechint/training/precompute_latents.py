"""Pre-encode an image dataset through one tokenizer; dump shards to HDF5.

Usage::

    uv run python -m diffmechint.training.precompute_latents \\
        tokenizer=eq_vae \\
        +data_dir=/path/to/imagenet256 \\
        +output_dir=$FAST/diffmechint/latents/eq_vae \\
        +shard_size=10000 +max_images=null

This avoids re-encoding inside the diffusion training loop, which is the dominant
I/O cost in matched-compute SiT training. See PLAN.md §5.4.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from diffmechint.tokenizers.base import TokenizerAdapter
from diffmechint.utils import error, info, ok, warn


def _build_loader(
    data_dir: str, image_size: int, batch_size: int, num_workers: int
) -> tuple[Dataset, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # → [-1, 1]
        ]
    )
    dataset = ImageFolder(data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader


def _open_shard(out_dir: Path, shard_idx: int, latent_shape: tuple[int, int, int]) -> h5py.File:
    path = out_dir / f"{shard_idx:05d}.h5"
    f = h5py.File(path, "w")
    f.create_dataset(
        "latents",
        shape=(0, *latent_shape),
        maxshape=(None, *latent_shape),
        dtype="float16",
        chunks=(1, *latent_shape),
        compression="lzf",
    )
    f.create_dataset("labels", shape=(0,), maxshape=(None,), dtype="int32")
    return f


def _stat_summary(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "count": int(values.size),
    }


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available — running on CPU. This is intended only for smoke tests.")

    # Adapter via Hydra _target_ instantiation.
    adapter: TokenizerAdapter = hydra.utils.instantiate(cfg.tokenizer)
    adapter.load()
    adapter.to(device)
    info(f"Loaded adapter: {adapter}")

    data_dir = cfg.get("data_dir")
    if data_dir is None:
        error("data_dir is required: pass `+data_dir=/path/to/imagenet256`.")
        raise SystemExit(2)

    out_dir = Path(cfg.get("output_dir", f"{cfg.output_root}/latents/{adapter.spec.name}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    info(f"Writing latents to {out_dir}")

    image_size = cfg.get("image_size", 256)
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)
    shard_size = cfg.get("shard_size", 10_000)
    max_images = cfg.get("max_images")

    dataset, loader = _build_loader(data_dir, image_size, batch_size, num_workers)
    info(f"Dataset: {len(dataset)} images; batch_size={batch_size}; shard_size={shard_size}")

    sample_buf: list[np.ndarray] = []
    written = 0
    shard_idx = 0
    shard = _open_shard(out_dir, shard_idx, adapter.spec.latent_shape)
    t0 = time.perf_counter()

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        z = adapter.encode(imgs).to(torch.float16).cpu().numpy()
        labels_np = labels.numpy().astype(np.int32)

        # Append to current shard, rolling over when full.
        n = z.shape[0]
        cur_size = shard["latents"].shape[0]
        shard["latents"].resize((cur_size + n, *adapter.spec.latent_shape))
        shard["latents"][cur_size : cur_size + n] = z
        shard["labels"].resize((cur_size + n,))
        shard["labels"][cur_size : cur_size + n] = labels_np
        written += n

        # Stats sample (cheap, used for run report).
        if len(sample_buf) < 8:
            sample_buf.append(z[:1].astype(np.float32))

        if (cur_size + n) >= shard_size:
            shard.close()
            shard_idx += 1
            shard = _open_shard(out_dir, shard_idx, adapter.spec.latent_shape)

        if max_images is not None and written >= max_images:
            break

        if batch_idx % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = written / max(elapsed, 1e-6)
            info(f"batch {batch_idx} | written {written} | {rate:.1f} img/s")

    shard.close()

    # Stats summary written next to the shards for quick eyeballing.
    if sample_buf:
        all_samples = np.concatenate(sample_buf, axis=0)
        stats = _stat_summary(all_samples)
        stats["adapter"] = adapter.spec.name
        stats["latent_shape"] = list(adapter.spec.latent_shape)
        stats["scaling_factor"] = adapter.spec.scaling_factor
        stats["images_written"] = written
        stats["wall_seconds"] = time.perf_counter() - t0
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    ok(f"Done — {written} images encoded in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
