"""Post-hoc FID against tokenizer-specific validation-holdout reconstructions."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

_INC_HOME = Path.home() / ".cache" / "cleanfid_models" / "inception-2015-12-05.pt"
_INC_TMP = Path("/tmp/inception-2015-12-05.pt")
if _INC_HOME.exists() and not _INC_TMP.exists():
    try:
        _INC_TMP.symlink_to(_INC_HOME)
    except FileExistsError:
        pass

_FAST = "/leonardo_scratch/fast/IscrC_YENDRI"
os.environ.setdefault("HF_HOME", f"{_FAST}/lcerovaz/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", f"{_FAST}/lcerovaz/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_FAST}/lcerovaz/hf_cache/transformers")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from diffmechint.sit import build_sit_model, create_transport, list_ema_checkpoints  # noqa: E402
from diffmechint.sit.transport import Sampler  # noqa: E402
from diffmechint.tokenizers import build, load_latent_stats  # noqa: E402
from diffmechint.training.data import CachedLatentDataset  # noqa: E402
from diffmechint.utils import error, info, ok, warn  # noqa: E402

LATENTS_BASE = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents")
DEFAULT_OUT_ROOT = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/by_model/"
    "sit_l_2/analysis/holdout_recon_fid"
)
NUM_CLASSES = 1000


def _count_pngs(path: Path) -> int:
    return sum(1 for _ in path.glob("*.png")) if path.exists() else 0


@torch.no_grad()
def _ensure_reference_images(
    *,
    adapter,
    adapter_name: str,
    shard_dir: Path,
    reference_dir: Path,
    n_ref: int,
    batch_size: int,
    holdout_fraction: float,
    holdout_seed: int,
    reference_seed: int,
    device: torch.device,
    rebuild: bool,
) -> int:
    manifest_path = reference_dir / "manifest.json"
    if not rebuild and manifest_path.exists() and _count_pngs(reference_dir) >= n_ref:
        manifest = json.loads(manifest_path.read_text())
        ok(f"Reference images already present: {reference_dir} ({manifest['n_images']} images)")
        return int(manifest["n_images"])

    if reference_dir.exists():
        shutil.rmtree(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)

    dataset = CachedLatentDataset(
        shard_dir=shard_dir,
        normalize=False,
        holdout_fraction=holdout_fraction,
        holdout_seed=holdout_seed,
        is_val=True,
    )
    if len(dataset) == 0:
        raise ValueError("validation holdout is empty")
    actual_n = min(n_ref, len(dataset))
    rng = np.random.default_rng(reference_seed)
    indices = np.arange(len(dataset))
    if actual_n < len(dataset):
        indices = rng.choice(indices, size=actual_n, replace=False)
    indices = np.sort(indices)
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    info(
        f"Decoding {actual_n} {adapter_name} holdout latents to reference images "
        f"from {shard_dir}"
    )
    n_done = 0
    for batch in loader:
        latents = batch["latent"].to(device, non_blocking=True)
        images = adapter.decode(latents).clamp_(-1, 1).add(1).div(2).clamp_(0, 1)
        for img in images:
            save_image(img, reference_dir / f"img_{n_done:06d}.png")
            n_done += 1

    manifest = {
        "kind": "training_holdout_reconstruction_reference",
        "adapter": adapter_name,
        "shard_dir": str(shard_dir),
        "holdout_fraction": holdout_fraction,
        "holdout_seed": holdout_seed,
        "reference_seed": reference_seed,
        "n_images": n_done,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    ok(f"Wrote reference images: {reference_dir} ({n_done} images)")
    return n_done


def _ensure_reference_stats(reference_name: str, reference_dir: Path, rebuild: bool) -> None:
    from cleanfid import fid

    if not rebuild and fid.test_stats_exists(reference_name, mode="clean"):
        ok(f"Clean-FID stats already present: {reference_name}")
        return
    info(f"Building Clean-FID stats '{reference_name}' from {reference_dir}")
    fid.make_custom_stats(reference_name, str(reference_dir), mode="clean")
    if not fid.test_stats_exists(reference_name, mode="clean"):
        raise RuntimeError(f"failed to build Clean-FID stats: {reference_name}")
    ok(f"Clean-FID stats ready: {reference_name}")


@torch.no_grad()
def _sample_checkpoint(
    *,
    model: torch.nn.Module,
    transport,
    adapter,
    mean: torch.Tensor,
    std: torch.Tensor,
    out_dir: Path,
    n_samples: int,
    batch_size: int,
    cfg_scale: float,
    sample_steps: int,
    sample_seed: int,
    sampler_kind: str,
    denormalize: bool,
    device: torch.device,
    input_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(transport)
    if sampler_kind == "sde":
        sample_fn = sampler.sample_sde(
            sampling_method="Euler",
            diffusion_form="SBDM",
            num_steps=sample_steps,
        )
    else:
        sample_fn = sampler.sample_ode(sampling_method="dopri5", num_steps=sample_steps)

    in_channels = int(mean.shape[1])
    generator = torch.Generator(device="cpu").manual_seed(sample_seed)
    all_classes = torch.randint(0, NUM_CLASSES, (n_samples,), generator=generator)

    n_done = 0
    while n_done < n_samples:
        this_b = min(batch_size, n_samples - n_done)
        noise = torch.randn(this_b, in_channels, input_size, input_size, device=device)
        labels = all_classes[n_done : n_done + this_b].to(device)
        null_label = torch.full_like(labels, NUM_CLASSES)
        noise_full = torch.cat([noise, noise], dim=0)
        labels_full = torch.cat([labels, null_label], dim=0)

        def model_fn(x, t, y, cfg=cfg_scale):
            out = model.forward_with_cfg(x, t, y, cfg)
            return out[:, :in_channels]

        samples = sample_fn(noise_full, model_fn, y=labels_full)[-1][:this_b]
        if denormalize:
            samples = samples * std.to(device) + mean.to(device)
        images = adapter.decode(samples).clamp_(-1, 1).add(1).div(2).clamp_(0, 1)
        for img in images:
            save_image(img, out_dir / f"img_{n_done:06d}.png")
            n_done += 1


def _read_done_steps(
    csv_path: Path,
    *,
    n_samples: int,
    n_reference: int,
    cfg_scale: float,
    sample_steps: int,
    sampler: str,
    reference_name: str,
) -> set[int]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        done: set[int] = set()
        for row in reader:
            if not row.get("step", "").isdigit():
                continue
            if int(row.get("n_samples", -1)) != n_samples:
                continue
            if int(row.get("n_reference", -1)) != n_reference:
                continue
            if float(row.get("cfg_scale", "nan")) != cfg_scale:
                continue
            if int(row.get("sample_steps", -1)) != sample_steps:
                continue
            if row.get("sampler") != sampler:
                continue
            if row.get("reference_name") != reference_name:
                continue
            done.add(int(row["step"]))
        return done


def _parse_steps(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(part) for part in raw.replace(":", ",").split(",") if part.strip()}


def _append_metric_row(csv_path: Path, row: list[object]) -> None:
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(row)
        fcntl.flock(lock, fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("adapter", type=str)
    parser.add_argument("--model_name", type=str, default="SiT-L/2")
    parser.add_argument("--shard_dir", type=Path, default=None)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--n_ref", type=int, default=5000)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--reference_batch_size", type=int, default=64)
    parser.add_argument("--holdout_fraction", type=float, default=0.05)
    parser.add_argument("--holdout_seed", type=int, default=42)
    parser.add_argument("--reference_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--sample_steps", type=int, default=250)
    parser.add_argument("--sampler", type=str, default="ode", choices=["ode", "sde"])
    parser.add_argument("--steps", type=str, default=None, help="comma/colon-separated steps")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--rebuild_reference", action="store_true")
    parser.add_argument("--keep_images", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available; evaluation will be very slow")
    from cleanfid import fid

    if not _INC_TMP.exists():
        error(f"Clean-FID Inception weights missing at {_INC_TMP}")
        return 1

    shard_dir = args.shard_dir or (LATENTS_BASE / args.adapter)
    mean, std, stats = load_latent_stats(args.adapter)
    mean = mean.to(device)
    std = std.to(device)
    input_size = int(stats["input_size"])
    in_channels = int(stats["feature_dim"])

    info(f"Loading adapter {args.adapter}")
    adapter = build(args.adapter)
    adapter.load()
    adapter.to(device)

    ref_name = (
        f"{args.adapter}_train_holdout_recon_n{args.n_ref}"
        f"_hseed{args.holdout_seed}_rseed{args.reference_seed}"
    )
    ref_dir = args.out_root / "references" / ref_name
    actual_n_ref = _ensure_reference_images(
        adapter=adapter,
        adapter_name=args.adapter,
        shard_dir=shard_dir,
        reference_dir=ref_dir,
        n_ref=args.n_ref,
        batch_size=args.reference_batch_size,
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
        reference_seed=args.reference_seed,
        device=device,
        rebuild=args.rebuild_reference,
    )
    _ensure_reference_stats(ref_name, ref_dir, args.rebuild_reference)

    transport = create_transport(path_type="Linear", prediction="velocity", loss_weight=None)
    info(f"Building {args.model_name} (in_ch={in_channels}, input_size={input_size})")
    model = build_sit_model(args.model_name, in_channels, input_size, device)

    metrics_path = args.run_dir / "metrics" / "validation" / "holdout_recon_fid.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        metrics_path.write_text(
            "step,n_samples,n_reference,cfg_scale,sample_steps,sampler,"
            "reference_name,reference_dir,fid,seconds\n"
        )

    requested_steps = _parse_steps(args.steps)
    done_steps = _read_done_steps(
        metrics_path,
        n_samples=args.n_samples,
        n_reference=actual_n_ref,
        cfg_scale=args.cfg,
        sample_steps=args.sample_steps,
        sampler=args.sampler,
        reference_name=ref_name,
    )
    ckpts = list_ema_checkpoints(args.run_dir, requested_steps)
    if not ckpts:
        raise FileNotFoundError(f"no EMA checkpoints found in {args.run_dir / 'checkpoints'}")
    info(f"Found {len(ckpts)} EMA checkpoints; {len(done_steps)} already in {metrics_path.name}")

    tmp_root = args.out_root / "tmp" / args.run_dir.name
    tmp_root.mkdir(parents=True, exist_ok=True)
    for step, ema_path in ckpts:
        if step in done_steps:
            info(f"[skip] step {step} already evaluated")
            continue
        out_images = tmp_root / f"step_{step:08d}"
        if out_images.exists():
            shutil.rmtree(out_images)
        start = time.perf_counter()
        info(f"--- step {step}: loading {ema_path.name} ---")
        state = {k.removeprefix("module."): v for k, v in load_file(str(ema_path)).items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            warn(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        _sample_checkpoint(
            model=model,
            transport=transport,
            adapter=adapter,
            mean=mean,
            std=std,
            out_dir=out_images,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            cfg_scale=args.cfg,
            sample_steps=args.sample_steps,
            sample_seed=args.sample_seed + step,
            sampler_kind=args.sampler,
            denormalize=not args.no_normalize,
            device=device,
            input_size=input_size,
        )
        info(f"Scoring step {step} against {ref_name}")
        score = fid.compute_fid(
            str(out_images),
            dataset_name=ref_name,
            dataset_split="custom",
            mode="clean",
        )
        seconds = time.perf_counter() - start
        ok(f"step={step}: holdout-recon FID-{args.n_samples // 1000}k = {score:.3f}")
        _append_metric_row(
            metrics_path,
            [
                step,
                args.n_samples,
                actual_n_ref,
                args.cfg,
                args.sample_steps,
                args.sampler,
                ref_name,
                str(ref_dir),
                f"{score:.6f}",
                f"{seconds:.3f}",
            ],
        )
        if not args.keep_images:
            shutil.rmtree(out_images, ignore_errors=True)

    # Multiple per-checkpoint jobs can share tmp_root for the same run. Remove
    # only the step directory above; deleting tmp_root races sibling jobs.
    ok(f"Done. Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
