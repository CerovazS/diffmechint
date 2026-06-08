"""Phase 4.10 sampling-time cross-tokenizer SAE substitution FID."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

_FAST = "/leonardo_scratch/fast/IscrC_YENDRI"
os.environ.setdefault("HF_HOME", f"{_FAST}/lcerovaz/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", f"{_FAST}/lcerovaz/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_FAST}/lcerovaz/hf_cache/transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_INC_HOME = Path.home() / ".cache" / "cleanfid_models" / "inception-2015-12-05.pt"
_INC_TMP = Path("/tmp/inception-2015-12-05.pt")
if _INC_HOME.exists() and not _INC_TMP.exists():
    try:
        _INC_TMP.symlink_to(_INC_HOME)
    except FileExistsError:
        pass

from diffmechint.analysis.alignment import (  # noqa: E402
    ACTIVATIONS_YNULL,
    SAE_ROOT,
    T_CENTERS,
    AffineMap,
    aligned_positions,
    cell_path,
    fit_ridge_affine,
    read_rows,
)
from diffmechint.hooks.timestep_router import current_t, timestep_context  # noqa: E402
from diffmechint.sae import load_matryoshka_sae, resolve_sae_ckpt  # noqa: E402
from diffmechint.sit import build_sit_model  # noqa: E402
from diffmechint.tokenizers import load_latent_stats  # noqa: E402
from diffmechint.utils import error, info, ok, warn  # noqa: E402

LATENTS_BASE = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents")
REF_NAME = "imagenet_val_50k"
NUM_CLASSES = 1000


class TorchAffine:
    """Torch row-vector affine map created from an analysis-space AffineMap."""

    def __init__(self, affine: AffineMap, device: torch.device, dtype: torch.dtype) -> None:
        self.weight = torch.from_numpy(affine.weight).to(device=device, dtype=dtype)
        self.bias = torch.from_numpy(affine.bias).to(device=device, dtype=dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


def _sample_tokens(acts: np.ndarray, max_tokens: int, seed: int) -> np.ndarray:
    flat = acts.reshape(-1, acts.shape[-1])
    if flat.shape[0] <= max_tokens:
        return flat
    rng = np.random.default_rng(seed)
    return flat[rng.choice(flat.shape[0], size=max_tokens, replace=False)]


def _fit_bidirectional_maps(
    activations_root: Path,
    source: str,
    target: str,
    dit_step: int,
    layer: int,
    t_bin: int,
    max_images: int,
    max_tokens: int,
    ridge_alpha: float,
    seed: int,
) -> tuple[AffineMap, AffineMap]:
    src_pos, tgt_pos, _ = aligned_positions(
        activations_root, source, target, dit_step, max_images, seed
    )
    src = _sample_tokens(
        read_rows(cell_path(activations_root, source, dit_step, layer, t_bin), src_pos),
        max_tokens,
        seed + 2,
    )
    tgt = _sample_tokens(
        read_rows(cell_path(activations_root, target, dit_step, layer, t_bin), tgt_pos),
        max_tokens,
        seed + 2,
    )
    return (
        fit_ridge_affine(tgt, src, alpha=ridge_alpha),
        fit_ridge_affine(src, tgt, alpha=ridge_alpha),
    )


def make_native_sae_hook(
    sae: torch.nn.Module,
    t_center: float,
    t_tol: float,
    stats: dict[str, int],
):
    """Return a hook that substitutes target residuals with target SAE reconstructions."""
    sae_dtype = next(sae.parameters()).dtype

    def hook(module, inputs, output):
        t = current_t()
        if t is None:
            stats["no_t"] += 1
            return None
        if abs(float(t) - t_center) > t_tol:
            stats["skipped"] += 1
            return None
        orig_dtype = output.dtype
        flat = output.to(dtype=sae_dtype).reshape(-1, output.shape[-1])
        with torch.no_grad():
            recon = sae(flat).reshape(output.shape).to(dtype=orig_dtype)
        stats["active"] += 1
        return recon

    return hook


def make_transferred_sae_hook(
    source_sae: torch.nn.Module,
    target_to_source,
    source_to_target,
    t_center: float,
    t_tol: float,
    stats: dict[str, int],
):
    """Return a hook that applies target→source map, source SAE, then source→target map."""
    sae_dtype = next(source_sae.parameters()).dtype

    def hook(module, inputs, output):
        t = current_t()
        if t is None:
            stats["no_t"] += 1
            return None
        if abs(float(t) - t_center) > t_tol:
            stats["skipped"] += 1
            return None
        orig_dtype = output.dtype
        shape = output.shape
        flat = output.to(dtype=sae_dtype).reshape(-1, shape[-1])
        with torch.no_grad():
            source_space = target_to_source(flat)
            source_recon = source_sae(source_space)
            target_recon = source_to_target(source_recon)
        stats["active"] += 1
        return target_recon.reshape(shape).to(dtype=orig_dtype)

    return hook


@torch.no_grad()
def _sample_with_hook(
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
    sampler_kind: str,
    denormalize: bool,
    hook_layer: int | None,
    hook_fn,
) -> dict[str, int]:
    from torchvision.utils import save_image

    from diffmechint.sit.transport import Sampler

    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(transport)
    sample_fn = (
        sampler.sample_sde(sampling_method="Euler", diffusion_form="SBDM", num_steps=sample_steps)
        if sampler_kind == "sde"
        else sampler.sample_ode(sampling_method="dopri5", num_steps=sample_steps)
    )
    handle = None
    if hook_layer is not None and hook_fn is not None:
        handle = model.blocks[hook_layer].register_forward_hook(hook_fn)
    stats = {"active": 0, "skipped": 0, "no_t": 0}
    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_classes = torch.randint(0, NUM_CLASSES, (n_samples,), generator=gen)
    in_ch = int(mean_d.shape[1])
    n_done = 0
    try:
        while n_done < n_samples:
            bsz = min(batch_size, n_samples - n_done)
            noise = torch.randn(bsz, in_ch, input_size, input_size, device=device)
            labels = all_classes[n_done : n_done + bsz].to(device)
            null_label = torch.full_like(labels, NUM_CLASSES)
            noise_full = torch.cat([noise, noise], dim=0)
            labels_full = torch.cat([labels, null_label], dim=0)

            def model_fn(x, t, y, cfg=cfg_scale):
                with timestep_context(float(t.detach().flatten()[0].item())):
                    out = model.forward_with_cfg(x, t, y, cfg)
                return out[:, :in_ch]

            samples = sample_fn(noise_full, model_fn, y=labels_full)[-1][:bsz]
            if denormalize:
                samples = samples / std_d + mean_d
            imgs = adapter.decode(samples).clamp_(-1, 1).add(1).div(2).clamp_(0, 1)
            for idx, img in enumerate(imgs):
                save_image(img, out_dir / f"img_{n_done + idx:06d}.png")
            n_done += bsz
    finally:
        if handle is not None:
            handle.remove()
    if hook_fn is not None and hasattr(hook_fn, "_stats_ref"):
        stats = hook_fn._stats_ref
    return stats


def _attach_stats(hook_fn, stats: dict[str, int]):
    hook_fn._stats_ref = stats
    return hook_fn


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target_run", type=Path, required=True)
    p.add_argument("--target_adapter", type=str, required=True)
    p.add_argument("--target_condition", type=str, required=True)
    p.add_argument("--source_condition", type=str, default=None)
    p.add_argument("--mode", choices=["baseline", "native", "transfer"], required=True)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--t_bin", type=int, choices=[0, 1, 2], default=None)
    p.add_argument("--dit_step", type=int, default=200_000)
    p.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    p.add_argument("--activations_root", type=Path, default=ACTIVATIONS_YNULL)
    p.add_argument("--out_root", type=Path, default=Path("outputs/phase4_10_tokenizer_dictionary_validation/fid"))
    p.add_argument("--model_name", type=str, default="SiT-B/2")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--sampler", type=str, default="ode", choices=["sde", "ode"])
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--sample_steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t_tol", type=float, default=0.01)
    p.add_argument("--fit_max_images", type=int, default=512)
    p.add_argument("--fit_max_tokens", type=int, default=50_000)
    p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--keep_images", action="store_true")
    args = p.parse_args()

    if args.mode != "baseline" and (args.layer is None or args.t_bin is None):
        error("--layer and --t_bin are required for native/transfer modes")
        return 1
    if args.mode == "transfer" and args.source_condition is None:
        error("--source_condition is required for transfer mode")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available; sampling will be slow.")

    from cleanfid import fid as cleanfid_mod

    if not cleanfid_mod.test_stats_exists(REF_NAME, mode="clean"):
        error(f"Reference stats '{REF_NAME}' not found in cleanfid cache.")
        return 1

    tag_bits = [args.target_condition, args.mode]
    if args.source_condition:
        tag_bits.append(f"src-{args.source_condition}")
    if args.layer is not None:
        tag_bits.append(f"L{args.layer}_T{args.t_bin}")
    tag_bits.append(f"step_{args.dit_step:06d}")
    run_tag = "__".join(tag_bits)
    out_dir = args.out_root / run_tag
    if out_dir.exists() and (out_dir / "fid.json").exists():
        raise FileExistsError(f"existing completed run dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "samples"
    if images_dir.exists():
        shutil.rmtree(images_dir)

    mean_d, std_d, stats = load_latent_stats(args.target_adapter)
    mean_d = mean_d.to(device)
    std_d = std_d.to(device)
    in_channels = int(stats["feature_dim"])
    input_size = int(stats["input_size"])

    info(f"Loading target adapter {args.target_adapter}")
    from diffmechint.tokenizers import build

    adapter = build(args.target_adapter)
    adapter.load()
    adapter.to(device)

    model = build_sit_model(args.model_name, in_channels, input_size, device)
    ema_path = args.target_run / "checkpoints" / f"step_{args.dit_step:08d}_ema.safetensors"
    if not ema_path.exists():
        error(f"SiT EMA checkpoint missing: {ema_path}")
        return 1
    from safetensors.torch import load_file

    state = {k.removeprefix("module."): v for k, v in load_file(str(ema_path)).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        warn(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    hook_fn = None
    hook_stats = {"active": 0, "skipped": 0, "no_t": 0}
    if args.mode == "native":
        ckpt = resolve_sae_ckpt(
            args.sae_root, args.target_condition, args.layer, args.t_bin, args.dit_step
        )
        info(f"Loading native target SAE {ckpt}")
        sae = load_matryoshka_sae(ckpt, device)
        hook_fn = _attach_stats(
            make_native_sae_hook(sae, T_CENTERS[args.t_bin], args.t_tol, hook_stats),
            hook_stats,
        )
    elif args.mode == "transfer":
        ckpt = resolve_sae_ckpt(
            args.sae_root, args.source_condition, args.layer, args.t_bin, args.dit_step
        )
        info(f"Loading source SAE {ckpt}")
        sae = load_matryoshka_sae(ckpt, device)
        info("Fitting target↔source ridge maps from y-null activations")
        tgt_to_src_np, src_to_tgt_np = _fit_bidirectional_maps(
            args.activations_root,
            args.source_condition,
            args.target_condition,
            args.dit_step,
            args.layer,
            args.t_bin,
            args.fit_max_images,
            args.fit_max_tokens,
            args.ridge_alpha,
            args.seed,
        )
        sae_dtype = next(sae.parameters()).dtype
        tgt_to_src = TorchAffine(tgt_to_src_np, device, sae_dtype)
        src_to_tgt = TorchAffine(src_to_tgt_np, device, sae_dtype)
        hook_fn = _attach_stats(
            make_transferred_sae_hook(
                sae,
                tgt_to_src,
                src_to_tgt,
                T_CENTERS[args.t_bin],
                args.t_tol,
                hook_stats,
            ),
            hook_stats,
        )

    from diffmechint.sit import create_transport

    transport = create_transport(path_type="Linear", prediction="velocity", loss_weight=None)
    t0 = time.perf_counter()
    info(f"Generating {args.n_samples} images for {run_tag}")
    hook_result = _sample_with_hook(
        model,
        transport,
        adapter,
        mean_d,
        std_d,
        args.n_samples,
        args.batch_size,
        args.cfg,
        args.sample_steps,
        images_dir,
        args.seed + args.dit_step,
        device,
        input_size,
        args.sampler,
        not args.no_normalize,
        args.layer if args.mode != "baseline" else None,
        hook_fn,
    )
    gen_sec = time.perf_counter() - t0
    if args.mode != "baseline" and hook_result["active"] == 0:
        warn("substitution hook never fired")

    info("Scoring with Clean-FID")
    t1 = time.perf_counter()
    fid_score = cleanfid_mod.compute_fid(
        str(images_dir),
        dataset_name=REF_NAME,
        dataset_split="custom",
        mode="clean",
    )
    fid_sec = time.perf_counter() - t1
    result = {
        "run_tag": run_tag,
        "mode": args.mode,
        "source_condition": args.source_condition,
        "target_condition": args.target_condition,
        "target_adapter": args.target_adapter,
        "layer": args.layer,
        "t_bin": args.t_bin,
        "t_bin_center": T_CENTERS.get(args.t_bin),
        "dit_step": args.dit_step,
        "n_samples": args.n_samples,
        "cfg_scale": args.cfg,
        "sampler": args.sampler,
        "sample_steps": args.sample_steps,
        "seed": args.seed,
        "no_normalize": args.no_normalize,
        "fid": float(fid_score),
        "hook_stats": hook_result,
        "gen_minutes": gen_sec / 60.0,
        "fid_minutes": fid_sec / 60.0,
        "ref_name": REF_NAME,
        "latents_base": str(LATENTS_BASE),
    }
    (out_dir / "fid.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_md = "\n".join(
        [
            f"# Cross-SAE FID - {run_tag}",
            "",
            "> [!summary] TL;DR",
            f"> **{args.mode}** sampling scored FID =={fid_score:.3f}== "
            f"on {args.n_samples} samples; compare against baseline and native target rows.",
            "",
            f"- Mode: `{args.mode}`",
            f"- Source condition: `{args.source_condition}`",
            f"- Target condition: `{args.target_condition}`",
            f"- Cell: `L{args.layer}/T{args.t_bin}`",
            f"- Hook stats: `{json.dumps(hook_result)}`",
            "",
        ]
    )
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    csv_path = args.out_root / "cross_sae_fid.csv"
    flat = {k: json.dumps(v) if isinstance(v, dict) else v for k, v in result.items()}
    with csv_path.open("a", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        writer = csv.DictWriter(fh, fieldnames=list(result.keys()))
        if csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(flat)
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    if not args.keep_images:
        shutil.rmtree(images_dir, ignore_errors=True)
    ok(f"Done {run_tag}: FID={fid_score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
