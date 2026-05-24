"""Sampling-time feature-level cross-tokenizer SAE patching."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffmechint.hooks.timestep_router import current_t, timestep_context  # noqa: E402
from diffmechint.utils import error, ok, warn  # noqa: E402
from scripts.analysis.sae_feature_patching import (  # noqa: E402
    _load_matryoshka_sae,
    _resolve_sae_ckpt,
)
from scripts.analysis.tokenizer_dictionary_validation import (  # noqa: E402
    ACTIVATIONS_YNULL,
    SAE_ROOT,
    T_CENTERS,
    _aligned_positions,
    _cell_path,
    _read_rows,
    fit_ridge_affine,
)
from scripts.eval.cross_tokenizer_sae_substitution_fid import (  # noqa: E402
    _build_model,
    _load_stats,
)

LATENTS_BASE = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents")
REF_NAME = "imagenet_val_50k"
NUM_CLASSES = 1000


class TorchAffine:
    def __init__(self, weight: np.ndarray, bias: np.ndarray, device: torch.device, dtype: torch.dtype) -> None:
        self.weight = torch.from_numpy(weight).to(device=device, dtype=dtype)
        self.bias = torch.from_numpy(bias).to(device=device, dtype=dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


def _sample_tokens(acts: np.ndarray, max_tokens: int, seed: int) -> np.ndarray:
    flat = acts.reshape(-1, acts.shape[-1]).astype(np.float32)
    if flat.shape[0] <= max_tokens:
        return flat
    rng = np.random.default_rng(seed)
    return flat[rng.choice(flat.shape[0], size=max_tokens, replace=False)]


def fit_target_to_source_map(
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
):
    src_pos, tgt_pos, _ = _aligned_positions(activations_root, source, target, dit_step, max_images, seed)
    src = _sample_tokens(_read_rows(_cell_path(activations_root, source, dit_step, layer, t_bin), src_pos), max_tokens, seed + 2)
    tgt = _sample_tokens(_read_rows(_cell_path(activations_root, target, dit_step, layer, t_bin), tgt_pos), max_tokens, seed + 2)
    return fit_ridge_affine(tgt, src, alpha=ridge_alpha)


def make_feature_patch_hook(
    target_sae: torch.nn.Module,
    source_sae: torch.nn.Module | None,
    target_to_source,
    *,
    mode: str,
    target_feature_id: int,
    source_feature_id: int | None,
    random_source_feature_id: int | None,
    source_to_target_scale: float,
    t_center: float,
    t_tol: float,
    stats: dict[str, int],
    source_to_target_bias: float = 0.0,
    target_clamp_value: float | None = None,
    clamp_quantile: float = 0.99,
):
    sae_dtype = next(target_sae.parameters()).dtype

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
            z_tgt = target_sae.encode(flat)
            native = target_sae.decode(z_tgt)
            err = flat - native
            if mode in {"zero_target", "native_ablate"}:
                z_tgt[:, target_feature_id] = 0
            elif mode == "native_clamp":
                if target_clamp_value is not None:
                    clamp = z_tgt.new_tensor(float(target_clamp_value))
                else:
                    vals = z_tgt[:, target_feature_id]
                    active_vals = vals[vals > 0]
                    clamp = (
                        torch.quantile(active_vals.float(), clamp_quantile).to(dtype=vals.dtype)
                        if active_vals.numel() > 0
                        else vals.max()
                    )
                z_tgt[:, target_feature_id] = clamp
            elif mode in {"matched_patch", "transfer_replace", "random_patch", "random_matched_control", "wrong_window_control"}:
                if float(source_to_target_scale) == 0.0:
                    z_tgt[:, target_feature_id] = source_to_target_bias
                    stats["active"] += 1
                    return (target_sae.decode(z_tgt) + err).reshape(shape).to(dtype=orig_dtype)
                if source_sae is None or target_to_source is None:
                    raise RuntimeError("source_sae and target_to_source are required for source patch modes")
                src_flat = target_to_source(flat)
                z_src = source_sae.encode(src_flat)
                fid = source_feature_id if mode in {"matched_patch", "transfer_replace", "wrong_window_control"} else random_source_feature_id
                if fid is None:
                    raise RuntimeError("source feature id is required")
                z_tgt[:, target_feature_id] = z_src[:, int(fid)] * source_to_target_scale + source_to_target_bias
            else:
                raise ValueError(f"unknown patch mode {mode!r}")
            recon = (target_sae.decode(z_tgt) + err).reshape(shape).to(dtype=orig_dtype)
        stats["active"] += 1
        return recon

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
    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_classes = torch.randint(0, NUM_CLASSES, (n_samples,), generator=gen)
    in_ch = int(mean_d.shape[1])
    stats = {"active": 0, "skipped": 0, "no_t": 0}
    try:
        n_done = 0
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
    p.add_argument(
        "--mode",
        choices=[
            "baseline",
            "matched_patch",
            "random_patch",
            "zero_target",
            "native_ablate",
            "native_clamp",
            "transfer_replace",
            "random_matched_control",
            "wrong_window_control",
        ],
        required=True,
    )
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--t_bin", type=int, choices=[0, 1, 2], default=None)
    p.add_argument("--source_feature_id", type=int, default=None)
    p.add_argument("--target_feature_id", type=int, default=None)
    p.add_argument("--random_source_feature_id", type=int, default=None)
    p.add_argument("--source_mean_act", type=float, default=1.0)
    p.add_argument("--target_mean_act", type=float, default=1.0)
    p.add_argument("--source_to_target_scale", type=float, default=None)
    p.add_argument("--source_to_target_bias", type=float, default=0.0)
    p.add_argument("--target_clamp_value", type=float, default=None)
    p.add_argument("--clamp_quantile", type=float, default=0.99)
    p.add_argument("--wrong_t_bin", type=int, choices=[0, 1, 2], default=None)
    p.add_argument("--dit_step", type=int, default=200_000)
    p.add_argument("--sae_root", type=Path, default=SAE_ROOT)
    p.add_argument("--activations_root", type=Path, default=ACTIVATIONS_YNULL)
    p.add_argument("--out_root", type=Path, default=Path("outputs/phase4_11_feature_activation_patching/sampling"))
    p.add_argument("--model_name", type=str, default="SiT-B/2")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--sampler", choices=["sde", "ode"], default="ode")
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--sample_steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t_tol", type=float, default=0.01)
    p.add_argument("--fit_max_images", type=int, default=512)
    p.add_argument("--fit_max_tokens", type=int, default=50_000)
    p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--keep_images", action="store_true")
    p.add_argument(
        "--slope_gate_epsilon",
        type=float,
        default=0.01,
        help=(
            "For mode=transfer_replace: if |source_to_target_scale| < epsilon the "
            "calibrated transfer collapses to a bias-only intervention; we write a "
            "bias_only_skip record and skip the expensive sampling+FID. Set <= 0 to disable."
        ),
    )
    args = p.parse_args()

    if args.mode != "baseline":
        missing = [name for name in ("layer", "t_bin", "target_feature_id") if getattr(args, name) is None]
        if missing:
            error(f"missing required args for {args.mode}: {missing}")
            return 1
    source_modes = {"matched_patch", "random_patch", "transfer_replace", "random_matched_control", "wrong_window_control"}
    if args.mode in source_modes and args.source_condition is None:
        error("--source_condition is required for source patch modes")
        return 1
    if args.mode in {"matched_patch", "transfer_replace", "wrong_window_control"} and args.source_feature_id is None:
        error("--source_feature_id is required for transfer patch modes")
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
    if args.target_feature_id is not None:
        tag_bits.append(f"tf{args.target_feature_id}")
    if args.source_feature_id is not None:
        tag_bits.append(f"sf{args.source_feature_id}")
    tag_bits.append(f"step_{args.dit_step:06d}")
    run_tag = "__".join(tag_bits)
    out_dir = args.out_root / run_tag
    if out_dir.exists() and (out_dir / "fid.json").exists():
        raise FileExistsError(f"existing completed run dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "samples"
    if images_dir.exists():
        shutil.rmtree(images_dir)

    mean_d, std_d, stats = _load_stats(args.target_adapter)
    mean_d = mean_d.to(device)
    std_d = std_d.to(device)
    in_channels = int(stats["feature_dim"])
    input_size = int(stats["input_size"])

    from diffmechint.tokenizers import build

    adapter = build(args.target_adapter)
    adapter.load()
    adapter.to(device)
    model = _build_model(args.model_name, in_channels, input_size, device)
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
    if args.mode != "baseline":
        target_sae = _load_matryoshka_sae(_resolve_sae_ckpt(args.sae_root, args.target_condition, args.layer, args.t_bin, args.dit_step), device)
        source_sae = None
        target_to_source = None
        explicit_zero_scale = args.source_to_target_scale is not None and float(args.source_to_target_scale) == 0.0
        if args.mode in source_modes and not explicit_zero_scale:
            source_sae = _load_matryoshka_sae(_resolve_sae_ckpt(args.sae_root, args.source_condition, args.layer, args.t_bin, args.dit_step), device)
            affine = fit_target_to_source_map(
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
            target_to_source = TorchAffine(affine.weight, affine.bias, device, next(target_sae.parameters()).dtype)
        random_fid = args.random_source_feature_id
        if args.mode in {"random_patch", "random_matched_control"} and random_fid is None:
            random_fid = int(np.random.default_rng(args.seed).integers(0, int(source_sae.cfg.d_sae)))
        if args.source_to_target_scale is not None:
            scale = float(args.source_to_target_scale)
        elif args.source_mean_act <= 0:
            scale = 1.0
        else:
            scale = float(args.target_mean_act) / max(float(args.source_mean_act), 1e-8)
        if (
            args.mode == "transfer_replace"
            and args.slope_gate_epsilon > 0
            and abs(scale) < args.slope_gate_epsilon
        ):
            warn(
                f"bias_only_skip: |scale|={abs(scale):.3e} < "
                f"slope_gate_epsilon={args.slope_gate_epsilon:.3e}; "
                f"transfer_replace would collapse to bias-only intervention."
            )
            skip_result = {
                "run_tag": run_tag,
                "mode": "bias_only_skip",
                "intended_mode": args.mode,
                "source_condition": args.source_condition,
                "target_condition": args.target_condition,
                "target_adapter": args.target_adapter,
                "layer": args.layer,
                "t_bin": args.t_bin,
                "source_feature_id": args.source_feature_id,
                "target_feature_id": args.target_feature_id,
                "source_to_target_scale": scale,
                "source_to_target_bias": args.source_to_target_bias,
                "slope_gate_epsilon": args.slope_gate_epsilon,
                "dit_step": args.dit_step,
                "n_samples": 0,
                "fid": None,
                "hook_stats": {"active": 0, "skipped": 0, "no_t": 0},
                "skip_reason": "calibration_slope_below_gate_epsilon",
            }
            (out_dir / "fid.json").write_text(json.dumps(skip_result, indent=2), encoding="utf-8")
            csv_path = args.out_root / "sampling_feature_patching.csv"
            flat = {k: json.dumps(v) if isinstance(v, dict) else v for k, v in skip_result.items()}
            with csv_path.open("a", newline="", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                writer = csv.DictWriter(fh, fieldnames=list(skip_result.keys()))
                if csv_path.stat().st_size == 0:
                    writer.writeheader()
                writer.writerow(flat)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return 0
        t_center = T_CENTERS[int(args.t_bin)]
        if args.mode == "wrong_window_control":
            wrong_t_bin = args.wrong_t_bin if args.wrong_t_bin is not None else (int(args.t_bin) + 1) % 3
            t_center = T_CENTERS[int(wrong_t_bin)]
        hook_fn = _attach_stats(
            make_feature_patch_hook(
                target_sae,
                source_sae,
                target_to_source,
                mode=args.mode,
                target_feature_id=args.target_feature_id,
                source_feature_id=args.source_feature_id,
                random_source_feature_id=random_fid,
                source_to_target_scale=scale,
                t_center=t_center,
                t_tol=args.t_tol,
                stats=hook_stats,
                source_to_target_bias=args.source_to_target_bias,
                target_clamp_value=args.target_clamp_value,
                clamp_quantile=args.clamp_quantile,
            ),
            hook_stats,
        )

    from diffmechint.sit import create_transport

    transport = create_transport(path_type="Linear", prediction="velocity", loss_weight=None)
    t0 = time.perf_counter()
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
        warn("feature patch hook never fired")

    t1 = time.perf_counter()
    fid_score = cleanfid_mod.compute_fid(str(images_dir), dataset_name=REF_NAME, dataset_split="custom", mode="clean")
    fid_sec = time.perf_counter() - t1
    result = {
        "run_tag": run_tag,
        "mode": args.mode,
        "source_condition": args.source_condition,
        "target_condition": args.target_condition,
        "target_adapter": args.target_adapter,
        "layer": args.layer,
        "t_bin": args.t_bin,
        "source_feature_id": args.source_feature_id,
        "target_feature_id": args.target_feature_id,
        "random_source_feature_id": args.random_source_feature_id,
        "source_to_target_bias": args.source_to_target_bias,
        "target_clamp_value": args.target_clamp_value,
        "wrong_t_bin": args.wrong_t_bin,
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
    }
    (out_dir / "fid.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = args.out_root / "sampling_feature_patching.csv"
    flat = {k: json.dumps(v) if isinstance(v, dict) else v for k, v in result.items()}
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
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
