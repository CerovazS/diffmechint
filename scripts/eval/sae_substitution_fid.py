"""Phase 4.5a — substitution-FID for one (condition, layer, t_bin) cell.

Generates `n_samples` images with a Matryoshka SAE inserted into the SiT's
residual stream at the chosen block, only on sampling steps whose diffusion
time falls within ±t_tol of the chosen t-bin centre, then scores them with
Clean-FID against ImageNet val 50k.

Designed to be launched once per cell (27 cells + 3 per-condition baselines).
Writes one JSON per run + a sample dir, both under
`<out_root>/<run_tag>/`. Aggregation across cells happens in a separate
plotting step.

Usage (inside an `srun --gres=gpu:1`):
    uv run python scripts/sae_substitution_fid.py \\
        --run <sit_run_dir> --adapter <name> \\
        --sae_root <matryoshka root> --condition <name> \\
        --layer 3 --t_bin 0 --dit_step 200000 \\
        --out_root outputs/phase4_5a_subst_fid \\
        [--no_substitute]  [--no_normalize]
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

# Symlink Inception into /tmp before any cleanfid import.
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
from torchvision.utils import save_image  # noqa: E402

from diffmechint.hooks.timestep_router import current_t, timestep_context  # noqa: E402
from diffmechint.sit import SiT_models, create_transport  # noqa: E402
from diffmechint.sit.transport import Sampler  # noqa: E402
from diffmechint.tokenizers import build  # noqa: E402
from diffmechint.utils import error, info, ok, warn  # noqa: E402

LATENTS_BASE = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents")
REF_NAME = "imagenet_val_50k"
NUM_CLASSES = 1000
T_BIN_CENTERS = (0.025, 0.20, 0.50)


def _load_stats(adapter_name: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    stats_path = LATENTS_BASE / adapter_name / "stats.json"
    s = json.loads(stats_path.read_text())
    mean = torch.from_numpy(np.asarray(s["per_feature_mean"], dtype=np.float32)).view(1, -1, 1, 1)
    std = torch.from_numpy(np.asarray(s["per_feature_std"], dtype=np.float32)).view(1, -1, 1, 1)
    return mean, std, s


def _build_model(model_name: str, in_channels: int, input_size: int, device: torch.device) -> torch.nn.Module:
    model = SiT_models[model_name](
        input_size=input_size,
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ).to(device).eval()
    return model


def _resolve_sae_ckpt(sae_root: Path, condition: str, layer: int,
                      t_bin: int, dit_step: int) -> Path:
    """Locate the production SAE final_* dir for one cell."""
    base = sae_root / condition / f"L{layer}_T{t_bin}" / f"step_{dit_step:06d}"
    if not base.exists():
        raise FileNotFoundError(f"SAE step dir missing: {base}")
    finals = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("final_"))
    if not finals:
        raise FileNotFoundError(f"no final_* under {base}")
    # If there are multiple, pick the largest token budget (last by sorted name).
    return finals[-1]


def _load_matryoshka_sae(ckpt_dir: Path, device: torch.device) -> torch.nn.Module:
    """Load the production Matryoshka SAE via SAELens load_from_disk."""
    from sae_lens import MatryoshkaBatchTopKTrainingSAE
    sae = MatryoshkaBatchTopKTrainingSAE.load_from_disk(str(ckpt_dir), device=str(device))
    sae.eval()
    return sae


def _make_substitution_hook(sae: torch.nn.Module, t_center: float, t_tol: float,
                            stats: dict):
    """Forward-hook that replaces the block output with sae(output) when t is
    within ±t_tol of t_center. Otherwise no-op.

    `stats` is mutated in place: counts active vs skipped fires for sanity.
    """
    sae_dtype = next(sae.parameters()).dtype

    def hook(module, inputs, output):
        t = current_t()
        if t is None:
            stats["no_t"] += 1
            return
        if abs(float(t) - t_center) > t_tol:
            stats["skipped"] += 1
            return
        # output: (B, T, D)
        orig_dtype = output.dtype
        x = output.to(dtype=sae_dtype)
        flat = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            recon = sae(flat)
        recon = recon.reshape(x.shape).to(dtype=orig_dtype)
        stats["active"] += 1
        return recon

    return hook


@torch.no_grad()
def _sample_with_substitution(
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
    *,
    sae: torch.nn.Module | None,
    sub_layer: int | None,
    sub_t_center: float | None,
    sub_t_tol: float,
) -> dict:
    """Generate `n_samples` images, optionally substituting block output with SAE
    reconstruction at (sub_layer, sub_t_center ± sub_t_tol).

    Returns a small stats dict counting hook fires.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(transport)
    if sampler_kind == "sde":
        sample_fn = sampler.sample_sde(
            sampling_method="Euler", diffusion_form="SBDM", num_steps=sample_steps,
        )
    else:
        sample_fn = sampler.sample_ode(sampling_method="dopri5", num_steps=sample_steps)

    in_ch = int(mean_d.shape[1])
    H = W = int(input_size)

    # Attach the hook for the duration of the sample loop.
    hook_handle = None
    stats = {"active": 0, "skipped": 0, "no_t": 0}
    if sae is not None and sub_layer is not None and sub_t_center is not None:
        hook_handle = model.blocks[sub_layer].register_forward_hook(
            _make_substitution_hook(sae, sub_t_center, sub_t_tol, stats)
        )

    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_classes = torch.randint(0, NUM_CLASSES, (n_samples,), generator=gen)

    n_done = 0
    try:
        while n_done < n_samples:
            bsz = min(batch_size, n_samples - n_done)
            noise = torch.randn(bsz, in_ch, H, W, device=device)
            labels = all_classes[n_done : n_done + bsz].to(device)
            null_label = torch.full_like(labels, NUM_CLASSES)
            noise_full = torch.cat([noise, noise], dim=0)
            labels_full = torch.cat([labels, null_label], dim=0)

            def model_fn(x, t, y, cfg=cfg_scale):
                # Set timestep_router so the substitution hook can read t.
                # x, t, y all share batch; the sampler integrates with a single
                # scalar t per call → all batch entries see the same value.
                t_scalar = float(t.detach().flatten()[0].item())
                with timestep_context(t_scalar):
                    out = model.forward_with_cfg(x, t, y, cfg)
                return out[:, :in_ch]

            samples = sample_fn(noise_full, model_fn, y=labels_full)[-1][:bsz]
            if denormalize:
                samples = samples / std_d + mean_d
            imgs = adapter.decode(samples).clamp_(-1, 1).add(1).div(2).clamp_(0, 1)
            for k, img in enumerate(imgs):
                save_image(img, out_dir / f"img_{n_done + k:06d}.png")
            n_done += bsz
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    return stats


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True,
                   help="Phase-2 SiT run dir.")
    p.add_argument("--adapter", type=str, required=True,
                   help="VAE adapter name for decode (sd_vae | repa_e | eq_vae).")
    p.add_argument("--condition", type=str, required=True,
                   help="Condition label for the SAE tree (sd_vae | repa_e | eq_vae).")
    p.add_argument("--sae_root", type=Path,
                   default=Path("/leonardo_scratch/fast/IscrC_PDR/lcerovaz/"
                                "diffmechint/sae_matryoshka_k256_d32k"),
                   help="Root of the production matryoshka SAE tree.")
    p.add_argument("--layer", type=int, required=False, default=None,
                   help="SiT block index to substitute at. Omit (or use --no_substitute) "
                        "for the baseline.")
    p.add_argument("--t_bin", type=int, required=False, default=None,
                   choices=[0, 1, 2],
                   help="t-bin index: 0=0.025, 1=0.20, 2=0.50. Required unless --no_substitute.")
    p.add_argument("--t_tol", type=float, default=0.01,
                   help="Tolerance (in t units) around the bin centre for substitution.")
    p.add_argument("--dit_step", type=int, default=200_000,
                   help="Which SiT EMA checkpoint to load.")
    p.add_argument("--no_substitute", action="store_true",
                   help="Baseline: skip SAE substitution entirely (no hook installed).")
    p.add_argument("--out_root", type=Path, required=True,
                   help="Per-cell output root; samples + fid.json land at "
                        "<out_root>/<run_tag>/.")
    p.add_argument("--model_name", type=str, default="SiT-B/2")
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--sampler", type=str, default="ode", choices=["sde", "ode"])
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--sample_steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_normalize", action="store_true",
                   help="Skip per-feature z-score (required for eq_vae_noz runs).")
    p.add_argument("--keep_images", action="store_true",
                   help="Don't delete generated PNGs after computing FID.")
    args = p.parse_args()

    # Validate substitution args.
    do_sub = not args.no_substitute
    if do_sub and (args.layer is None or args.t_bin is None):
        error("--layer and --t_bin are required unless --no_substitute is set.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available — this will be very slow.")

    from cleanfid import fid as cleanfid_mod
    if not cleanfid_mod.test_stats_exists(REF_NAME, mode="clean"):
        error(f"Reference stats '{REF_NAME}' not found in cleanfid cache.")
        return 1

    # Tag identifying this run.
    if do_sub:
        run_tag = f"{args.condition}__L{args.layer}_T{args.t_bin}__step_{args.dit_step:06d}"
    else:
        run_tag = f"{args.condition}__baseline__step_{args.dit_step:06d}"
    out_dir = args.out_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    info(f"run_tag={run_tag}  out_dir={out_dir}")

    # Adapter + latent stats.
    mean_d, std_d, stats = _load_stats(args.adapter)
    mean_d = mean_d.to(device)
    std_d = std_d.to(device)
    in_channels = int(stats["feature_dim"])
    input_size = int(stats["input_size"])
    info(f"adapter={args.adapter} in_ch={in_channels} input_size={input_size}")

    info("Loading frozen VAE adapter for decode")
    adapter = build(args.adapter)
    adapter.load()
    adapter.to(device)

    transport = create_transport(path_type="Linear", prediction="velocity", loss_weight=None)
    info(f"Building model {args.model_name}")
    model = _build_model(args.model_name, in_channels, input_size, device)

    # Load SiT EMA checkpoint at the requested step.
    ema_path = args.run / "checkpoints" / f"step_{args.dit_step:08d}_ema.safetensors"
    if not ema_path.exists():
        error(f"SiT EMA checkpoint missing: {ema_path}")
        return 1
    info(f"Loading SiT EMA from {ema_path.name}")
    sd = load_file(str(ema_path))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        warn(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    # Load SAE (if substituting).
    sae = None
    sae_ckpt_dir = None
    if do_sub:
        sae_ckpt_dir = _resolve_sae_ckpt(args.sae_root, args.condition,
                                         args.layer, args.t_bin, args.dit_step)
        info(f"Loading Matryoshka SAE from {sae_ckpt_dir}")
        sae = _load_matryoshka_sae(sae_ckpt_dir, device)
        info(f"  d_in={sae.cfg.d_in} d_sae={sae.cfg.d_sae} k={sae.cfg.k}")

    sub_t_center = T_BIN_CENTERS[args.t_bin] if do_sub else None
    images_dir = out_dir / "samples"
    if images_dir.exists():
        shutil.rmtree(images_dir)

    t0 = time.perf_counter()
    info(f"Generating {args.n_samples} samples "
         f"(sampler={args.sampler} steps={args.sample_steps} cfg={args.cfg})…")
    hook_stats = _sample_with_substitution(
        model, transport, adapter, mean_d, std_d,
        n_samples=args.n_samples, batch_size=args.batch_size,
        cfg_scale=args.cfg, sample_steps=args.sample_steps,
        out_dir=images_dir, seed=args.seed + args.dit_step, device=device,
        input_size=input_size, sampler_kind=args.sampler,
        denormalize=not args.no_normalize,
        sae=sae, sub_layer=args.layer, sub_t_center=sub_t_center,
        sub_t_tol=args.t_tol,
    )
    gen_sec = time.perf_counter() - t0
    info(f"  generation done in {gen_sec/60:.1f} min  hook_stats={hook_stats}")

    if do_sub and hook_stats["active"] == 0:
        warn("Hook never fired actively — substitution had zero effect. "
             "Check --t_tol and sampler t-trajectory.")

    info("Scoring with Clean-FID against imagenet_val_50k…")
    t1 = time.perf_counter()
    score = cleanfid_mod.compute_fid(
        str(images_dir),
        dataset_name=REF_NAME,
        dataset_split="custom",
        mode="clean",
    )
    fid_sec = time.perf_counter() - t1
    ok(f"FID-{args.n_samples//1000}k = {score:.3f}   "
       f"(gen {gen_sec/60:.1f} min, fid {fid_sec/60:.1f} min)")

    result = {
        "run_tag": run_tag,
        "condition": args.condition,
        "adapter": args.adapter,
        "layer": args.layer,
        "t_bin": args.t_bin,
        "t_bin_center": sub_t_center,
        "t_tol": args.t_tol,
        "dit_step": args.dit_step,
        "substituted": do_sub,
        "sae_ckpt_dir": str(sae_ckpt_dir) if sae_ckpt_dir else None,
        "model_name": args.model_name,
        "n_samples": args.n_samples,
        "cfg_scale": args.cfg,
        "sampler": args.sampler,
        "sample_steps": args.sample_steps,
        "seed": args.seed,
        "no_normalize": args.no_normalize,
        "hook_stats": hook_stats,
        "fid": float(score),
        "gen_minutes": gen_sec / 60.0,
        "fid_minutes": fid_sec / 60.0,
        "ref_name": REF_NAME,
    }
    (out_dir / "fid.json").write_text(json.dumps(result, indent=2))
    info(f"Wrote fid.json → {out_dir / 'fid.json'}")

    # Append to the run-level CSV log (one row per cell across the sweep).
    csv_path = args.out_root / "aggregate.csv"
    cols = list(result.keys())
    flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in result.items()}
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow(flat)
    info(f"Appended row to {csv_path}")

    if not args.keep_images:
        shutil.rmtree(images_dir, ignore_errors=True)
        info(f"Removed sample dir {images_dir}")

    ok(f"Done. run_tag={run_tag}  FID={score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
