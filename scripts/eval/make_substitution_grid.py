"""E06 — qualitative 2x2 grid for one class.

Compose: [real image | VAE round-trip ]
         [SiT baseline | SiT + SAE substitution at one cell]

Default target: golden retriever (synset n02099601, class idx 207). The cell
with the largest ΔFID in 4.5a is sd_vae L6/T2 (ΔFID = +1.80) — chosen to
maximize the visible effect of the substitution.

Usage (inside an `srun --gres=gpu:1`):
    uv run python scripts/eval/make_substitution_grid.py \\
        --run <sit_run_dir> --adapter sd_vae --condition sd_vae \\
        --layer 6 --t_bin 2 --class_idx 207 --seed 0 \\
        --out_path flywheel/sae/e06_substitution_fid/plots/qualitative_grid.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Symlink cleanfid Inception (unused here but cleanfid import is light).
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from diffmechint.hooks.timestep_router import current_t, timestep_context  # noqa: E402
from diffmechint.sae import load_matryoshka_sae, resolve_sae_ckpt  # noqa: E402
from diffmechint.sit import build_sit_model, create_transport  # noqa: E402
from diffmechint.sit.transport import Sampler  # noqa: E402
from diffmechint.tokenizers import build, load_latent_stats  # noqa: E402
from diffmechint.utils import info, ok, warn  # noqa: E402

IMAGENET_VAL = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/"
                    "imagenet_val_imagefolder")
SYNSETS_FILE = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/.venv/lib/"
                    "python3.11/site-packages/timm/data/_info/imagenet_synsets.txt")
NUM_CLASSES = 1000
T_BIN_CENTERS = (0.025, 0.20, 0.50)


def _class_idx_to_synset(class_idx):
    synsets = SYNSETS_FILE.read_text().splitlines()
    return synsets[class_idx]


def _load_real_image(class_idx, image_size=256):
    """Load and centre-crop one real ImageNet val image of the chosen class."""
    synset = _class_idx_to_synset(class_idx)
    class_dir = IMAGENET_VAL / synset
    if not class_dir.exists():
        raise FileNotFoundError(f"class dir missing: {class_dir}")
    candidates = sorted(class_dir.glob("*.JPEG"))
    if not candidates:
        raise FileNotFoundError(f"no JPEGs under {class_dir}")
    img_path = candidates[0]
    info(f"Real image: {img_path.relative_to(IMAGENET_VAL)}")
    pil = Image.open(img_path).convert("RGB")
    # Centre-crop to square then resize, matching the train preprocessing.
    w, h = pil.size
    s = min(w, h)
    pil = pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    pil = pil.resize((image_size, image_size), Image.BICUBIC)
    return pil, img_path


@torch.no_grad()
def _vae_roundtrip(pil_img, adapter, mean_d, std_d, device, denormalize=True):
    """Encode through the VAE adapter then decode — the "ideal" reconstruction
    upper bound for any latent-space intervention."""
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0  # (H, W, 3) ∈ [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    t = t * 2.0 - 1.0  # → [-1, 1]
    z = adapter.encode(t)
    # The z is already in "raw" latent space; we don't apply the train z-score
    # for decode (the adapter handles the scaling_factor internally; the
    # per-feature z-score is only applied when training the SiT).
    out = adapter.decode(z).clamp_(-1, 1).add(1).div(2).clamp_(0, 1)  # (1,3,H,W)
    return out[0].cpu()


@torch.no_grad()
def _generate_one(model, transport, adapter, mean_d, std_d, class_idx, seed,
                  cfg_scale, sample_steps, device, input_size, denormalize,
                  sae=None, sub_layer=None, sub_t_center=None, sub_t_tol=0.01):
    """Generate a single image at the chosen class label with optional SAE hook."""
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method="dopri5", num_steps=sample_steps)

    in_ch = int(mean_d.shape[1])
    H = W = int(input_size)

    hook_stats = {"active": 0, "skipped": 0, "no_t": 0}
    hook_handle = None
    if sae is not None and sub_layer is not None and sub_t_center is not None:
        sae_dtype = next(sae.parameters()).dtype

        def hook(module, inputs, output):
            t = current_t()
            if t is None:
                hook_stats["no_t"] += 1
                return
            if abs(float(t) - sub_t_center) > sub_t_tol:
                hook_stats["skipped"] += 1
                return
            orig_dtype = output.dtype
            x = output.to(dtype=sae_dtype)
            flat = x.reshape(-1, x.shape[-1])
            recon = sae(flat).reshape(x.shape).to(dtype=orig_dtype)
            hook_stats["active"] += 1
            return recon

        hook_handle = model.blocks[sub_layer].register_forward_hook(hook)

    try:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        # Same per-class noise for the two generations to keep them comparable.
        noise = torch.randn(1, in_ch, H, W, device=device,
                            generator=torch.Generator(device=device).manual_seed(seed))
        labels = torch.tensor([class_idx], dtype=torch.long, device=device)
        null_label = torch.full_like(labels, NUM_CLASSES)
        noise_full = torch.cat([noise, noise], dim=0)
        labels_full = torch.cat([labels, null_label], dim=0)

        def model_fn(x, t, y, cfg=cfg_scale):
            t_scalar = float(t.detach().flatten()[0].item())
            with timestep_context(t_scalar):
                out = model.forward_with_cfg(x, t, y, cfg)
            return out[:, :in_ch]

        samples = sample_fn(noise_full, model_fn, y=labels_full)[-1][:1]
        if denormalize:
            samples = samples / std_d + mean_d
        imgs = adapter.decode(samples).clamp_(-1, 1).add(1).div(2).clamp_(0, 1)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    return imgs[0].cpu(), hook_stats


def _tile_grid(rows_baseline, rows_substituted, row_labels, banner,
               out_path, image_size=256, pad=10, header=56, col_label_h=32):
    """Compose an N-row × 2-column grid: baseline | substituted, per class.

    rows_baseline, rows_substituted : list of (3, H, W) tensors, one per class.
    row_labels : list[str] — class names, one per row.
    banner : top banner text (cell + run metadata).
    """
    n_rows = len(rows_baseline)
    canvas_w = 2 * image_size + 3 * pad
    canvas_h = n_rows * image_size + (n_rows + 1) * pad + header + col_label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 240, 220))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16,
        )
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22,
        )
        col_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18,
        )
    except OSError:
        font = ImageFont.load_default()
        title_font = font
        col_font = font

    draw = ImageDraw.Draw(canvas)
    # Header banner
    draw.rectangle([(0, 0), (canvas_w, header)], fill=(53, 92, 103))  # #335C67
    draw.text((pad, 16), banner, fill=(255, 243, 176), font=title_font)
    # Column headers
    draw.rectangle([(0, header), (canvas_w, header + col_label_h)], fill=(84, 11, 14))  # #540B0E
    draw.text((pad + image_size // 2 - 60, header + 6), "baseline",
              fill=(255, 243, 176), font=col_font)
    draw.text((image_size + 2 * pad + image_size // 2 - 110, header + 6),
              "+ Matryoshka substitution", fill=(255, 243, 176), font=col_font)

    y0 = header + col_label_h + pad
    for i, (left_t, right_t, lbl) in enumerate(zip(rows_baseline, rows_substituted, row_labels)):
        y = y0 + i * (image_size + pad)
        for col, (x, tensor) in enumerate(((pad, left_t),
                                            (image_size + 2 * pad, right_t))):
            arr = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr).resize((image_size, image_size), Image.BICUBIC)
            canvas.paste(pil, (x, y))
        # Per-row class label, on the left image, bottom band.
        band_h = 26
        draw.rectangle([(pad, y + image_size - band_h),
                        (pad + image_size, y + image_size)], fill=(0, 0, 0))
        draw.text((pad + 6, y + image_size - band_h + 3), lbl,
                  fill=(255, 243, 176), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return canvas


CLASS_NAMES_BY_IDX = {
    207: "golden retriever",
    281: "tabby cat",
    933: "cheeseburger",
    980: "volcano",
    388: "giant panda",
    207: "golden retriever",
    963: "pizza",
    948: "Granny Smith apple",
    283: "Persian cat",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--condition", type=str, required=True)
    p.add_argument("--sae_root", type=Path,
                   default=Path("/leonardo_scratch/fast/IscrC_PDR/lcerovaz/"
                                "diffmechint/sae_matryoshka_k256_d32k"))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--t_bin", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--t_tol", type=float, default=0.01)
    p.add_argument("--dit_step", type=int, default=200_000)
    p.add_argument("--class_indices", type=int, nargs="+",
                   default=[207, 933, 281, 980],
                   help="ImageNet class indices, one per grid row.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--sample_steps", type=int, default=250)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--model_name", type=str, default="SiT-B/2")
    p.add_argument("--out_path", type=Path, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available — this will be very slow.")

    mean_d, std_d, stats = load_latent_stats(args.adapter)
    mean_d = mean_d.to(device)
    std_d = std_d.to(device)
    in_channels = int(stats["feature_dim"])
    input_size = int(stats["input_size"])
    info(f"adapter={args.adapter} in_ch={in_channels} input_size={input_size}")

    adapter = build(args.adapter)
    adapter.load()
    adapter.to(device)

    transport = create_transport(path_type="Linear", prediction="velocity", loss_weight=None)
    model = build_sit_model(args.model_name, in_channels, input_size, device)

    ema_path = args.run / "checkpoints" / f"step_{args.dit_step:08d}_ema.safetensors"
    info(f"loading SiT EMA from {ema_path.name}")
    sd = load_file(str(ema_path))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    sae_ckpt = resolve_sae_ckpt(args.sae_root, args.condition,
                                 args.layer, args.t_bin, args.dit_step)
    info(f"loading Matryoshka SAE from {sae_ckpt}")
    sae = load_matryoshka_sae(sae_ckpt, device)

    baselines = []
    subs = []
    labels = []
    hook_stats_all = []
    for class_idx in args.class_indices:
        name = CLASS_NAMES_BY_IDX.get(class_idx, f"class_{class_idx}")
        info(f"--- class {class_idx} ({name}) ---")

        # Baseline (no substitution)
        t0 = time.perf_counter()
        sit_base, _ = _generate_one(
            model, transport, adapter, mean_d, std_d,
            class_idx=class_idx, seed=args.seed, cfg_scale=args.cfg,
            sample_steps=args.sample_steps, device=device, input_size=input_size,
            denormalize=not args.no_normalize, sae=None,
        )
        info(f"  baseline done in {(time.perf_counter() - t0)/60:.1f} min")

        # Substituted
        t0 = time.perf_counter()
        sit_sub, hs_sub = _generate_one(
            model, transport, adapter, mean_d, std_d,
            class_idx=class_idx, seed=args.seed, cfg_scale=args.cfg,
            sample_steps=args.sample_steps, device=device, input_size=input_size,
            denormalize=not args.no_normalize,
            sae=sae, sub_layer=args.layer,
            sub_t_center=T_BIN_CENTERS[args.t_bin], sub_t_tol=args.t_tol,
        )
        info(f"  sub done in {(time.perf_counter() - t0)/60:.1f} min  hook={hs_sub}")

        baselines.append(sit_base)
        subs.append(sit_sub)
        labels.append(f"{name} (idx {class_idx})")
        hook_stats_all.append(hs_sub)

    banner = (f"E06  •  {args.condition} L{args.layer}/T{args.t_bin} "
              f"(t={T_BIN_CENTERS[args.t_bin]:.3f})  •  "
              f"same class label + same seed for both columns")
    _tile_grid(baselines, subs, labels, banner, args.out_path)
    ok(f"Saved {len(args.class_indices)}-row grid → {args.out_path}")

    # Side-car JSON with metadata.
    meta = {
        "class_indices": args.class_indices,
        "class_names": [CLASS_NAMES_BY_IDX.get(i, f"class_{i}")
                        for i in args.class_indices],
        "synsets": [_class_idx_to_synset(i) for i in args.class_indices],
        "condition": args.condition,
        "adapter": args.adapter,
        "layer": args.layer,
        "t_bin": args.t_bin,
        "t_bin_center": T_BIN_CENTERS[args.t_bin],
        "t_tol": args.t_tol,
        "dit_step": args.dit_step,
        "sae_ckpt_dir": str(sae_ckpt),
        "seed": args.seed,
        "cfg": args.cfg,
        "sample_steps": args.sample_steps,
        "no_normalize": args.no_normalize,
        "hook_stats_per_class": hook_stats_all,
    }
    (args.out_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
