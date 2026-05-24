"""Image → SAE feature inspector for a single (cond, layer, t_bin) cell.

Given an image (or a class index sampling N val images), encode through the
condition's VAE, build the diffusion noisy latent x_t at the cell's t_bin
center, forward through SiT with y=null_token, tap the residual stream at
the cell's layer, encode through the cell's Matryoshka SAE, and report the
top-K firing features. Join each feature with the dashboard JSON to attach
VLM interpretation, density, and top-9 reference images.

Usage::

    # single image
    uv run python scripts/analysis/inspect_image.py \\
        --cell repa_e_L3_T0 --image path/to/panda.jpg --topk 20

    # class batch (sample 10 val images from class 388 = giant panda)
    uv run python scripts/analysis/inspect_image.py \\
        --cell repa_e_L3_T0 --class_idx 388 --n_images 10
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

_FAST = "/leonardo_scratch/fast/IscrC_YENDRI"
os.environ.setdefault("HF_HOME", f"{_FAST}/lcerovaz/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", f"{_FAST}/lcerovaz/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_FAST}/lcerovaz/hf_cache/transformers")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torchvision import transforms  # noqa: E402

from diffmechint.hooks import ActivationBuffer, ResidualStreamTap  # noqa: E402
from diffmechint.hooks.timestep_router import timestep_context  # noqa: E402
from diffmechint.sit import SiT_models  # noqa: E402
from diffmechint.tokenizers import build as build_tokenizer  # noqa: E402
from diffmechint.utils import error, info, ok, warn  # noqa: E402

NUM_CLASSES = 1000
T_BIN_CENTERS = (0.025, 0.20, 0.50)
LATENTS_BASE = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents")
IMAGEFOLDER_ROOT = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/imagenet_val_imagefolder"
)
SAE_ROOT = Path("/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k")
SIT_RUN_ROOT = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs")
DASHBOARD_ROOT = Path(
    "/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/phase4_5b_feature_viz_ynull"
)
TIMM_SYNSETS = Path(
    "/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/.venv/lib/python3.11/"
    "site-packages/timm/data/_info/imagenet_synsets.txt"
)
TIMM_LEMMAS = Path(
    "/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/.venv/lib/python3.11/"
    "site-packages/timm/data/_info/imagenet_synset_to_lemma.txt"
)

CELL_RE = re.compile(r"^(?P<cond>[a-z_0-9]+)_L(?P<layer>\d+)_T(?P<tbin>\d+)$")


def parse_cell(cell: str) -> tuple[str, int, int]:
    m = CELL_RE.match(cell)
    if not m:
        raise ValueError(f"bad --cell '{cell}'; expected <cond>_L<int>_T<int>")
    return m["cond"], int(m["layer"]), int(m["tbin"])


def load_class_names() -> tuple[list[str], dict[str, str]]:
    syns = [s.strip() for s in TIMM_SYNSETS.read_text().splitlines() if s.strip()]
    lemmas: dict[str, str] = {}
    for line in TIMM_LEMMAS.read_text().splitlines():
        if not line.strip():
            continue
        s, lemma = line.split("\t", 1)
        lemmas[s.strip()] = lemma.strip()
    return syns, lemmas


def build_image_transform(image_size: int = 256) -> transforms.Compose:
    # Matches scripts/extraction/precompute_latents.py exactly.
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def load_image(path: Path, tform: transforms.Compose) -> torch.Tensor:
    with open(path, "rb") as fh:
        raw = fh.read()
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        return tform(img)


def sample_class_images(class_idx: int, n: int, synsets: list[str],
                        tform: transforms.Compose, rng: np.random.Generator,
                        ) -> tuple[torch.Tensor, list[str]]:
    syn = synsets[class_idx]
    files = sorted(p for p in (IMAGEFOLDER_ROOT / syn).iterdir() if not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"no images in {IMAGEFOLDER_ROOT / syn}")
    picks = rng.choice(len(files), size=min(n, len(files)), replace=False)
    paths = [files[int(i)] for i in picks]
    tensors = torch.stack([load_image(p, tform) for p in paths], dim=0)
    return tensors, [p.name for p in paths]


def load_normalization(cond: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    stats = json.loads((LATENTS_BASE / cond / "stats.json").read_text())
    mean = np.asarray(stats["per_feature_mean"], dtype=np.float32)
    std = np.asarray(stats["per_feature_std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    n_dims = len(stats["latent_shape"])
    feat_axis = stats["feature_axis"]
    shape = [1] + [1] * n_dims
    shape[feat_axis] = mean.shape[0]
    return (
        torch.from_numpy(mean).view(*shape),
        torch.from_numpy(1.0 / std).view(*shape),
        stats,
    )


def load_sit(cond: str, dit_step: int, in_channels: int, input_size: int,
             model_name: str, device: torch.device) -> torch.nn.Module:
    model = SiT_models[model_name](
        input_size=input_size,
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ).to(device).eval()
    ema_path = SIT_RUN_ROOT / f"sit_{cond}" / "checkpoints" / f"step_{dit_step:06d}_ema.safetensors"
    if not ema_path.exists():
        raise FileNotFoundError(f"SiT EMA ckpt missing: {ema_path}")
    sd = load_file(str(ema_path))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        warn(f"SiT load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    return model


def load_sae(cond: str, layer: int, t_bin: int, dit_step: int, device: torch.device):
    from sae_lens import MatryoshkaBatchTopKTrainingSAE
    base = SAE_ROOT / cond / f"L{layer}_T{t_bin}" / f"step_{dit_step:06d}"
    finals = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("final_"))
    if not finals:
        raise FileNotFoundError(f"no final_* under {base}")
    sae = MatryoshkaBatchTopKTrainingSAE.load_from_disk(str(finals[-1]), device=str(device))
    sae.eval()
    return sae


def load_feature_meta(cell_dir: Path, feature_ids: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for fid in feature_ids:
        fp = cell_dir / "features" / f"feature_{fid}.json"
        if not fp.exists():
            out[fid] = {"feature_id": fid, "live": False}
            continue
        d = json.loads(fp.read_text())
        top = d.get("top", [])
        out[fid] = {
            "feature_id": fid,
            "live": d.get("live", True),
            "density": d.get("density"),
            "entropy": d.get("entropy"),
            "vlm_interpretation": d.get("vlm_interpretation"),
            "top_label": top[0]["label"] if top else None,
            "top_class_idx": top[0]["class_idx"] if top else None,
            "n_unique_classes": d.get("unique_classes"),
        }
    return out


@torch.no_grad()
def run_pipeline(
    imgs: torch.Tensor,            # (B, 3, 256, 256), normalized [-1, 1]
    cond: str,
    layer: int,
    t_bin: int,
    dit_step: int,
    model_name: str,
    seed: int,
    y_null: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward `imgs` through VAE → SiT (with tap) → SAE; return per-image
    max-pooled SAE activations (B, d_sae) and per-feature mean over B.
    """
    mean, inv_std, stats = load_normalization(cond)
    in_channels = int(stats["feature_dim"])
    input_size = int(stats["input_size"])

    info(f"Building VAE adapter ({cond})")
    adapter = build_tokenizer(cond)
    adapter.load()
    adapter.to(device).eval()

    info(f"Building SiT {model_name} (in_ch={in_channels}, input_size={input_size})")
    model = load_sit(cond, dit_step, in_channels, input_size, model_name, device)

    info(f"Loading Matryoshka SAE for L{layer}_T{t_bin} step_{dit_step:06d}")
    sae = load_sae(cond, layer, t_bin, dit_step, device)

    imgs = imgs.to(device)
    z = adapter.encode(imgs)
    z = (z.to(torch.float32) - mean.to(device)) * inv_std.to(device)
    info(f"  z_clean shape={tuple(z.shape)}  mean={z.mean():.3f}  std={z.std():.3f}")

    torch.manual_seed(seed)
    eps = torch.randn_like(z)
    t_val = float(T_BIN_CENTERS[t_bin])
    x_t = (1.0 - t_val) * eps + t_val * z
    y_for_forward = torch.full((z.shape[0],), NUM_CLASSES if y_null else 0,
                               dtype=torch.long, device=device)
    t_vec = torch.full((z.shape[0],), t_val, device=device)

    buf = ActivationBuffer(max_records_per_cell=0,
                           shard_dir=None,
                           store_dtype=torch.float16)
    tap = ResidualStreamTap(model, block_indices=[layer], buffer=buf,
                            bins=(t_val,), tol=0.01)
    tap.attach()
    try:
        with timestep_context(t_val):
            _ = model(x_t, t_vec, y_for_forward)
    finally:
        tap.detach()

    # bins=(t_val,) → the only bin is index 0; one entry per batched sample.
    records = buf.get(layer=layer, t_bin=0)
    if not records:
        raise RuntimeError(
            f"buffer empty — tap did not fire (t_val={t_val}); check timestep_context."
        )
    h = torch.stack(records, dim=0).to(device=device, dtype=next(sae.parameters()).dtype)
    info(f"  h (residual @ L{layer}) shape={tuple(h.shape)}")

    b, t_tok, d = h.shape
    z_sae = sae.encode(h.reshape(b * t_tok, d))  # (B*T, d_sae)
    z_sae = z_sae.float().view(b, t_tok, -1)
    acts_per_image, _ = z_sae.max(dim=1)  # (B, d_sae)
    acts_mean = acts_per_image.mean(dim=0)  # (d_sae,)
    return acts_per_image.cpu().numpy(), acts_mean.cpu().numpy()


def render_table(rows: list[dict], topk: int) -> str:
    lines = []
    header = f"{'rank':>4}  {'fid':>6}  {'act':>8}  {'count':>5}  {'dens%':>6}  {'top_label':<30}  vlm"
    lines.append(header)
    lines.append("-" * len(header))
    for i, r in enumerate(rows[:topk]):
        dens = (r.get("density") or 0.0) * 100
        top_lbl = (r.get("top_label") or "")[:30]
        vlm = r.get("vlm_interpretation") or ""
        count_str = f"{r.get('count', '-')}"
        lines.append(
            f"{i+1:>4}  {r['feature_id']:>6}  {r['act_score']:>8.3f}  "
            f"{count_str:>5}  {dens:>6.3f}  {top_lbl:<30}  {vlm}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=str, required=True,
                    help="Cell selector: <cond>_L<layer>_T<tbin>, e.g. repa_e_L3_T0")
    ap.add_argument("--image", type=Path, default=None,
                    help="Single image path (mutually exclusive with --class_idx)")
    ap.add_argument("--class_idx", type=int, default=None,
                    help="ImageNet class index (0-999); samples --n_images from val")
    ap.add_argument("--n_images", type=int, default=10,
                    help="Number of val images to sample for class batch mode")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--dit_step", type=int, default=200000)
    ap.add_argument("--model_name", type=str, default="SiT-B/2")
    ap.add_argument("--y_null", action="store_true", default=True,
                    help="(default) Forward with null-token conditioning")
    ap.add_argument("--no_y_null", dest="y_null", action="store_false",
                    help="Use class label 0 for conditioning instead of null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", type=Path,
                    default=Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/inspect"))
    ap.add_argument("--dashboard_root", type=Path, default=DASHBOARD_ROOT)
    args = ap.parse_args()

    if (args.image is None) == (args.class_idx is None):
        error("specify exactly one of --image or --class_idx")
        return 2

    cond, layer, t_bin = parse_cell(args.cell)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available — inspection will be slow.")

    info(f"Cell: cond={cond} layer={layer} t_bin={t_bin} (t_val={T_BIN_CENTERS[t_bin]})")
    synsets, lemmas = load_class_names()

    tform = build_image_transform()
    rng = np.random.default_rng(args.seed)
    if args.image is not None:
        imgs = load_image(args.image, tform).unsqueeze(0)
        source_paths = [str(args.image)]
        info(f"Single image: {args.image}")
    else:
        imgs, names = sample_class_images(args.class_idx, args.n_images,
                                          synsets, tform, rng)
        source_paths = [str(IMAGEFOLDER_ROOT / synsets[args.class_idx] / n) for n in names]
        info(f"Class {args.class_idx} ({lemmas[synsets[args.class_idx]]}): "
             f"sampled {len(source_paths)} images")

    t0 = time.perf_counter()
    acts_per_image, acts_mean = run_pipeline(
        imgs=imgs, cond=cond, layer=layer, t_bin=t_bin,
        dit_step=args.dit_step, model_name=args.model_name,
        seed=args.seed, y_null=args.y_null, device=device,
    )
    ok(f"forward pass done in {(time.perf_counter() - t0):.1f}s "
       f"({imgs.shape[0]} imgs, d_sae={acts_per_image.shape[1]})")

    # Two rankings: by mean activation (default), and by count-active
    # (consistency). Both are useful — print mean-act on top, but write both.
    count_active = (acts_per_image > 1e-6).sum(axis=0)  # (d_sae,)
    rank_mean = np.argsort(-acts_mean)[: args.topk]
    rank_count = np.argsort(-count_active)[: args.topk]
    cited = sorted(set(rank_mean.tolist()) | set(rank_count.tolist()))

    cell_dir = args.dashboard_root / f"{cond}_L{layer}_T{t_bin}"
    meta = load_feature_meta(cell_dir, cited)

    def make_rows(rank: np.ndarray) -> list[dict]:
        rows = []
        for fid in rank.tolist():
            m = dict(meta.get(fid, {"feature_id": fid, "live": False}))
            m["act_score"] = float(acts_mean[fid])
            m["count"] = int(count_active[fid])
            rows.append(m)
        return rows

    rows_mean = make_rows(rank_mean)
    rows_count = make_rows(rank_count)

    print("\nTop-K features by MEAN activation:")
    print(render_table(rows_mean, args.topk))
    print("\nTop-K features by COUNT (images firing):")
    print(render_table(rows_count, args.topk))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"{args.cell}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "cell": args.cell,
        "cond": cond, "layer": layer, "t_bin": t_bin, "t_val": T_BIN_CENTERS[t_bin],
        "dit_step": args.dit_step,
        "model_name": args.model_name,
        "y_null": args.y_null,
        "seed": args.seed,
        "mode": "image" if args.image is not None else "class_batch",
        "image_path": str(args.image) if args.image is not None else None,
        "class_idx": args.class_idx,
        "class_label": (lemmas[synsets[args.class_idx]] if args.class_idx is not None else None),
        "n_images": int(imgs.shape[0]),
        "source_paths": source_paths,
        "topk": args.topk,
        "ranking_mean_act": rows_mean,
        "ranking_count_active": rows_count,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    np.save(out_dir / "acts_per_image.npy", acts_per_image.astype(np.float32))
    ok(f"Wrote result to {out_dir}/result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
