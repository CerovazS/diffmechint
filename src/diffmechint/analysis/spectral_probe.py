"""E38 spectral probing: per-band linear probes of class information in latents and DiT residuals."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

from diffmechint.analysis.alignment import cell_path, load_manifest, resolve_labels
from diffmechint.analysis.latent_probe import (
    VAL_ROOT,
    build_features,
    load_val,
    stratified_split,
    torch_probe,
)
from diffmechint.sae.data_provider import patchify_latents
from diffmechint.spectral import (
    dct2,
    grid_to_tokens,
    idct2,
    octave_band_masks,
    tokens_to_grid,
)
from diffmechint.utils import info, ok, warn
from diffmechint.utils.plotting import PALETTE_B

LATENTS_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents_val")
ACT_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_ynull")
CONDITIONS = ["sd_vae", "eq_vae", "repa_e"]
DEFAULT_BAND_SPECS = ["broadband", "B0", "B1", "B2", "B3", "B4", "LP1", "LP2", "LP3"]
CSV_FIELDS = [
    "source", "condition", "layer", "t_bin", "band_spec", "feature_set",
    "dim", "n_train", "n_test", "n_classes", "top1", "top5", "chance",
]


def spec_to_mask(spec: str, masks: torch.Tensor) -> torch.Tensor | None:
    """Band spec string -> bool DCT mask (h,w); None means broadband (no filtering)."""
    if spec == "broadband":
        return None
    if spec.startswith("LP"):
        k = int(spec[2:])
        if not 0 <= k < masks.shape[0]:
            raise ValueError(f"low-pass cutoff out of range: {spec}")
        return masks[: k + 1].any(dim=0)
    if spec.startswith("B"):
        b = int(spec[1:])
        if not 0 <= b < masks.shape[0]:
            raise ValueError(f"band index out of range: {spec}")
        return masks[b]
    raise ValueError(f"unknown band spec: {spec}")


def filter_tokens_from_coeffs(coeffs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cached DCT coeffs (N,d,h,w) + bool mask (h,w) -> band-limited tokens (N,h*w,d)."""
    return grid_to_tokens(idct2(coeffs * mask.to(coeffs.dtype)))


def token_selection(n_images: int, n_tokens: int, per_img: int, seed: int) -> np.ndarray:
    """Deterministic (n_images, per_img) token indices, shared across all band specs."""
    rng = np.random.default_rng(seed)
    base = np.tile(np.arange(n_tokens), (n_images, 1))
    return rng.permuted(base, axis=1)[:, :per_img]


def gather_selected_tokens(tokens: torch.Tensor, sel: np.ndarray) -> torch.Tensor:
    """(N,T,d) tokens + (N,per_img) indices -> (N,per_img,d)."""
    idx = torch.from_numpy(sel).long()
    return tokens[torch.arange(tokens.shape[0])[:, None], idx]


def probe_band_specs(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    tr: np.ndarray,
    te: np.ndarray,
    band_specs: list[str],
    feature_sets: list[str],
    n_classes: int,
    device: str,
    epochs: int,
    lr: float,
    batch: int,
    wd: float,
    token_cap: int = 600_000,
    seed: int = 0,
) -> list[dict]:
    """Band-filter (N,T,d) tokens per spec, probe each feature set; returns metric rows."""
    n, t, _d = tokens.shape
    gh = math.isqrt(t)
    if gh * gh != t:
        raise ValueError(f"token count {t} is not a square grid")
    masks = octave_band_masks(gh, gh, device=tokens.device)
    coeffs = dct2(tokens_to_grid(tokens, gh, gh))
    labels_np = labels.numpy()
    per_img = max(1, token_cap // n)
    tok_sel = token_selection(n, t, per_img, seed) if "token" in feature_sets else None

    rows = []
    for spec in band_specs:
        mask = spec_to_mask(spec, masks)
        filt = tokens if mask is None else filter_tokens_from_coeffs(coeffs, mask)
        for fset in feature_sets:
            if fset == "token":
                sel_feats = gather_selected_tokens(filt, tok_sel)  # (N,per_img,d)
                x_tr = sel_feats[tr].reshape(-1, sel_feats.shape[-1])
                x_te = sel_feats[te].reshape(-1, sel_feats.shape[-1])
                y_tr = torch.from_numpy(np.repeat(labels_np[tr], per_img)).long()
                y_te = torch.from_numpy(np.repeat(labels_np[te], per_img)).long()
            else:
                feats = build_features(filt, fset)
                x_tr, y_tr = feats[tr], labels[tr]
                x_te, y_te = feats[te], labels[te]
            top1, top5 = torch_probe(
                x_tr.float(), y_tr, x_te.float(), y_te, n_classes,
                device, epochs, lr, batch, wd,
            )
            info(f"    {spec:9s} {fset:9s} dim={x_tr.shape[1]:6d} "
                 f"top1={top1:.4f} top5={top5:.4f}")
            rows.append({
                "band_spec": spec, "feature_set": fset, "dim": int(x_tr.shape[1]),
                "n_train": int(x_tr.shape[0]), "n_test": int(x_te.shape[0]),
                "n_classes": n_classes, "top1": round(top1, 5), "top5": round(top5, 5),
                "chance": round(1.0 / n_classes, 5),
            })
    return rows


def load_val_capped(cond: str, patch_size: int, max_images: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Like latent_probe.load_val but stops after `max_images` (low-memory smoke path)."""
    import glob

    shards = sorted(glob.glob(str(VAL_ROOT / cond / "*.h5")))
    if not shards:
        raise FileNotFoundError(f"no val shards under {VAL_ROOT / cond}")
    toks, labs, count = [], [], 0
    for s in shards:
        with h5py.File(s, "r") as f:
            take = min(f["latents"].shape[0], max_images - count)
            lat = torch.from_numpy(f["latents"][:take]).float()
            labs.append(torch.from_numpy(f["labels"][:take]).long())
        toks.append(patchify_latents(lat, patch_size))
        count += take
        if count >= max_images:
            break
    return torch.cat(toks), torch.cat(labs)


def run_latent(args: argparse.Namespace, device: str) -> list[dict]:
    """Per-condition spectral probing of tokenizer latent tokens."""
    rows = []
    for cond in args.conditions:
        info(f"[latent] loading val latents: {cond}")
        if args.max_images is not None:
            tokens, labels = load_val_capped(cond, args.patch_size, args.max_images)
        else:
            tokens, labels = load_val(cond, args.patch_size)
        labels_np = labels.numpy()
        n_classes = int(np.unique(labels_np).shape[0])
        tr_idx, te_idx = stratified_split(labels_np, args.n_train, args.n_test, args.seed)
        sub = np.concatenate([tr_idx, te_idx])
        tokens, labels = tokens[sub], labels[sub]
        tr = np.arange(len(tr_idx))
        te = np.arange(len(tr_idx), len(sub))
        info(f"[latent] {cond}: N={len(sub)} T={tokens.shape[1]} D={tokens.shape[2]} "
             f"classes={n_classes} train={len(tr)} test={len(te)}")
        for r in probe_band_specs(
            tokens, labels, tr, te, args.band_specs, args.feature_sets,
            n_classes, device, args.epochs, args.lr, args.batch,
            args.weight_decay, args.token_cap, args.seed,
        ):
            rows.append({"source": "latent", "condition": cond, "layer": "", "t_bin": "", **r})
    return rows


def stream_dit_band_features(
    h5_path: Path,
    specs: list[str],
    masks: torch.Tensor,
    tok_sel: np.ndarray | None,
    chunk: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Stream a DiT cell h5, band-decompose chunk-wise, return per-spec mean_pool / token features.

    Returns (mean_feats[spec] -> (N,d) fp16, tok_feats[spec] -> (N,per_img,d) fp16);
    tok_feats empty when `tok_sel` is None. Never holds (B,N,T,d) in memory.
    """
    mean_acc: dict[str, list[torch.Tensor]] = {s: [] for s in specs}
    tok_acc: dict[str, list[torch.Tensor]] = {s: [] for s in specs}
    with h5py.File(h5_path, "r") as f:
        ds = f["activations"]
        n, t, _d = ds.shape
        gh = math.isqrt(t)
        for lo in range(0, n, chunk):
            x = torch.from_numpy(ds[lo:lo + chunk]).float().to(device)  # (n,T,d)
            coeffs = dct2(tokens_to_grid(x, gh, gh))
            for spec in specs:
                mask = spec_to_mask(spec, masks)
                ftok = x if mask is None else filter_tokens_from_coeffs(coeffs, mask)
                mean_acc[spec].append(ftok.mean(dim=1).half().cpu())
                if tok_sel is not None:
                    sel = tok_sel[lo:lo + x.shape[0]]
                    tok_acc[spec].append(gather_selected_tokens(ftok, sel).half().cpu())
            del x, coeffs
    mean_feats = {s: torch.cat(v) for s, v in mean_acc.items()}
    tok_feats = {s: torch.cat(v) for s, v in tok_acc.items()} if tok_sel is not None else {}
    return mean_feats, tok_feats


def run_dit(args: argparse.Namespace, device: str) -> list[dict]:
    """Per-condition x (layer, t_bin) spectral probing of DiT residual activations."""
    rows = []
    for cond in args.conditions:
        manifest = load_manifest(ACT_ROOT, cond, args.dit_step)
        sample_idx = np.asarray(manifest["sample_idx"], dtype=np.int64)
        labels_np = resolve_labels(LATENTS_ROOT, cond, sample_idx)
        n_classes = int(np.unique(labels_np).shape[0])
        tr_idx, te_idx = stratified_split(labels_np, args.n_train, args.n_test, args.seed)
        tr_idx, te_idx = np.sort(tr_idx), np.sort(te_idx)
        labels = torch.from_numpy(labels_np).long()
        info(f"[dit] {cond}: rows={len(sample_idx)} classes={n_classes} "
             f"train={len(tr_idx)} test={len(te_idx)}")

        for layer in args.layers:
            for t_bin in args.t_bins:
                path = cell_path(ACT_ROOT, cond, args.dit_step, layer, t_bin)
                info(f"[dit] {cond} L{layer} t{t_bin}: streaming {path.name}")
                with h5py.File(path, "r") as f:
                    n, t, _d = f["activations"].shape
                gh = math.isqrt(t)
                masks = octave_band_masks(gh, gh, device=device)
                want_token = "token" in args.feature_sets
                per_img = max(1, args.token_cap // (len(tr_idx) + len(te_idx)))
                tok_sel = token_selection(n, t, per_img, args.seed) if want_token else None
                mean_feats, tok_feats = stream_dit_band_features(
                    path, args.band_specs, masks, tok_sel, args.chunk, device,
                )
                for spec in args.band_specs:
                    for fset in args.feature_sets:
                        if fset == "token":
                            tf = tok_feats[spec]
                            x_tr = tf[tr_idx].reshape(-1, tf.shape[-1])
                            x_te = tf[te_idx].reshape(-1, tf.shape[-1])
                            y_tr = torch.from_numpy(np.repeat(labels_np[tr_idx], per_img)).long()
                            y_te = torch.from_numpy(np.repeat(labels_np[te_idx], per_img)).long()
                        elif fset == "mean_pool":
                            x_tr, y_tr = mean_feats[spec][tr_idx], labels[tr_idx]
                            x_te, y_te = mean_feats[spec][te_idx], labels[te_idx]
                        else:
                            raise ValueError(f"unsupported DiT feature set: {fset}")
                        top1, top5 = torch_probe(
                            x_tr.float(), y_tr, x_te.float(), y_te, n_classes,
                            device, args.epochs, args.lr, args.batch, args.weight_decay,
                        )
                        info(f"    {spec:9s} {fset:9s} dim={x_tr.shape[1]:6d} "
                             f"top1={top1:.4f} top5={top5:.4f}")
                        rows.append({
                            "source": "dit", "condition": cond, "layer": layer,
                            "t_bin": t_bin, "band_spec": spec, "feature_set": fset,
                            "dim": int(x_tr.shape[1]), "n_train": int(x_tr.shape[0]),
                            "n_test": int(x_te.shape[0]), "n_classes": n_classes,
                            "top1": round(top1, 5), "top5": round(top5, 5),
                            "chance": round(1.0 / n_classes, 5),
                        })
                del mean_feats, tok_feats

                if args.include_flat_dit:
                    rows.extend(_probe_flat_dit(
                        args, path, cond, layer, t_bin, masks, labels_np,
                        tr_idx, te_idx, n_classes, device,
                    ))
    return rows


def _probe_flat_dit(
    args: argparse.Namespace, path: Path, cond: str, layer: int, t_bin: int,
    masks: torch.Tensor, labels_np: np.ndarray, tr_idx: np.ndarray,
    te_idx: np.ndarray, n_classes: int, device: str,
) -> list[dict]:
    """One streaming pass per band spec for the 196608-dim flat DiT feature set."""
    warn("flat DiT features are T*d-dim (~196608); expect very large host/GPU memory use")
    rows = []
    labels = torch.from_numpy(labels_np).long()
    for spec in args.band_specs:
        flat_acc: list[torch.Tensor] = []
        with h5py.File(path, "r") as f:
            ds = f["activations"]
            n, t, _d = ds.shape
            gh = math.isqrt(t)
            for lo in range(0, n, args.chunk):
                x = torch.from_numpy(ds[lo:lo + args.chunk]).float().to(device)
                mask = spec_to_mask(spec, masks)
                if mask is not None:
                    x = filter_tokens_from_coeffs(dct2(tokens_to_grid(x, gh, gh)), mask)
                flat_acc.append(x.reshape(x.shape[0], -1).half().cpu())
        flat = torch.cat(flat_acc)
        del flat_acc
        top1, top5 = torch_probe(
            flat[tr_idx].float(), labels[tr_idx], flat[te_idx].float(), labels[te_idx],
            n_classes, device, args.epochs, args.lr, args.batch, args.weight_decay,
        )
        info(f"    {spec:9s} flat      dim={flat.shape[1]:6d} top1={top1:.4f} top5={top5:.4f}")
        rows.append({
            "source": "dit", "condition": cond, "layer": layer, "t_bin": t_bin,
            "band_spec": spec, "feature_set": "flat", "dim": int(flat.shape[1]),
            "n_train": len(tr_idx), "n_test": len(te_idx),
            "n_classes": n_classes, "top1": round(top1, 5), "top5": round(top5, 5),
            "chance": round(1.0 / n_classes, 5),
        })
        del flat
    return rows


def write_csv(out_dir: Path, new_rows: list[dict]) -> Path:
    """Append-merge metric rows into metrics/spectral_probe.csv (latent + dit share one file)."""
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "spectral_probe.csv"
    rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open() as fh:
            rows.extend(csv.DictReader(fh))
    rows.extend(new_rows)
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return csv_path


def plot_latent(rows: list[dict], band_specs: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    conds = sorted({r["condition"] for r in rows})
    fsets = sorted({r["feature_set"] for r in rows})
    styles = {fs: ls for fs, ls in zip(fsets, ["-", "--", ":", "-."], strict=False)}
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    ax.set_facecolor("white")
    x = np.arange(len(band_specs))
    for ci, cond in enumerate(conds):
        for fs in fsets:
            ys = []
            for spec in band_specs:
                m = [r for r in rows if r["condition"] == cond
                     and r["feature_set"] == fs and r["band_spec"] == spec]
                ys.append(float(m[0]["top1"]) if m else np.nan)
            ax.plot(x, ys, marker="o", ms=4, ls=styles[fs],
                    color=PALETTE_B[ci % len(PALETTE_B)], label=f"{cond} / {fs}")
    chance = float(rows[0]["chance"])
    ax.axhline(chance, ls="--", lw=0.8, color=PALETTE_B[3], label=f"chance ({chance:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(band_specs)
    ax.set_xlabel("band spec")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("E38 — latent class decodability per spectral band")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_dit(rows: list[dict], band_specs: list[str], layers: list[int],
             t_bins: list[int], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    conds = sorted({r["condition"] for r in rows})
    fsets = sorted({r["feature_set"] for r in rows})
    styles = {fs: ls for fs, ls in zip(fsets, ["-", "--", ":", "-."], strict=False)}
    nrows, ncols = len(t_bins), len(layers)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                             facecolor="white", sharey=True, squeeze=False)
    x = np.arange(len(band_specs))
    chance = float(rows[0]["chance"])
    for ti, t_bin in enumerate(t_bins):
        for li, layer in enumerate(layers):
            ax = axes[ti][li]
            ax.set_facecolor("white")
            for ci, cond in enumerate(conds):
                for fs in fsets:
                    ys = []
                    for spec in band_specs:
                        m = [r for r in rows if r["condition"] == cond
                             and r["feature_set"] == fs and r["band_spec"] == spec
                             and int(r["layer"]) == layer and int(r["t_bin"]) == t_bin]
                        ys.append(float(m[0]["top1"]) if m else np.nan)
                    ax.plot(x, ys, marker="o", ms=3, ls=styles[fs],
                            color=PALETTE_B[ci % len(PALETTE_B)], label=f"{cond} / {fs}")
            ax.axhline(chance, ls="--", lw=0.8, color=PALETTE_B[3])
            ax.set_xticks(x)
            ax.set_xticklabels(band_specs, fontsize=7, rotation=45)
            ax.set_title(f"L{layer}  t_bin={t_bin}", fontsize=9)
            ax.grid(alpha=0.25)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("E38 — DiT residual class decodability per spectral band")
    fig.supylabel("top-1 accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary_skeleton(out_dir: Path) -> None:
    path = out_dir / "reports" / "summary.md"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# E38 — Spectral Probing (class information per DCT octave band)\n\n"
        "> [!summary] TL;DR\n"
        "> **Per-band linear probes** of 1000-way class information in tokenizer latents and "
        "DiT residuals. Headline: ==TODO== (accuracy(band)/accuracy(broadband)). Caveat: TODO.\n\n"
        "## Setting\n\n"
        "- Bands: B0 (DC) + 4 radial DCT octaves on the 16x16 token grid; LP_k cumulative low-pass.\n"
        "- Probe: multinomial logistic (torch GPU), stratified 40/10 per class, seed 0 (E34a protocol).\n"
        "- Sources: tokenizer latents (mean_pool, flat) and DiT residual cells L{3,6,9} x t{0,1,2} "
        "(mean_pool, token).\n\n"
        "## Results\n\n"
        "- Latent: TODO (where does the class signal live spectrally?)\n"
        "- DiT: TODO (does class become band-distributed with depth?)\n\n"
        "## Artifacts\n\n"
        "- `metrics/spectral_probe.csv`\n"
        "- `plots/spectral_probe_latent.png`, `plots/spectral_probe_dit.png`\n"
    )


def write_repro(out_dir: Path, args: argparse.Namespace) -> None:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
    path = out_dir / "reproducibility.md"
    header = (
        "# Reproducibility — E38 spectral probing\n\n"
        f"- commit: `{sha}` (branch `{branch}`)\n"
        f"- inputs: latents `{LATENTS_ROOT}/<cond>/*.h5`; DiT activations "
        f"`{ACT_ROOT}/<cond>/step_{args.dit_step}/<layer>_<tbin>.h5`\n"
        "- env: CINECA Leonardo, `uv sync` venv, `module load cuda/12.2`, 1x A100 for full runs\n"
        "- framework: DCT-II ortho full-grid octave bands (diffmechint.spectral), "
        "torch logistic probe (E34a protocol)\n"
    )
    section = (
        f"\n## Invocation `--source {args.source}`\n\n"
        f"- command: `uv run python scripts/analysis/spectral_probe.py --source {args.source} "
        f"--out_dir {out_dir} --conditions {' '.join(args.conditions)} "
        f"--band_specs {' '.join(args.band_specs)} --feature_sets {' '.join(args.feature_sets)} "
        f"--n_train {args.n_train} --n_test {args.n_test} --token_cap {args.token_cap} "
        f"--epochs {args.epochs} --lr {args.lr} --batch {args.batch} "
        f"--weight_decay {args.weight_decay} --seed {args.seed}"
        + (f" --layers {' '.join(map(str, args.layers))} --t_bins {' '.join(map(str, args.t_bins))}"
           if args.source == "dit" else "")
        + (f" --max_images {args.max_images}" if args.max_images is not None else "")
        + (" --smoke" if args.smoke else "") + "`\n"
        f"- seed: {args.seed}; split: stratified {args.n_train}/{args.n_test} per class\n"
    )
    if path.exists():
        path.write_text(path.read_text() + section)
    else:
        path.write_text(header + section)
    (out_dir / "commit.txt").write_text(f"{sha} {branch} https://github.com/CerovazS/diffmechint\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["latent", "dit"], required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--band_specs", nargs="+", default=DEFAULT_BAND_SPECS,
                    help="e.g. broadband B0..B4 LP1..LP3")
    ap.add_argument("--feature_sets", nargs="+", default=None,
                    help="default: latent -> mean_pool flat; dit -> mean_pool token")
    ap.add_argument("--layers", nargs="+", type=int, default=[3, 6, 9])
    ap.add_argument("--t_bins", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dit_step", type=int, default=200_000)
    ap.add_argument("--patch_size", type=int, default=2)
    ap.add_argument("--n_train", type=int, default=40, help="images/class for train")
    ap.add_argument("--n_test", type=int, default=10, help="images/class for test")
    ap.add_argument("--token_cap", type=int, default=600_000,
                    help="max token-set samples (train+test) for the per-token probe")
    ap.add_argument("--include_flat_dit", action="store_true",
                    help="also probe 196608-dim flat DiT features (very memory hungry)")
    ap.add_argument("--chunk", type=int, default=2048, help="h5 streaming chunk (images)")
    ap.add_argument("--max_images", type=int, default=None,
                    help="cap loaded val images for the latent source (smoke / low memory)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="1 condition, bands {broadband,B1,B4}, mean_pool only, 5 epochs, latent only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warn("no CUDA — running on CPU (slow for full runs).")
    if args.smoke:
        if args.source != "latent":
            warn("--smoke forces --source latent")
            args.source = "latent"
        args.conditions = args.conditions[:1]
        args.band_specs = ["broadband", "B1", "B4"]
        args.feature_sets = ["mean_pool"]
        args.epochs = 5
        if args.max_images is None:
            args.max_images = 10_000
    if args.feature_sets is None:
        args.feature_sets = ["mean_pool", "flat"] if args.source == "latent" else ["mean_pool", "token"]

    rows = run_latent(args, device) if args.source == "latent" else run_dit(args, device)

    csv_path = write_csv(args.out_dir, rows)
    ok(f"wrote {csv_path} (+{len(rows)} rows)")
    if args.source == "latent":
        plot_latent(rows, args.band_specs, args.out_dir / "plots" / "spectral_probe_latent.png")
    else:
        plot_dit(rows, args.band_specs, args.layers, args.t_bins,
                 args.out_dir / "plots" / "spectral_probe_dit.png")
    write_summary_skeleton(args.out_dir)
    write_repro(args.out_dir, args)
    ok(f"done → {args.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
