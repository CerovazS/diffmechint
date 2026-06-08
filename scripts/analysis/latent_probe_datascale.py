"""E34a control — does the latent probe accuracy saturate with training data?

Falsification check for the "low accuracy is just data-starvation" objection:
trains the 1000-way latent probe on the TRAIN split (sweeping images/class)
and tests on the VAL split, for the per-image feature sets that have only
40 samples/class in the main E34a run. If accuracy keeps climbing with N it
was data-limited; if it plateaus the linear signal is genuinely that weak.

Outputs (under --out_dir/metrics): latent_probe_datascale.csv + plot.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import h5py
import numpy as np
import torch

from diffmechint.analysis.latent_probe import build_features, gather_tokens, torch_probe
from diffmechint.sae.data_provider import patchify_latents
from diffmechint.utils import info, ok, warn

TRAIN_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents")
VAL_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents_val")
CONDITIONS = ["sd_vae", "eq_vae", "repa_e"]


def gather_train(cond: str, patch_size: int, max_per_class: int, n_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream train shards, keep up to `max_per_class` images per class.

    Returns (tokens (M,T,D) fp16, labels (M,)). Classes sit in contiguous
    shard blocks, so a per-class reservoir over a single pass suffices.
    """
    shards = sorted(glob.glob(str(TRAIN_ROOT / cond / "*.h5")))
    if not shards:
        raise FileNotFoundError(f"no train shards under {TRAIN_ROOT / cond}")
    counts = np.zeros(n_classes, dtype=np.int64)
    keep_tok: list[torch.Tensor] = []
    keep_lab: list[int] = []
    for s in shards:
        with h5py.File(s, "r") as f:
            lab = f["labels"][()].astype(np.int64)
            need = np.array([counts[c] < max_per_class for c in lab])
            if not need.any():
                continue
            lat = torch.from_numpy(f["latents"][need.nonzero()[0]]).float()
        tok = patchify_latents(lat, patch_size).half()  # (k,T,D)
        for i, c in enumerate(lab[need]):
            if counts[c] >= max_per_class:
                continue
            keep_tok.append(tok[i])
            keep_lab.append(int(c))
            counts[c] += 1
        if (counts >= max_per_class).all():
            break
    info(f"{cond}: gathered {len(keep_lab)} train images "
         f"(min/class={counts.min()}, max/class={counts.max()})")
    return torch.stack(keep_tok), torch.tensor(keep_lab, dtype=torch.long)


def load_val(cond: str, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    shards = sorted(glob.glob(str(VAL_ROOT / cond / "*.h5")))
    toks, labs = [], []
    for s in shards:
        with h5py.File(s, "r") as f:
            lat = torch.from_numpy(f["latents"][()]).float()
            labs.append(torch.from_numpy(f["labels"][()]).long())
        toks.append(patchify_latents(lat, patch_size).half())
    return torch.cat(toks), torch.cat(labs)


def subsample_per_class(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        rng.shuffle(ci)
        idx.extend(ci[:n])
    return np.asarray(idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--feature_sets", nargs="+", default=["flat", "mean_pool", "token"])
    ap.add_argument("--patch_size", type=int, default=2)
    ap.add_argument("--n_train_grid", type=int, nargs="+", default=[40, 100, 200, 400])
    ap.add_argument("--token_cap", type=int, default=600_000)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warn("no CUDA — CPU probe (slow).")
    if args.smoke:
        args.conditions = args.conditions[:1]
        args.feature_sets = ["mean_pool"]
        args.n_train_grid = [40, 100]
        args.epochs = 5

    metrics_dir = args.out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    max_n = max(args.n_train_grid)
    rows = []

    for cond in args.conditions:
        val_tok, val_lab = load_val(cond, args.patch_size)
        n_classes = int(np.unique(val_lab.numpy()).shape[0])
        tr_tok, tr_lab = gather_train(cond, args.patch_size, max_n, n_classes)
        tr_lab_np = tr_lab.numpy()

        for fset in args.feature_sets:
            for n in args.n_train_grid:
                sel = subsample_per_class(tr_lab_np, n, args.seed)
                if fset == "token":
                    rng = np.random.default_rng(args.seed)
                    per_img = max(1, args.token_cap // (len(sel) + val_tok.shape[0]))
                    x_tr, y_tr = gather_tokens(tr_tok, tr_lab_np, sel, per_img, rng)
                    x_te, y_te = gather_tokens(val_tok, val_lab.numpy(),
                                               np.arange(val_tok.shape[0]), per_img, rng)
                else:
                    x_tr = build_features(tr_tok[sel].float(), fset)
                    y_tr = tr_lab[sel]
                    x_te = build_features(val_tok.float(), fset)
                    y_te = val_lab
                top1, top5 = torch_probe(
                    x_tr.float(), y_tr, x_te.float(), y_te, n_classes,
                    device, args.epochs, args.lr, args.batch, args.weight_decay,
                )
                info(f"{cond:7s} {fset:9s} N={n:4d}/class  n_tr={x_tr.shape[0]:7d}  "
                     f"top1={top1:.4f} top5={top5:.4f}")
                rows.append({
                    "condition": cond, "feature_set": fset, "n_train_per_class": n,
                    "n_train": int(x_tr.shape[0]), "n_test": int(x_te.shape[0]),
                    "n_classes": n_classes, "top1": round(top1, 5), "top5": round(top5, 5),
                    "chance": round(1.0 / n_classes, 5),
                })

    csv_path = metrics_dir / "latent_probe_datascale.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok(f"wrote {csv_path} ({len(rows)} rows)")
    _plot(rows, args.out_dir / "plots" / "latent_probe_datascale.png")
    ok("done")


def _plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PB = ["#335C67", "#E09F3E", "#9E2A2B", "#540B0E", "#FFF3B0"]
    fsets = sorted({r["feature_set"] for r in rows})
    conds = sorted({r["condition"] for r in rows})
    fig, axes = plt.subplots(1, len(fsets), figsize=(5 * len(fsets), 4.3),
                             facecolor="white", squeeze=False)
    for col, fs in enumerate(fsets):
        ax = axes[0][col]
        ax.set_facecolor("white")
        for ci, cond in enumerate(conds):
            pts = sorted([r for r in rows if r["feature_set"] == fs and r["condition"] == cond],
                         key=lambda r: r["n_train_per_class"])
            ax.plot([p["n_train_per_class"] for p in pts], [p["top1"] for p in pts],
                    "o-", color=PB[ci % len(PB)], label=cond)
        ax.axhline(rows[0]["chance"], ls="--", color=PB[3], lw=1, label="chance")
        ax.set_xlabel("train images / class")
        ax.set_ylabel("top-1 accuracy")
        ax.set_title(fs)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Latent probe accuracy vs training-set size (test on val)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
