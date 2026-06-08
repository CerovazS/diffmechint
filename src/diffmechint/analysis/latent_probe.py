"""Linear probing of class information in tokenizer latent tokens (E34/E35)."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import h5py
import numpy as np
import torch

from diffmechint.sae.data_provider import patchify_latents
from diffmechint.utils import info, ok, warn

VAL_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents_val")
CONDITIONS = ["sd_vae", "eq_vae", "repa_e"]
FEATURE_SETS = ["token", "mean_pool", "max_pool", "flat"]


def load_val(cond: str, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (tokens (N,T,D), labels (N,)) for a condition's full val split."""
    shards = sorted(glob.glob(str(VAL_ROOT / cond / "*.h5")))
    if not shards:
        raise FileNotFoundError(f"no val shards under {VAL_ROOT / cond}")
    toks, labs = [], []
    for s in shards:
        with h5py.File(s, "r") as f:
            lat = torch.from_numpy(f["latents"][()]).float()  # (n,C,H,W)
            labs.append(torch.from_numpy(f["labels"][()]).long())
        toks.append(patchify_latents(lat, patch_size))  # (n,T,D)
    return torch.cat(toks), torch.cat(labs)


def stratified_split(labels: np.ndarray, n_train: int, n_test: int, seed: int):
    """Per-class indices: first n_train -> train, next n_test -> test."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        tr.extend(idx[:n_train])
        te.extend(idx[n_train:n_train + n_test])
    return np.asarray(tr), np.asarray(te)


def build_features(tokens: torch.Tensor, fset: str) -> torch.Tensor:
    """(N,T,D) -> per-image feature matrix for the requested set."""
    if fset == "mean_pool":
        return tokens.mean(dim=1)
    if fset == "max_pool":
        return tokens.amax(dim=1)
    if fset == "flat":
        return tokens.reshape(tokens.shape[0], -1)
    raise ValueError(fset)  # 'token' handled separately (per-token samples)


def gather_tokens(
    tokens: torch.Tensor, labels_np: np.ndarray, img_idx: np.ndarray,
    per_img: int, rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `per_img` random tokens from each image; label = image label."""
    t = tokens.shape[1]
    feats, labs = [], []
    for im in img_idx:
        sel = rng.choice(t, size=min(per_img, t), replace=False)
        feats.append(tokens[im, sel])
        labs.append(torch.full((len(sel),), int(labels_np[im])))
    return torch.cat(feats), torch.cat(labs)


@torch.no_grad()
def _topk_acc(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=1).indices
    return (topk == y[:, None]).any(dim=1).float().mean().item()


def torch_probe(
    x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, y_te: torch.Tensor,
    n_classes: int, device: str, epochs: int, lr: float, batch: int, wd: float,
) -> tuple[float, float]:
    """Multinomial logistic probe; returns (top1, top5) on the test split."""
    mu = x_tr.mean(0, keepdim=True)
    sd = x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr = ((x_tr - mu) / sd).to(device)
    x_te = ((x_te - mu) / sd).to(device)
    y_tr = y_tr.to(device)
    y_te = y_te.to(device)

    clf = torch.nn.Linear(x_tr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = x_tr.shape[0]
    for _ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(clf(x_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
    clf.eval()
    logits = clf(x_te)
    return _topk_acc(logits, y_te, 1), _topk_acc(logits, y_te, 5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--feature_sets", nargs="+", default=FEATURE_SETS)
    ap.add_argument("--patch_size", type=int, default=2)
    ap.add_argument("--n_train", type=int, default=40, help="images/class for train")
    ap.add_argument("--n_test", type=int, default=10, help="images/class for test")
    ap.add_argument("--token_cap", type=int, default=600_000,
                    help="max token-set samples (train+test) to keep the linear probe tractable")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 conditions, mean/flat only, 5 epochs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warn("no CUDA — running probe on CPU (slow).")
    if args.smoke:
        args.conditions = args.conditions[:1]
        args.feature_sets = ["mean_pool", "flat"]
        args.epochs = 5

    metrics_dir = args.out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cond in args.conditions:
        info(f"loading val latents: {cond}")
        tokens, labels = load_val(cond, args.patch_size)  # (N,T,D),(N,)
        N, T, D = tokens.shape
        labels_np = labels.numpy()
        n_classes = int(np.unique(labels_np).shape[0])
        tr_idx, te_idx = stratified_split(labels_np, args.n_train, args.n_test, args.seed)
        info(f"{cond}: N={N} T={T} D={D} classes={n_classes} "
             f"train={len(tr_idx)} test={len(te_idx)}")

        for fset in args.feature_sets:
            if fset == "token":
                # Per-token samples; cap total to keep the probe tractable.
                rng = np.random.default_rng(args.seed)
                per_img = max(1, args.token_cap // (len(tr_idx) + len(te_idx)))
                x_tr, y_tr = gather_tokens(tokens, labels_np, tr_idx, per_img, rng)
                x_te, y_te = gather_tokens(tokens, labels_np, te_idx, per_img, rng)
            else:
                feats = build_features(tokens, fset)  # (N, d)
                x_tr, y_tr = feats[tr_idx], labels[tr_idx]
                x_te, y_te = feats[te_idx], labels[te_idx]

            top1, top5 = torch_probe(
                x_tr.float(), y_tr, x_te.float(), y_te, n_classes,
                device, args.epochs, args.lr, args.batch, args.weight_decay,
            )
            info(f"  {fset:9s} dim={x_tr.shape[1]:5d} n_tr={x_tr.shape[0]:7d}  "
                 f"top1={top1:.4f} top5={top5:.4f}  (chance={1.0/n_classes:.4f})")
            rows.append({
                "condition": cond, "feature_set": fset, "patch_size": args.patch_size,
                "dim": int(x_tr.shape[1]), "n_train": int(x_tr.shape[0]),
                "n_test": int(x_te.shape[0]), "n_classes": n_classes,
                "top1": round(top1, 5), "top5": round(top5, 5),
                "chance": round(1.0 / n_classes, 5),
            })

    csv_path = metrics_dir / "latent_probe_accuracy.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok(f"wrote {csv_path} ({len(rows)} rows)")
    _plot(rows, args.out_dir / "plots" / "latent_probe_accuracy.png")
    ok("done")


def _plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PB = ["#335C67", "#E09F3E", "#9E2A2B", "#540B0E", "#FFF3B0"]
    fsets = sorted({r["feature_set"] for r in rows},
                   key=lambda s: FEATURE_SETS.index(s) if s in FEATURE_SETS else 99)
    conds = sorted({r["condition"] for r in rows})
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
    ax.set_facecolor("white")
    x = np.arange(len(fsets))
    width = 0.8 / max(1, len(conds))
    for ci, cond in enumerate(conds):
        vals = []
        for fs in fsets:
            m = [r for r in rows if r["condition"] == cond and r["feature_set"] == fs]
            vals.append(m[0]["top1"] if m else 0.0)
        ax.bar(x + (ci - (len(conds) - 1) / 2) * width, vals, width, label=cond, color=PB[ci % len(PB)])
    chance = rows[0]["chance"]
    ax.axhline(chance, ls="--", color=PB[3], label=f"chance ({chance:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(fsets)
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Latent linear probe: class decodability by pooling (SiT-B/2 token grid)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
