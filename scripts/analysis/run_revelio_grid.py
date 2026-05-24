"""Phase 5 driver — Revelio-style linear probe grid on one (condition, dit_step).

For each cell `(layer, t_bin)` in `activations_val/<cond>/step_<NNN>/`, train a
multinomial logistic-regression probe per concept axis and write the resulting
accuracy heatmap + JSON.

Concept axes default to the 6 we have without extra labelling:
  `object`, `animal_binary`, `broad_8`, `vehicle_binary`, `food_binary`,
  `instrument_binary` — see `src/diffmechint/probing/concepts.py`.

Usage:
    uv run python scripts/analysis/run_revelio_grid.py \\
        --condition eq_vae --dit_step 200000 \\
        --out_root outputs/phase5_revelio_grid
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diffmechint.probing.concepts import CONCEPTS, get_concept
from diffmechint.probing.revelio_grid import evaluate_grid, write_grid_result
from diffmechint.utils import error, info, ok

ACTIVATIONS_VAL = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_val"
)
LATENTS_VAL = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/latents_val"
)
DEFAULT_CONCEPTS = (
    "object",
    "animal_binary",
    "broad_8",
    "vehicle_binary",
    "food_binary",
    "instrument_binary",
)
LAYERS = (3, 6, 9)
T_BINS = (0, 1, 2)
T_CENTERS = {0: 0.025, 1: 0.20, 2: 0.50}

PB = {"win": "#335C67", "neutral": "#E09F3E", "loss": "#9E2A2B",
      "cream": "#FFF3B0", "dark": "#540B0E"}


def _resolve_val_labels(
    activation_shard_dir: Path, latents_root: Path, condition: str,
) -> np.ndarray:
    """Resolve the (N,) ImageNet labels for one (cond, dit_step) val activation
    dir. The Phase 3 val extraction skipped label storage in HDF5; this helper
    reads `manifest.json` (carries `sample_idx`) + the source latent shards
    (which DO carry per-image `labels`) and stitches them.
    """
    import h5py

    manifest = json.loads((activation_shard_dir / "manifest.json").read_text())
    sample_idx = np.asarray(manifest["sample_idx"], dtype=np.int64)

    shard_root = latents_root / condition
    shards = sorted(shard_root.glob("*.h5"))
    if not shards:
        msg = f"No latent shards under {shard_root}"
        raise FileNotFoundError(msg)
    all_labels_pieces: list[np.ndarray] = []
    for s in shards:
        with h5py.File(s, "r") as f:
            all_labels_pieces.append(np.asarray(f["labels"][()], dtype=np.int64))
    all_labels = np.concatenate(all_labels_pieces)

    if sample_idx.max() >= all_labels.shape[0]:
        msg = (f"sample_idx max {sample_idx.max()} exceeds available labels "
               f"({all_labels.shape[0]}) — manifest / latents mismatch.")
        raise IndexError(msg)
    return all_labels[sample_idx]


def _plot_heatmap(matrix: np.ndarray, *, title: str, num_classes: int,
                  baseline_acc: float, out_path: Path) -> None:
    """Render a 3×3 (layer × t_bin) accuracy heatmap with Palette B."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "pb_seq",
        [(0.0, PB["cream"]), (0.5, PB["neutral"]), (1.0, PB["win"])],
        N=256,
    )
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(matrix, cmap=cmap, vmin=baseline_acc, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(T_BINS)),
                  [f"T{t}\n{T_CENTERS[t]:.3f}" for t in T_BINS])
    ax.set_yticks(range(len(LAYERS)), [f"L{l}" for l in LAYERS])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center",
                        color="#444", fontsize=14)
                continue
            txtcolor = "#fafafa" if v > 0.7 else "#202020"
            ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                    color=txtcolor, fontsize=12, fontweight="bold")
    ax.set_xlabel("diffusion timestep bin")
    ax.set_ylabel("SAE residual-stream layer")
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="probe accuracy")
    cbar.ax.axhline(baseline_acc, color="#444", lw=0.6, ls=":")
    cbar.ax.text(
        1.05, baseline_acc, f"  chance ({baseline_acc*100:.1f}%)",
        transform=cbar.ax.get_yaxis_transform(),
        va="center", fontsize=8, color="#444",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", type=str, required=True,
                   choices=["sd_vae", "repa_e", "eq_vae"])
    p.add_argument("--dit_step", type=int, default=200_000)
    p.add_argument("--concepts", nargs="+", default=list(DEFAULT_CONCEPTS),
                   help="Concept axis names from CONCEPTS registry.")
    p.add_argument("--pool", type=str, default="tokens", choices=["tokens", "mean"])
    p.add_argument("--max_samples", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--activations_root", type=Path, default=ACTIVATIONS_VAL)
    args = p.parse_args()

    # Validate concept axes
    for name in args.concepts:
        if name not in CONCEPTS:
            error(f"unknown concept axis {name!r}; available: {sorted(CONCEPTS)}")
            return 1
        c = CONCEPTS[name]
        if not c.available:
            error(f"concept {name!r} marked unavailable (no labels). Skip or fix.")
            return 2

    shard_dir = args.activations_root / args.condition / f"step_{args.dit_step:06d}"
    if not shard_dir.exists():
        error(f"activation shard dir missing: {shard_dir}")
        return 3
    info(f"shard_dir={shard_dir}")

    # Phase 3 val extraction skipped label storage in HDF5; resolve them from
    # manifest.json + the source latent shards.
    info("resolving val labels from manifest + latent shards…")
    val_labels = _resolve_val_labels(shard_dir, LATENTS_VAL, args.condition)
    info(f"  resolved {val_labels.shape[0]} labels  unique={len(np.unique(val_labels))}")

    out_dir = args.out_root / f"{args.condition}__step_{args.dit_step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    info(f"out_dir={out_dir}")

    all_rows: list[dict] = []
    for cname in args.concepts:
        concept = get_concept(cname)
        info(f"--- concept: {cname} ({concept.num_classes}-way) ---")
        t0 = time.perf_counter()
        grid = evaluate_grid(
            shard_dir,
            concept=concept,
            layers=list(LAYERS),
            t_bins=list(T_BINS),
            pool=args.pool,
            max_samples=args.max_samples,
            seed=args.seed,
            labels_override=val_labels,
        )
        elapsed = time.perf_counter() - t0
        info(f"  grid done in {elapsed/60:.1f} min")

        # JSON
        json_path = out_dir / f"{cname}.json"
        write_grid_result(grid, json_path)

        # Heatmap
        mat = grid.matrix(list(LAYERS), list(T_BINS))
        chance = 1.0 / concept.num_classes
        title = (f"{args.condition}  step={args.dit_step//1000}k  •  "
                 f"{cname}  ({concept.num_classes}-way, chance {chance*100:.1f}%)")
        _plot_heatmap(mat, title=title, num_classes=concept.num_classes,
                      baseline_acc=chance,
                      out_path=plots_dir / f"{cname}.png")

        # Aggregate
        for cell in grid.cells:
            all_rows.append({
                "condition": args.condition,
                "dit_step": args.dit_step,
                "concept": cname,
                "num_classes": concept.num_classes,
                "chance_acc": chance,
                "layer": cell.layer,
                "t_bin": cell.t_bin,
                "t_center": T_CENTERS[cell.t_bin],
                "accuracy": cell.accuracy,
                "n_train": cell.n_train,
                "n_test": cell.n_test,
                "n_classes_actual": cell.n_classes,
                "pool_mode": cell.pool_mode,
            })
        peak = grid.peak()
        if peak is not None:
            ok(f"  [{cname}] peak L{peak.layer}/T{peak.t_bin} acc={peak.accuracy*100:.2f}%")

    # Master CSV
    if all_rows:
        csv_path = out_dir / "aggregate.csv"
        with csv_path.open("w", newline="") as fh:
            cols = list(all_rows[0].keys())
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        ok(f"Wrote {len(all_rows)} rows → {csv_path}")

    # Summary JSON
    summary = {
        "condition": args.condition,
        "dit_step": args.dit_step,
        "concepts": args.concepts,
        "layers": list(LAYERS),
        "t_bins": list(T_BINS),
        "t_centers": T_CENTERS,
        "pool": args.pool,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "n_cells": len(all_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    ok(f"Done. Outputs at {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
