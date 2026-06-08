"""Generate E04-style TopK vs Matryoshka plots on y-null activations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diffmechint.utils import write_csv

PB = {
    "win": "#335C67",
    "neutral": "#E09F3E",
    "loss": "#9E2A2B",
    "cream": "#FFF3B0",
    "deep_red": "#540B0E",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _load_eval_csv(path: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "sweep": label,
                "condition": r["condition"],
                "layer": int(r["layer"]),
                "t_bin": int(r["t_bin"]),
                "dit_step": int(r["dit_step"]),
                "stage": r["stage"],
                "k": int(float(r["k"])),
                "d_sae": int(float(r["d_sae"])),
                "ev": float(r["ev"]),
                "mse": float(r["mse"]),
                "recon_cosine": float(r["recon_cosine"]),
                "l0_mean": float(r["l0_mean"]),
                "live_features": int(float(r["live_features"])),
                "dead_features": int(float(r["dead_features"])),
                "dead_pct": float(r["dead_pct"]),
                "n_tokens": int(float(r["n_tokens"])),
                "ckpt_dir": r["ckpt_dir"],
                "val_shard": r["val_shard"],
            })
    return rows


def _matched_pairs(rows: list[dict], topk_label: str, matryo_label: str) -> list[tuple[dict, dict]]:
    by_key: dict[tuple[str, int, int, int], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r["stage"] != "final":
            continue
        key = (r["condition"], r["layer"], r["t_bin"], r["dit_step"])
        by_key[key][r["sweep"]] = r
    pairs = []
    for variants in by_key.values():
        if topk_label in variants and matryo_label in variants:
            pairs.append((variants[topk_label], variants[matryo_label]))
    return sorted(pairs, key=lambda p: (p[0]["condition"], p[0]["layer"], p[0]["t_bin"], p[0]["dit_step"]))


def _plot_ev_scatter(pairs: list[tuple[dict, dict]], out_dir: Path, topk_label: str, matryo_label: str) -> None:
    xs = np.asarray([p[0]["ev"] for p in pairs])
    ys = np.asarray([p[1]["ev"] for p in pairs])
    win = ys > xs
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs[win], ys[win], s=34, c=PB["win"], alpha=0.82,
               label=f"Matryoshka wins (n={int(win.sum())})", zorder=3)
    ax.scatter(xs[~win], ys[~win], s=34, c=PB["loss"], alpha=0.82,
               label=f"TopK wins (n={int((~win).sum())})", zorder=3)
    lo = min(float(xs.min()), float(ys.min())) - 0.01
    hi = min(1.0, max(float(xs.max()), float(ys.max())) + 0.01)
    ax.plot((lo, hi), (lo, hi), ls="--", c="#444444", lw=0.8, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"y-null EV - {topk_label}")
    ax.set_ylabel(f"y-null EV - {matryo_label}")
    ax.set_title(f"Head-to-head y-null EV at DiT step 200k (n={len(pairs)})")
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "ynull_ev_scatter_step200k.png", dpi=180)
    plt.close(fig)


def _plot_ev_scatter_compat(pairs: list[tuple[dict, dict]], out_dir: Path, topk_label: str, matryo_label: str) -> None:
    """E04-compatible scatter filename and visual style."""
    xs = np.asarray([p[0]["ev"] for p in pairs])
    ys = np.asarray([p[1]["ev"] for p in pairs])
    win = ys > xs
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs[win], ys[win], s=22, c=PB["win"], alpha=0.75,
               label=f"Matryoshka wins (n={int(win.sum())})", zorder=3)
    ax.scatter(xs[~win], ys[~win], s=22, c=PB["loss"], alpha=0.75,
               label=f"TopK wins (n={int((~win).sum())})", zorder=3)
    lim = (min(float(xs.min()), float(ys.min())) - 0.02, 1.0)
    ax.plot(lim, lim, ls="--", c="#444444", lw=0.8, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"y-null val EV - {topk_label}")
    ax.set_ylabel(f"y-null val EV - {matryo_label}")
    ax.set_title(f"Head-to-head y-null val EV  (DiT step 200k, n={len(xs)})")
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "val_ev_scatter.png", dpi=160)
    plt.close(fig)


def _plot_grouped_bars(
    pairs: list[tuple[dict, dict]],
    out_dir: Path,
    metric: str,
    ylabel: str,
    fname: str,
    *,
    scale: float = 1.0,
    ylim: tuple[float, float] | None = None,
) -> None:
    labels = ["TopK k=128", "Matryoshka k=256"]
    by_group: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for topk, matryo in pairs:
        key = (topk["condition"], topk["layer"])
        by_group[key][labels[0]].append(topk[metric] * scale)
        by_group[key][labels[1]].append(matryo[metric] * scale)

    conds = sorted({k[0] for k in by_group})
    layers = sorted({k[1] for k in by_group})
    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4), sharey=True)
    if len(conds) == 1:
        axes = [axes]
    x = np.arange(len(layers))
    width = 0.36
    colors = {labels[0]: PB["loss"], labels[1]: PB["win"]}

    for ax, cond in zip(axes, conds):
        for i, label in enumerate(labels):
            vals = [np.mean(by_group[(cond, layer)][label]) for layer in layers]
            ax.bar(x + (i - 0.5) * width, vals, width=width, color=colors[label],
                   edgecolor="#202020", linewidth=0.6, label=label if ax is axes[0] else None)
        ax.set_xticks(x, [f"L{layer}" for layer in layers])
        ax.set_title(cond)
        ax.set_xlabel("SAE residual-stream layer")
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="lower right" if "EV" in ylabel else "upper right", framealpha=0.92, fontsize=9)
    fig.suptitle(f"{ylabel} on y-null activations (DiT step 200k, mean over T0/T1/T2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=180)
    plt.close(fig)


def _plot_grouped_bars_compat(
    pairs: list[tuple[dict, dict]],
    out_dir: Path,
    metric: str,
    ylabel: str,
    fname: str,
    *,
    scale: float = 1.0,
    ylim: tuple[float, float] | None = None,
) -> None:
    """E04-compatible grouped bars: same filenames and compact titles."""
    labels = ["TopK k=128 d=32k", "Matryoshka k=256 d=32k"]
    by_group: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for topk, matryo in pairs:
        key = (topk["condition"], topk["layer"])
        by_group[key][labels[0]].append(topk[metric] * scale)
        by_group[key][labels[1]].append(matryo[metric] * scale)

    conds = sorted({k[0] for k in by_group})
    layers = sorted({k[1] for k in by_group})
    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4), sharey=True)
    if len(conds) == 1:
        axes = [axes]
    x = np.arange(len(layers))
    width = 0.36
    colors = {labels[0]: PB["loss"], labels[1]: PB["win"]}

    for ax, cond in zip(axes, conds):
        for i, label in enumerate(labels):
            vals = [np.mean(by_group[(cond, layer)][label]) for layer in layers]
            ax.bar(x + (i - 0.5) * width, vals, width=width, color=colors[label],
                   edgecolor="#202020", linewidth=0.6, label=label if ax is axes[0] else None)
        ax.set_xticks(x, [f"L{layer}" for layer in layers])
        ax.set_title(cond)
        ax.set_xlabel("SAE residual-stream layer")
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="lower right" if "EV" in ylabel else "upper right", framealpha=0.92, fontsize=9)
    fig.suptitle(f"{ylabel}   (y-null activations, DiT step 200k, mean over t-bin)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=160)
    plt.close(fig)


def _plot_single_step_trajectory_compat(
    pairs: list[tuple[dict, dict]],
    out_dir: Path,
    topk_label: str,
    matryo_label: str,
) -> None:
    """E04-compatible 3x3 trajectory panel for the single y-null DiT step.

    The original E04 plot has a DiT-checkpoint trajectory. y-null activations
    were extracted only at step 200k, so this preserves the panel semantics but
    shows one point per variant, averaged over the three diffusion-time bins.
    """
    by_group: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for topk, matryo in pairs:
        key = (topk["condition"], topk["layer"])
        by_group[key][topk_label].append(topk["ev"])
        by_group[key][matryo_label].append(matryo["ev"])

    conds = sorted({k[0] for k in by_group})
    layers = sorted({k[1] for k in by_group})
    fig, axes = plt.subplots(len(layers), len(conds), figsize=(4.5 * len(conds), 3 * len(layers)),
                             sharex=True, sharey=False)
    styles = {
        topk_label: dict(color=PB["loss"], marker="o", linestyle="-"),
        matryo_label: dict(color=PB["win"], marker="s", linestyle="-"),
    }
    x = [200000]
    for i, layer in enumerate(layers):
        for j, cond in enumerate(conds):
            ax = axes[i, j]
            for label, style in styles.items():
                vals = by_group[(cond, layer)].get(label)
                if vals:
                    ax.plot(x, [float(np.mean(vals))], label=label if (i == 0 and j == 0) else None, **style)
            ax.set_xlim(180000, 220000)
            ax.set_ylim(0.86, 1.0)
            ax.set_title(f"{cond}  -  L{layer}", fontsize=9)
            if j == 0:
                ax.set_ylabel("mean y-null val EV")
            if i == len(layers) - 1:
                ax.set_xlabel("DiT checkpoint step")
    axes[0, 0].legend(loc="lower right", framealpha=0.92, fontsize=8)
    fig.suptitle("y-null val EV at DiT checkpoint step 200k (mean over t-bin)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "val_ev_vs_dit_step.png", dpi=160)
    plt.close(fig)


def _write_e04_compat_plots(pairs: list[tuple[dict, dict]], out_dir: Path, topk_label: str, matryo_label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_ev_scatter_compat(pairs, out_dir, topk_label, matryo_label)
    _plot_grouped_bars_compat(
        pairs, out_dir, "ev", "mean y-null val EV",
        "val_ev_grouped_bars.png", ylim=(0.7, 1.0),
    )
    _plot_grouped_bars_compat(
        pairs, out_dir, "dead_pct", "mean y-null dead %",
        "dead_pct_grouped_bars.png", scale=100.0, ylim=(0, 80),
    )
    _plot_single_step_trajectory_compat(pairs, out_dir, topk_label, matryo_label)


def _plot_delta_heatmap(pairs: list[tuple[dict, dict]], out_dir: Path, metric: str, title: str, fname: str, scale: float) -> None:
    conds = sorted({p[0]["condition"] for p in pairs})
    layers = sorted({p[0]["layer"] for p in pairs})
    tbins = sorted({p[0]["t_bin"] for p in pairs})
    fig, axes = plt.subplots(1, len(conds), figsize=(4.8 * len(conds), 3.6), sharey=True)
    if len(conds) == 1:
        axes = [axes]
    deltas = {(a["condition"], a["layer"], a["t_bin"]): (b[metric] - a[metric]) * scale for a, b in pairs}
    vmax = max(abs(v) for v in deltas.values())
    for ax, cond in zip(axes, conds):
        mat = np.asarray([[deltas[(cond, layer, t)] for t in tbins] for layer in layers])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(cond)
        ax.set_xticks(range(len(tbins)), [f"T{t}" for t in tbins])
        ax.set_yticks(range(len(layers)), [f"L{layer}" for layer in layers])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.82)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_dir / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_summaries(pairs: list[tuple[dict, dict]], out_dir: Path) -> None:
    deltas_ev = np.asarray([b["ev"] - a["ev"] for a, b in pairs])
    deltas_dead = np.asarray([b["dead_pct"] - a["dead_pct"] for a, b in pairs])
    topk_ev = np.asarray([a["ev"] for a, _ in pairs])
    matryo_ev = np.asarray([b["ev"] for _, b in pairs])
    topk_dead = np.asarray([a["dead_pct"] for a, _ in pairs])
    matryo_dead = np.asarray([b["dead_pct"] for _, b in pairs])
    head = {
        "n_pairs": len(pairs),
        "dit_steps": sorted({a["dit_step"] for a, _ in pairs}),
        "matryoshka_ev_wins": int((deltas_ev > 0).sum()),
        "topk_ev_wins": int((deltas_ev <= 0).sum()),
        "mean_delta_ev_matryo_minus_topk": float(deltas_ev.mean()),
        "median_delta_ev_matryo_minus_topk": float(np.median(deltas_ev)),
        "mean_topk_ev": float(topk_ev.mean()),
        "mean_matryoshka_ev": float(matryo_ev.mean()),
        "mean_topk_dead_pct": float(topk_dead.mean()),
        "mean_matryoshka_dead_pct": float(matryo_dead.mean()),
        "mean_delta_dead_pct_matryo_minus_topk": float(deltas_dead.mean()),
        "max_delta_dead_pct_matryo_minus_topk": float(deltas_dead.max()),
        "min_delta_dead_pct_matryo_minus_topk": float(deltas_dead.min()),
    }
    (out_dir / "headline.json").write_text(json.dumps(head, indent=2))

    by_group: dict[tuple[str, int], list[tuple[dict, dict]]] = defaultdict(list)
    for pair in pairs:
        by_group[(pair[0]["condition"], pair[0]["layer"])].append(pair)
    group_rows = []
    for (cond, layer), group in sorted(by_group.items()):
        group_rows.append({
            "condition": cond,
            "layer": layer,
            "n_t_bins": len(group),
            "topk_ev_mean": float(np.mean([a["ev"] for a, _ in group])),
            "matryoshka_ev_mean": float(np.mean([b["ev"] for _, b in group])),
            "delta_ev_mean": float(np.mean([b["ev"] - a["ev"] for a, b in group])),
            "topk_dead_pct_mean": float(np.mean([a["dead_pct"] for a, _ in group])),
            "matryoshka_dead_pct_mean": float(np.mean([b["dead_pct"] for _, b in group])),
            "delta_dead_pct_mean": float(np.mean([b["dead_pct"] - a["dead_pct"] for a, b in group])),
        })
    write_csv(out_dir / "summary_by_condition_layer.csv", group_rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--topk_csv", type=Path, required=True)
    p.add_argument("--matryoshka_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--e04_compat_out_dir", type=Path, default=None,
                   help="Optional directory where E04-compatible plot filenames "
                        "(val_ev_scatter.png, val_ev_grouped_bars.png, "
                        "dead_pct_grouped_bars.png, val_ev_vs_dit_step.png) "
                        "are written for Flywheel artifact replacement.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    topk_label = "TopK k=128 d=32k"
    matryo_label = "Matryoshka k=256 d=32k"
    rows = _load_eval_csv(args.topk_csv, topk_label) + _load_eval_csv(args.matryoshka_csv, matryo_label)
    rows = [r for r in rows if r["dit_step"] == 200000 and r["stage"] == "final"]
    write_csv(args.out_dir / "combined_ynull_step200k.csv", rows)

    pairs = _matched_pairs(rows, topk_label, matryo_label)
    if len(pairs) != 27:
        raise RuntimeError(f"Expected 27 matched pairs, got {len(pairs)}")
    _write_summaries(pairs, args.out_dir)
    _plot_ev_scatter(pairs, args.out_dir, topk_label, matryo_label)
    _plot_grouped_bars(
        pairs, args.out_dir, "ev", "mean y-null validation EV",
        "ynull_ev_grouped_bars_step200k.png", ylim=(0.7, 1.0),
    )
    _plot_grouped_bars(
        pairs, args.out_dir, "dead_pct", "mean y-null dead %",
        "ynull_dead_pct_grouped_bars_step200k.png", scale=100.0, ylim=(0, 80),
    )
    _plot_delta_heatmap(
        pairs, args.out_dir, "ev",
        "Delta y-null EV: Matryoshka minus TopK (DiT step 200k)",
        "ynull_delta_ev_heatmap_step200k.png", scale=1.0,
    )
    _plot_delta_heatmap(
        pairs, args.out_dir, "dead_pct",
        "Delta y-null dead percentage points: Matryoshka minus TopK (DiT step 200k)",
        "ynull_delta_dead_heatmap_step200k.png", scale=100.0,
    )
    if args.e04_compat_out_dir is not None:
        _write_e04_compat_plots(pairs, args.e04_compat_out_dir, topk_label, matryo_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
