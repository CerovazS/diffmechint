"""Plot SiT FID curves against validation-holdout reconstructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PALETTE_B = ["#335C67", "#E09F3E", "#9E2A2B", "#540B0E", "#3a6080", "#946030"]


def _load_curve(
    run: Path,
    label: str,
    *,
    n_samples: int | None,
    n_reference: int | None,
) -> pd.DataFrame | None:
    path = run / "metrics" / "validation" / "holdout_recon_fid.csv"
    if not path.exists():
        print(f"[skip] missing {path}")
        return None
    df = pd.read_csv(path)
    if n_samples is not None and "n_samples" in df.columns:
        df = df[df["n_samples"] == n_samples]
    if n_reference is not None and "n_reference" in df.columns:
        df = df[df["n_reference"] == n_reference]
    if df.empty:
        print(f"[skip] no matching rows in {path}")
        return None
    df = df.sort_values("step").copy()
    df["label"] = label
    df["run_dir"] = str(run)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/by_model/"
            "sit_l_2/analysis/holdout_recon_fid/plots"
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="SiT-L/2 FID vs training-holdout reconstructions",
    )
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n_reference", type=int, default=5000)
    parser.add_argument("--allow_any_recipe", action="store_true")
    args = parser.parse_args()

    if len(args.runs) != len(args.labels):
        raise SystemExit(f"got {len(args.runs)} runs but {len(args.labels)} labels")

    curves: list[pd.DataFrame] = []
    for run, label in zip(args.runs, args.labels, strict=True):
        df = _load_curve(
            run,
            label,
            n_samples=None if args.allow_any_recipe else args.n_samples,
            n_reference=None if args.allow_any_recipe else args.n_reference,
        )
        if df is not None:
            curves.append(df)
    if not curves:
        raise SystemExit("no holdout-reconstruction FID data found")

    all_df = pd.concat(curves, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.out_dir / "holdout_recon_fid_all.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = PALETTE_B[: len(curves)]
    for df, color in zip(curves, colors, strict=True):
        label = str(df["label"].iloc[0])
        for ax in axes:
            ax.plot(df["step"], df["fid"], marker="o", lw=1.8, ms=5, color=color, label=label)

    axes[0].set_title("Linear")
    axes[0].set_xlabel("global step")
    axes[0].set_ylabel("Clean-FID (5k generated vs 5k holdout recon)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Log")
    axes[1].set_xlabel("global step")
    axes[1].set_ylabel("Clean-FID")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend()

    fig.suptitle(args.title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_png = args.out_dir / "holdout_recon_fid_compare.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = []
    for df in curves:
        label = str(df["label"].iloc[0])
        best = df.loc[df["fid"].idxmin()]
        latest = df.loc[df["step"].idxmax()]
        summary.append(
            {
                "label": label,
                "n_points": len(df),
                "best_step": int(best["step"]),
                "best_fid": float(best["fid"]),
                "latest_step": int(latest["step"]),
                "latest_fid": float(latest["fid"]),
            }
        )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"wrote {out_png}")
    print(f"wrote {args.out_dir / 'holdout_recon_fid_all.csv'}")
    print(f"wrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
