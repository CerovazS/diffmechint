"""Convert Lightning's wide CSV into the standardized `metrics/{train,validation}/` layout.

Input:  <run>/lightning_logs/version_0/metrics.csv
Output: <run>/metrics/train/loss_step.csv
        <run>/metrics/train/loss_epoch.csv
        <run>/metrics/validation/loss.csv
        <run>/metrics/validation/fid.csv     (empty/append-only — populated by post_hoc_fid.py)
        <run>/metrics/summary.json

Idempotent. Safe to call mid-training (Lightning keeps appending to its CSV).
Usage:
    uv run python scripts/extract_metrics.py <run_dir>
    uv run python scripts/extract_metrics.py /path/to/runs/sit_b2_<tok>_full_<TS>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def extract(run_dir: Path) -> None:
    src_csv = run_dir / "lightning_logs" / "version_0" / "metrics.csv"
    if not src_csv.exists():
        print(f"  [skip] no metrics.csv in {run_dir.name}")
        return

    df = pd.read_csv(src_csv)

    metrics_root = run_dir / "metrics"
    train_dir = metrics_root / "train"
    val_dir = metrics_root / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Per-step training loss (most rows).
    if "train/loss_step" in df.columns:
        d = df.dropna(subset=["train/loss_step"])[["epoch", "step", "train/loss_step"]].copy()
        d.columns = ["epoch", "step", "loss"]
        d["step"] = d["step"].astype(int)
        d.to_csv(train_dir / "loss_step.csv", index=False)
        n_train_step = len(d)
    else:
        n_train_step = 0

    # Per-epoch training loss.
    if "train/loss_epoch" in df.columns:
        d = df.dropna(subset=["train/loss_epoch"])[["epoch", "step", "train/loss_epoch"]].copy()
        d.columns = ["epoch", "step", "loss"]
        d["step"] = d["step"].astype(int)
        d.to_csv(train_dir / "loss_epoch.csv", index=False)
        n_train_epoch = len(d)
    else:
        n_train_epoch = 0

    # Validation loss.
    if "val/loss" in df.columns:
        d = df.dropna(subset=["val/loss"])[["epoch", "step", "val/loss"]].copy()
        d.columns = ["epoch", "step", "loss"]
        d["step"] = d["step"].astype(int)
        d.to_csv(val_dir / "loss.csv", index=False)
        n_val = len(d)
    else:
        n_val = 0

    # FID file: create empty header if not yet present (post_hoc_fid.py appends).
    fid_path = val_dir / "fid.csv"
    if not fid_path.exists():
        fid_path.write_text("step,n_samples,cfg_scale,fid\n")

    summary = {
        "run_id": run_dir.name,
        "lightning_csv": str(src_csv),
        "n_train_step_rows": n_train_step,
        "n_train_epoch_rows": n_train_epoch,
        "n_val_rows": n_val,
        "max_epoch": int(df["epoch"].max()) if len(df) else None,
        "max_step": int(df["step"].max()) if len(df) else None,
    }
    (metrics_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"  ✓ {run_dir.name}: train_step={n_train_step}, train_epoch={n_train_epoch}, val={n_val}, max_step={summary['max_step']}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    runs = [Path(p) for p in sys.argv[1:]]
    for r in runs:
        if not r.is_dir():
            print(f"  [skip] not a directory: {r}")
            continue
        extract(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
