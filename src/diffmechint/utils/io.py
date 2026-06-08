"""Shared filesystem and CSV helpers for analysis/eval entry points."""

from __future__ import annotations

import csv
import time
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of string dicts."""
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write dict rows to CSV.

    Without explicit `fieldnames` the header is the ordered union of row keys
    and empty input raises; with `fieldnames`, missing keys become "" and an
    empty input still writes a header-only file.
    """
    if fieldnames is None:
        if not rows:
            raise ValueError(f"refusing to write empty CSV: {path}")
        fieldnames = []
        for row in rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_run_dir(out_root: Path, run_id: str | None, *, resume: bool = False) -> Path:
    """Create a unique per-run output dir with metrics/plots/artifacts subdirs."""
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / run_id
    if out_dir.exists() and not resume:
        raise FileExistsError(f"run directory already exists: {out_dir}")
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "artifacts").mkdir(exist_ok=True)
    return out_dir


def write_summary_md(out_dir: Path, title: str, lines: list[str]) -> None:
    """Write `summary.md` opening with the mandatory TL;DR callout."""
    payload = [f"# {title}", "", "> [!summary] TL;DR"]
    payload.extend(f"> {line}" for line in lines[:4])
    payload.extend(["", *lines, ""])
    (out_dir / "summary.md").write_text("\n".join(payload), encoding="utf-8")
