"""Shared helpers for the 27-cell (condition x layer x t-bin) atlas grid (E11/E12/E26)."""

from __future__ import annotations

import re
from pathlib import Path

CELL_RE = re.compile(r"^(?P<cond>[a-z_0-9]+?)_L(?P<layer>\d+)_T(?P<tbin>\d+)$")

COND_ORDER = ["sd_vae", "repa_e", "eq_vae"]
COND_LABEL = {"sd_vae": "SD-VAE", "repa_e": "REPA-E", "eq_vae": "EQ-VAE"}
TBIN_ORDER = [0, 1, 2]

# Mono criterion matches diffmechint.analysis.dashboard.
MONO_DENSITY_MIN = 1e-4
MONO_DENSITY_MAX = 0.10
MONO_ENTROPY_MAX = 2.5


def parse_cell_name(name: str) -> tuple[str, int, int]:
    m = CELL_RE.match(name)
    if not m:
        raise ValueError(f"bad cell name: {name}")
    return m["cond"], int(m["layer"]), int(m["tbin"])


def select_cell_dirs(root: Path, conditions: set[str], layers: set[int]) -> list[Path]:
    """Sorted `<cond>_L<L>_T<T>` cell dirs under `root` matching the filters."""
    cell_dirs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not CELL_RE.match(d.name):
            continue
        cond, layer, _tbin = parse_cell_name(d.name)
        if cond in conditions and layer in layers:
            cell_dirs.append(d)
    return cell_dirs
