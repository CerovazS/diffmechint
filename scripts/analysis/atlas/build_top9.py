"""Extract top-9 (class_idx, dataset_idx) arrays for every live feature, per cell.

Second pass over outputs/phase4_5b_feature_viz_ynull/<cell>/features/feature_*.json.
Writes:

    outputs/phase4_8_atlas/top9/<cell>.npz
        feature_ids      (N,)      int32
        is_mono          (N,)      bool
        top9_class_idx   (N, 9)    int16   (-1 = missing slot)
        top9_dataset_idx (N, 9)    int32   (-1 = missing slot)

These compact arrays are the canonical input for class-coverage and Hungarian
matching analyses — far smaller than the original JSONs.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ATLAS_ROOT = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/phase4_5b_feature_viz_ynull")
OUT_ROOT = Path("/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/outputs/phase4_8_atlas/top9")

CELL_RE = re.compile(r"^(?P<cond>[a-z_0-9]+?)_L(?P<layer>\d+)_T(?P<tbin>\d+)$")
MONO_DENSITY_MIN = 1e-4
MONO_DENSITY_MAX = 0.10
MONO_ENTROPY_MAX = 2.5


def process_cell(cell_dir_str: str) -> tuple[str, int]:
    cell_dir = Path(cell_dir_str)
    cell = cell_dir.name
    out_path = OUT_ROOT / f"{cell}.npz"
    if out_path.exists():
        return cell, 0

    fids: list[int] = []
    is_mono_list: list[bool] = []
    top9_class: list[list[int]] = []
    top9_data: list[list[int]] = []

    for f in sorted((cell_dir / "features").iterdir()):
        if not (f.name.startswith("feature_") and f.name.endswith(".json")):
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not d.get("live"):
            continue

        ent = d.get("entropy")
        dens = float(d.get("density", 0.0))
        ent_ok = ent is not None and np.isfinite(ent) and ent < MONO_ENTROPY_MAX
        dens_ok = MONO_DENSITY_MIN <= dens <= MONO_DENSITY_MAX
        is_mono = bool(ent_ok and dens_ok)

        top = d.get("top") or []
        cls = [int(t.get("class_idx", -1)) for t in top[:9]]
        ds = [int(t.get("dataset_idx", -1)) for t in top[:9]]
        # Pad to 9 if missing.
        while len(cls) < 9:
            cls.append(-1)
            ds.append(-1)

        fids.append(int(d["feature_id"]))
        is_mono_list.append(is_mono)
        top9_class.append(cls)
        top9_data.append(ds)

    arr_fids = np.asarray(fids, dtype=np.int32)
    arr_mono = np.asarray(is_mono_list, dtype=bool)
    arr_cls = np.asarray(top9_class, dtype=np.int16) if fids else np.zeros((0, 9), dtype=np.int16)
    arr_ds = np.asarray(top9_data, dtype=np.int32) if fids else np.zeros((0, 9), dtype=np.int32)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        feature_ids=arr_fids,
                        is_mono=arr_mono,
                        top9_class_idx=arr_cls,
                        top9_dataset_idx=arr_ds)
    return cell, len(fids)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cell_dirs = sorted(d for d in ATLAS_ROOT.iterdir()
                       if d.is_dir() and CELL_RE.match(d.name))
    print(f"processing {len(cell_dirs)} cells (parallel workers=8)...")
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=8) as ex:
        fut_to_dir = {ex.submit(process_cell, str(d)): d for d in cell_dirs}
        for i, fut in enumerate(as_completed(fut_to_dir), start=1):
            cell, n = fut.result()
            print(f"  [{i:2d}/{len(cell_dirs)}] {cell:22s}  n_live={n}")

    print(f"total time: {(time.perf_counter() - t0):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
