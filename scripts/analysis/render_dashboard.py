"""Phase 4.5b — re-render `features.html` from per-feature JSONs (no SAE pass).

Useful after `feature_vlm_interp.py` or `recompute_histograms.py` have augmented
the per-feature JSON files: this regenerates `features.html` with the new
`vlm_interpretation` and `activation_histogram` fields exposed in the UI,
without re-running the SAE forward pass over the activation shards.

Wraps `feature_dashboard.py --rebuild_html_only <cell_dir>` so analysis-time
re-rendering has a dedicated entrypoint.

Usage::

    uv run python scripts/analysis/render_dashboard.py \\
        --cell_dir outputs/phase4_5b_feature_viz_ynull/repa_e_L3_T2

    # batch over every cell under a root:
    uv run python scripts/analysis/render_dashboard.py \\
        --root outputs/phase4_5b_feature_viz_ynull
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `feature_dashboard` importable whether this script is run from the repo
# root via `uv run python scripts/analysis/render_dashboard.py` or as a module.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from diffmechint.utils import error, info, ok, warn  # noqa: E402

# Reuse the canonical rebuild path inside feature_dashboard.py to avoid drift.
from feature_dashboard import _rebuild_html_only  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--cell_dir", type=Path, default=None,
                     help="One dashboard cell to re-render.")
    grp.add_argument("--root", type=Path, default=None,
                     help="Root containing `<cond>_L<L>_T<T>/` cells — render all.")
    args = p.parse_args()

    if args.cell_dir is not None:
        rc = _rebuild_html_only(args.cell_dir)
        return rc

    root: Path = args.root
    if not root.is_dir():
        error(f"root not a directory: {root}")
        return 1
    cells = sorted(p for p in root.iterdir()
                   if p.is_dir() and (p / "meta.json").exists())
    if not cells:
        warn(f"no cells with meta.json under {root}")
        return 1
    info(f"re-rendering {len(cells)} cells under {root}")
    n_ok, n_fail = 0, 0
    for c in cells:
        rc = _rebuild_html_only(c)
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
    ok(f"render_dashboard done: {n_ok} ok / {n_fail} failed")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
