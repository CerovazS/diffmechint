"""Bridge: E31 latent atlas CSV → dashboard-format cell dir for VLM captioning.

Writes `features/feature_<j>.json` per live latent feature in the format
`feature_vlm_interp.py` consumes (top entries carry synset+filename so the
VLM grid builder reads ImageNet-val originals directly; no thumbs needed).
Also writes `live_feature_ids.txt` for the `--features` explicit list.

Usage:
    uv run python scripts/analysis/latent_atlas_to_dashboard.py \
        --atlas_dir outputs/phase4_20_.../metrics --out_base outputs/phase4_20_.../feature_viz
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from diffmechint.utils import info, ok

IMAGEFOLDER_ROOT = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/imagenet_val_imagefolder"
)


def build_samples() -> list[tuple[str, str]]:
    """(synset, filename) in torchvision ImageFolder order (sorted dirs/files)."""
    samples = []
    for syn_dir in sorted(p for p in IMAGEFOLDER_ROOT.iterdir() if p.is_dir()):
        for f in sorted(p.name for p in syn_dir.iterdir()):
            samples.append((syn_dir.name, f))
    return samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas_dir", type=Path, required=True)
    p.add_argument("--out_base", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", default=["sd_vae", "eq_vae", "repa_e"])
    args = p.parse_args()

    samples = build_samples()
    info(f"ImageFolder index: {len(samples)} samples")
    assert len(samples) == 50_000, "unexpected ImageNet-val size"

    for cond in args.conditions:
        rows = list(csv.DictReader((args.atlas_dir / f"latent_atlas_{cond}.csv").open()))
        cell_dir = args.out_base / f"latent_{cond}"
        feat_dir = cell_dir / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        live_ids = []
        for r in rows:
            if r["live"] != "1":
                continue
            fid = int(r["feature"])
            live_ids.append(fid)
            idxs = [int(x) for x in r["top9_image_idx"].split()]
            labels = [int(x) for x in r["top9_labels"].split()]
            acts = [float(x) for x in r["top9_activations"].split()]
            top = [
                {
                    "image_local_idx": i,
                    "class_idx": lab,
                    "synset": samples[i][0],
                    "filename": samples[i][1],
                    "activation": a,
                }
                for i, lab, a in zip(idxs, labels, acts, strict=True)
            ]
            (feat_dir / f"feature_{fid}.json").write_text(json.dumps({
                "feature_id": fid,
                "live": True,
                "density": float(r["density"]),
                "entropy": float(r["top9_entropy_bits"]),
                "top": top,
            }, indent=1))
        (cell_dir / "live_feature_ids.txt").write_text(
            ",".join(str(i) for i in live_ids)
        )
        ok(f"{cond}: {len(live_ids)} live feature JSONs → {cell_dir}")


if __name__ == "__main__":
    main()
