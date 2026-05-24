"""Compose a PNG gallery of the most monosemantic SAE features.

Alternative visualization to the 6 MB HTML dashboard. Reads the per-feature
JSONs produced by `feature_dashboard.py`, ranks features by class-entropy
(ascending), and composes a single PNG with one row per feature and 9
thumbnails per row + class labels.

Usage:
    uv run python scripts/analysis/feature_gallery_png.py \\
        --dashboard_dir outputs/phase4_5b_feature_viz/eq_vae_L6_T2 \\
        --n_rows 24 --out gallery_top24.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_records(feat_dir: Path) -> list[dict]:
    out = []
    for f in feat_dir.glob("feature_*.json"):
        d = json.load(open(f))
        if not d.get("live"):
            continue
        out.append(d)
    return out


def _rank_monosemantic(
    records: list[dict],
    *,
    min_density: float = 1e-4,
    max_density: float = 0.10,
    max_entropy: float = 2.5,
) -> list[dict]:
    """Filter to plausible monosemantic candidates then sort by ascending
    entropy. Density bounds drop both dead-ish features (<0.01% fires) and
    always-on features (>10% — usually grammar/positional)."""
    filtered = [
        d for d in records
        if min_density < d["density"] < max_density
        and d.get("entropy", 99.0) < max_entropy
    ]
    return sorted(filtered, key=lambda d: (d["entropy"], -d["density"]))


def _short_label(name: str, max_len: int = 22) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def _compose_gallery(
    selected: list[dict],
    thumbs_dir: Path,
    out_path: Path,
    *,
    thumb_size: int = 96,
    row_label_w: int = 200,
    inter_thumb_pad: int = 4,
    row_pad: int = 8,
    header_h: int = 60,
    sub_label_h: int = 16,
) -> None:
    """Render an N-row × 9-col image grid as a single PNG.

    Layout per row:
      [feature-stats label column | 9 thumbnails | per-thumb class label]
    """
    n_rows = len(selected)
    cols = 9
    row_h = thumb_size + sub_label_h + row_pad
    width = row_label_w + cols * (thumb_size + inter_thumb_pad) + inter_thumb_pad
    height = header_h + n_rows * row_h + row_pad

    bg = (245, 240, 220)   # Palette B cream-ish
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11,
        )
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12,
        )
        font_header = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20,
        )
    except OSError:
        font = ImageFont.load_default()
        font_bold = font
        font_header = font

    # Header banner (Palette B teal `#335C67`).
    draw.rectangle([(0, 0), (width, header_h)], fill=(51, 92, 103))
    draw.text(
        (12, 14),
        "SAE feature gallery — top-24 monosemantic features",
        fill=(255, 243, 176),
        font=font_header,
    )
    draw.text(
        (12, 38),
        "eq_vae · L6 · T2 · matryoshka K=256 d=32 768 · DiT step 200k",
        fill=(255, 243, 176),
        font=font,
    )

    for row_i, d in enumerate(selected):
        y0 = header_h + row_i * row_h

        # Row stats column (feature id, density, entropy, mean act, n classes).
        stats_lines = [
            f"feature {d['feature_id']}",
            f"density {d['density'] * 100:.3f}%",
            f"entropy {d['entropy']:.2f}",
            f"mean act {d['mean_act']:.2f}",
            f"unique cls {d['unique_classes']}",
        ]
        for j, ln in enumerate(stats_lines):
            f = font_bold if j == 0 else font
            draw.text((10, y0 + 2 + j * 14), ln, fill=(28, 27, 25), font=f)

        # Thumbnails + per-thumb label.
        x0 = row_label_w
        for k, top in enumerate(d["top"][:cols]):
            x = x0 + k * (thumb_size + inter_thumb_pad)
            local_idx = top.get("image_local_idx")
            thumb_path = thumbs_dir / f"img_{local_idx:05d}.jpg"
            if thumb_path.exists():
                try:
                    im = Image.open(thumb_path).convert("RGB")
                    im = im.resize((thumb_size, thumb_size), Image.BICUBIC)
                    canvas.paste(im, (x, y0))
                except Exception:
                    draw.rectangle(
                        [(x, y0), (x + thumb_size, y0 + thumb_size)],
                        fill=(80, 80, 80),
                    )
            else:
                draw.rectangle(
                    [(x, y0), (x + thumb_size, y0 + thumb_size)],
                    fill=(80, 80, 80),
                )
            # Per-thumb label band (class name, truncated).
            band_y = y0 + thumb_size
            draw.rectangle(
                [(x, band_y), (x + thumb_size, band_y + sub_label_h)],
                fill=(40, 40, 40),
            )
            label = _short_label(str(top.get("label", "?")), max_len=14)
            draw.text(
                (x + 2, band_y + 1),
                label,
                fill=(255, 243, 176),
                font=font,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dashboard_dir", type=Path,
        default=Path("outputs/phase4_5b_feature_viz/eq_vae_L6_T2"),
        help="Dashboard directory (must contain features/ and thumbs/).",
    )
    p.add_argument("--n_rows", type=int, default=24)
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path. Default: <dashboard_dir>/gallery_top<N>.png")
    p.add_argument("--thumb_size", type=int, default=96)
    p.add_argument("--max_density", type=float, default=0.10)
    p.add_argument("--max_entropy", type=float, default=2.5)
    args = p.parse_args()

    feat_dir = args.dashboard_dir / "features"
    thumbs_dir = args.dashboard_dir / "thumbs"
    if not feat_dir.exists() or not thumbs_dir.exists():
        print(f"Missing features/ or thumbs/ under {args.dashboard_dir}",
              file=sys.stderr)
        return 1

    print(f"Loading feature records from {feat_dir} …")
    records = _load_records(feat_dir)
    print(f"  {len(records)} live features total")

    selected = _rank_monosemantic(
        records,
        max_density=args.max_density,
        max_entropy=args.max_entropy,
    )
    print(f"  {len(selected)} monosemantic candidates "
          f"(density < {args.max_density*100:.0f}%, entropy < {args.max_entropy})")
    selected = selected[: args.n_rows]
    print(f"  → keeping top {len(selected)} for gallery")

    out_path = args.out or args.dashboard_dir / f"gallery_top{args.n_rows}.png"
    _compose_gallery(
        selected, thumbs_dir, out_path,
        thumb_size=args.thumb_size,
    )
    # Print a small summary table for the user.
    print("\nGallery rows (ascending entropy):")
    print(f"{'feat':>6} {'density':>8} {'entropy':>7} {'uniq':>4}  top-3 classes")
    for d in selected:
        top3 = ", ".join(t["label"] for t in d["top"][:3])
        print(f"{d['feature_id']:>6} {d['density']*100:7.3f}% "
              f"{d['entropy']:7.3f} {d['unique_classes']:>4}  {top3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
