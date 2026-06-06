"""Phase 4.5b — VLM auto-interpretation of monosemantic SAE features.

For one already-rendered dashboard cell (`outputs/.../<cond>_L<L>_T<T>/`),
walk every monosemantic feature (density 1e-4..0.10 AND entropy < 2.5 bits),
build a 3x3 grid (576x576) from its top-9 thumbnails, prompt Qwen3-VL-8B with a
short multi-image classification question, and write the resulting caption back
into `features/feature_<j>.json` as `vlm_interpretation`.

Also writes `vlm_log.jsonl` (one row per feature: id, caption, latency_s).

Run **after** `feature_dashboard.py` for the same cell. Requires the Qwen3-VL
weights cached at `$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct`.

Usage::

    uv run python scripts/analysis/feature_vlm_interp.py \\
        --cell_dir outputs/phase4_5b_feature_viz_ynull/repa_e_L3_T2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# HF offline + cache layout — same as the rest of Phase 4.5b.
_FAST = "/leonardo_scratch/fast/IscrC_YENDRI"
os.environ.setdefault("HF_HOME", f"{_FAST}/lcerovaz/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", f"{_FAST}/lcerovaz/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_FAST}/lcerovaz/hf_cache/transformers")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from diffmechint.utils import error, info, ok, warn  # noqa: E402

VLM_MODEL_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"
IMAGEFOLDER_ROOT = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/imagenet_val_imagefolder"
)

# Prompt registry. v3 is the winner of the Phase 4.5b A/B test (see vlm_prompt_ab):
# it pins the model to "one ImageNet category", forbids the most common hedging
# preambles, and gives few-shot examples of the desired label shape.
PROMPTS: dict[str, str] = {
    "v1": (
        "Name the single object or scene class these 9 images share, in 5-7 words. "
        "No hedging."
    ),
    "v2": (
        "In 3-5 words, what single ImageNet class do these 9 images share? "
        "Just the name. If multiple categories, say 'mixed: X, Y'."
    ),
    "v3": (
        "These 9 images are top activations of one neuron in a vision model. "
        "The neuron likely fires on one ImageNet category. In 1-4 words, name it. "
        "Examples: 'giant panda', 'wading shorebird', 'industrial machinery', "
        "'mixed: dogs and cats'. No preamble. No 'The commonality is'. Just the label."
    ),
}
DEFAULT_PROMPT_VERSION = "v3"

GRID_TILE = 192  # one cell = 192x192 → 576x576 composite (3x3)
GRID_SIDE = 3
DENSITY_MIN = 1e-4
DENSITY_MAX = 0.10
ENTROPY_MAX = 2.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_monosemantic(rec: dict) -> bool:
    if not rec.get("live", False):
        return False
    d = float(rec.get("density", 0.0))
    e = float(rec.get("entropy", 99.0))
    return (DENSITY_MIN < d < DENSITY_MAX) and (e < ENTROPY_MAX)


def _open_tile(thumb_path: Path, entry: dict | None, tile: int) -> Image.Image:
    """Open a dashboard thumb, or reconstruct it from the ImageNet source."""
    src = thumb_path
    if src.exists():
        with Image.open(src) as img:
            return img.convert("RGB").resize((tile, tile), Image.LANCZOS)
    if entry is None:
        raise FileNotFoundError(str(thumb_path))
    synset = entry.get("synset")
    filename = entry.get("filename")
    if not synset or not filename:
        raise FileNotFoundError(str(thumb_path))
    src = IMAGEFOLDER_ROOT / str(synset) / str(filename)
    with Image.open(src) as img:
        img = img.convert("RGB")
        w, h = img.size
        side = min(w, h)
        img = img.crop(((w - side) // 2, (h - side) // 2,
                        (w + side) // 2, (h + side) // 2))
        return img.resize((tile, tile), Image.LANCZOS)


def _build_grid(
    thumb_paths: list[Path],
    top_entries: list[dict] | None = None,
    tile: int = GRID_TILE,
) -> Image.Image:
    """3x3 composite from thumbnails, falling back to cited ImageNet JPEGs."""
    side = GRID_SIDE
    canvas = Image.new("RGB", (tile * side, tile * side), (255, 243, 176))
    for i, p in enumerate(thumb_paths[: side * side]):
        try:
            entry = top_entries[i] if top_entries is not None and i < len(top_entries) else None
            img = _open_tile(p, entry, tile)
            canvas.paste(img, ((i % side) * tile, (i // side) * tile))
        except Exception as exc:
            warn(f"thumb {p} unreadable: {exc}")
    return canvas


def _post_process(caption: str) -> str:
    """Strip whitespace, collapse newlines into one line."""
    txt = caption.strip()
    return " ".join(txt.split())


def _load_qwen3vl(model_id: str, device: torch.device):
    """Load Qwen3-VL in bf16 with FlashAttention2 if available."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    info(f"Loading processor for {model_id}…")
    processor = AutoProcessor.from_pretrained(model_id)

    # Prefer FA2; fall back to SDPA if the flash-attn build is missing.
    kwargs = dict(
        dtype=torch.bfloat16,
        device_map=str(device),
        low_cpu_mem_usage=True,
    )
    try:
        info("Loading Qwen3-VL in bf16 with flash_attention_2…")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation="flash_attention_2", **kwargs,
        )
    except Exception as exc:
        warn(f"flash_attention_2 unavailable ({exc!r}); falling back to SDPA")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation="sdpa", **kwargs,
        )
    model.eval()
    return processor, model


@torch.inference_mode()
def _caption(processor, model, image: Image.Image, device: torch.device,
             user_prompt: str = PROMPTS[DEFAULT_PROMPT_VERSION]) -> str:
    """Single VLM inference on a composite 3x3 grid image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    # apply_chat_template handles the Qwen3-VL chat schema (system/image tokens)
    # and returns processed model inputs in one call when `tokenize=True`.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
    )
    # Strip the prompt tokens — keep only the model's continuation.
    in_len = inputs["input_ids"].shape[1]
    gen = out[:, in_len:]
    decoded = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return _post_process(decoded)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell_dir", type=Path, required=True,
                   help="Cell dashboard dir (contains features/ and thumbs/).")
    p.add_argument("--model_id", default=VLM_MODEL_DEFAULT)
    p.add_argument("--limit", "--max_features", type=int, default=None,
                   dest="limit",
                   help="Process at most N features (smoke testing).")
    p.add_argument("--force", action="store_true",
                   help="Re-caption features that already have vlm_interpretation.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print captions but do not write JSONs / vlm_log.")
    p.add_argument("--prompt_version", choices=sorted(PROMPTS.keys()),
                   default=DEFAULT_PROMPT_VERSION,
                   help=f"Which prompt to use (default {DEFAULT_PROMPT_VERSION}).")
    p.add_argument("--features", type=str, default=None,
                   help="Comma-separated explicit feature IDs to caption "
                        "(skips the monosemantic filter and the dedup check).")
    p.add_argument("--log_path", type=Path, default=None,
                   help="Override the JSONL log path. Each row also gets a "
                        "prompt_version field. Used by the A/B harness.")
    args = p.parse_args()
    user_prompt = PROMPTS[args.prompt_version]

    cell_dir: Path = args.cell_dir
    feat_dir = cell_dir / "features"
    thumb_dir = cell_dir / "thumbs"
    if not feat_dir.is_dir():
        error(f"feature dir missing: {feat_dir}")
        return 1
    if not thumb_dir.is_dir():
        warn(f"thumb dir missing: {thumb_dir}; falling back to ImageNet files in feature JSONs")

    # Discover candidate features.
    candidates: list[Path] = []
    t0 = time.perf_counter()
    if args.features is not None:
        # Explicit feature ID list: skip the monosemantic filter and the dedup
        # check. Used by the A/B harness to re-caption a fixed comparison set.
        ids = [int(x) for x in args.features.split(",") if x.strip()]
        info(f"Explicit feature IDs: {ids}")
        for fid in ids:
            fp = feat_dir / f"feature_{fid}.json"
            if not fp.exists():
                warn(f"feature_{fid}.json not found under {feat_dir}")
                continue
            candidates.append(fp)
    else:
        info(f"Scanning {feat_dir}…")
        for fp in sorted(feat_dir.glob("feature_*.json")):
            try:
                rec = json.loads(fp.read_text())
            except Exception as exc:
                warn(f"unparseable {fp.name}: {exc}")
                continue
            if not _is_monosemantic(rec):
                continue
            if not args.force and "vlm_interpretation" in rec:
                continue
            candidates.append(fp)
            if args.limit is not None and len(candidates) >= args.limit:
                break
    info(f"  scanned in {time.perf_counter()-t0:.1f}s, {len(candidates)} features to caption")
    if args.limit is not None:
        candidates = candidates[: args.limit]
        info(f"  --limit={args.limit} → {len(candidates)} features")
    if not candidates:
        log_path = args.log_path if args.log_path is not None else (cell_dir / "vlm_log.jsonl")
        if not args.dry_run:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch(exist_ok=True)
        ok("nothing to do")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available — VLM on CPU will be extremely slow.")
    processor, model = _load_qwen3vl(args.model_id, device)
    ok(f"Qwen3-VL loaded on {device}")

    log_path = args.log_path if args.log_path is not None else (cell_dir / "vlm_log.jsonl")
    if log_path != (cell_dir / "vlm_log.jsonl") and not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = None if args.dry_run else log_path.open("a", encoding="utf-8")
    info(f"prompt_version={args.prompt_version}  log_path={log_path}")

    n_ok, n_fail, t_loop = 0, 0, time.perf_counter()
    for i, fp in enumerate(candidates):
        rec = json.loads(fp.read_text())
        top = rec.get("top", [])
        thumbs = [
            thumb_dir / f"img_{int(e['image_local_idx']):05d}.jpg"
            for e in top[:9]
        ]
        try:
            grid = _build_grid(thumbs, top_entries=top)
            t_inf = time.perf_counter()
            cap = _caption(processor, model, grid, device, user_prompt=user_prompt)
            dt = time.perf_counter() - t_inf
        except Exception as exc:
            warn(f"feature {rec['feature_id']}: {exc}")
            if log_f is not None:
                log_f.write(json.dumps({
                    "feature_id": rec["feature_id"],
                    "prompt_version": args.prompt_version,
                    "error": str(exc),
                }) + "\n")
                log_f.flush()
            n_fail += 1
            continue

        # When --log_path is overridden (A/B mode) or --features is used, do not
        # mutate the per-feature JSONs — those carry the chosen production caption.
        write_back = (
            not args.dry_run
            and args.log_path is None
            and args.features is None
        )
        if args.dry_run:
            info(f"  [dry] feat #{rec['feature_id']:5d} → {cap!r} ({dt:.2f}s)")
        else:
            if write_back:
                rec["vlm_interpretation"] = cap
                fp.write_text(json.dumps(rec))
            log_f.write(json.dumps({
                "feature_id": rec["feature_id"],
                "prompt_version": args.prompt_version,
                "caption": cap,
                "latency_s": round(dt, 3),
                "density": rec.get("density"),
                "entropy": rec.get("entropy"),
            }) + "\n")
            log_f.flush()
        n_ok += 1

        if i < 5 or i % 25 == 0 or i == len(candidates) - 1:
            elapsed = time.perf_counter() - t_loop
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(candidates) - i - 1) / max(rate, 1e-6)
            info(f"  [{i+1}/{len(candidates)}] feat #{rec['feature_id']:5d}  "
                 f"{dt:.2f}s  rate={rate:.2f}/s  ETA={eta/60:.1f}min  "
                 f"cap={cap[:80]!r}")

    if log_f is not None:
        log_f.close()
    ok(f"VLM captions done: {n_ok} ok / {n_fail} failed; "
       f"{'(dry run, no JSONs written)' if args.dry_run else f'log={log_path}'}")
    return 0 if n_fail == 0 else (0 if n_ok > 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
