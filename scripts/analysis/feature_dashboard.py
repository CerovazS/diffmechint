"""Phase 4.5b — per-feature visualisation dashboard for one SAE cell.

For a single `(condition, layer, t_bin, dit_step)` cell:

1. Load the production Matryoshka SAE (`d_sae=32768, k=256`).
2. Load the matching val activation shard `<L>_<T>.h5` (N=5000, T=256, D=768).
3. Stream-encode the activations through the SAE in batches; per feature track:
     - density (fraction of val tokens where the feature fires)
     - mean activation strength on firing tokens
     - per-image max activation + the token position where it occurred
4. After the pass, for every live feature pick the 9 images with the largest
   per-image max-activation (one entry per image — never 9 tokens from the same
   image), then compute Shannon entropy over the 9 ImageNet class labels.
5. Resolve every selected `(image_idx)` back to a concrete JPEG on disk via
   the deterministic ImageFolder ordering used by precompute_latents (synsets
   sorted alphabetically, 50 imgs/class, filenames sorted within synset).
6. Render thumbnails for every cited image into `<out>/thumbs/` and write
     - `features/feature_<j>.json` — one record per live feature
     - `features.html`              — sortable single-page index

Designed to be paper-grade: deterministic mapping, no shortcuts that would
mis-attribute an activation to the wrong image.

Usage::

    uv run python scripts/analysis/feature_dashboard.py \\
        --condition eq_vae --layer 6 --t_bin 2 --dit_step 200000
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import time
from pathlib import Path

# HF cache offline — SAELens may try to phone home otherwise.
_FAST = "/leonardo_scratch/fast/IscrC_YENDRI"
os.environ.setdefault("HF_HOME", f"{_FAST}/lcerovaz/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", f"{_FAST}/lcerovaz/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_FAST}/lcerovaz/hf_cache/transformers")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from diffmechint.utils import error, info, model_subdir, model_variant_spec, ok, warn  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — paths on Leonardo + Palette B
# ---------------------------------------------------------------------------
SAE_ROOT_DEFAULT = Path(
    "/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_matryoshka_k256_d32k"
)
ACTIVATIONS_VAL_ROOT = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/activations_val"
)
SCRATCH_PROJECT_ROOT = ACTIVATIONS_VAL_ROOT.parent
FAST_PROJECT_ROOT = SAE_ROOT_DEFAULT.parent
OUTPUT_ROOT = Path("outputs")
IMAGEFOLDER_ROOT = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/imagenet_val_imagefolder"
)
TIMM_SYNSETS = Path(
    "/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/.venv/lib/python3.11/"
    "site-packages/timm/data/_info/imagenet_synsets.txt"
)
TIMM_LEMMAS = Path(
    "/leonardo_work/IscrC_PDR/lcerovaz/diffmechint/.venv/lib/python3.11/"
    "site-packages/timm/data/_info/imagenet_synset_to_lemma.txt"
)
IMGS_PER_CLASS = 50  # val ImageFolder has exactly 50 imgs per synset.
T_BIN_CENTERS = (0.025, 0.20, 0.50)
TOP_K = 9
DENSITY_THRESHOLD = 1e-6  # match `evaluate_sae_on_tokens`

PB = {
    "bg": "#f5f0dc",
    "ink": "#1c1b19",
    "teal": "#335C67",
    "amber": "#E09F3E",
    "red": "#9E2A2B",
    "cream": "#FFF3B0",
    "wine": "#540B0E",
}

# ---------------------------------------------------------------------------
# Image-resolution helpers
# ---------------------------------------------------------------------------


def _read_synsets() -> list[str]:
    syns = [s.strip() for s in TIMM_SYNSETS.read_text().splitlines() if s.strip()]
    if len(syns) != 1000:
        raise RuntimeError(f"expected 1000 synsets in {TIMM_SYNSETS}, got {len(syns)}")
    return syns


def _read_lemmas() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in TIMM_LEMMAS.read_text().splitlines():
        if not line.strip():
            continue
        s, lemma = line.split("\t", 1)
        out[s.strip()] = lemma.strip()
    return out


def _list_synset_dir(d: Path, *, max_retries: int = 4,
                     base_sleep: float = 2.0) -> list[str]:
    """List one synset directory with Lustre-resilient retries.

    Uses `os.listdir(str(d))` which is a single round-trip to the metadata
    server — unlike `Path.iterdir()` + `p.is_file()`, which calls `stat()`
    on every entry. On this val tree the per-entry `stat()` follows a
    symlink to a *different* filesystem (`/leonardo_scratch/fast/IscrC_YENDRI`),
    making it especially prone to the `BrokenPipeError: [Errno 108]` Lustre
    flake observed on lrdn0094 / lrdn3265.

    On transient `OSError` (incl. `BrokenPipeError`), we sleep with
    exponential backoff and retry. After all retries fail, the exception
    is re-raised so the caller can decide.
    """
    last_exc: OSError | None = None
    for attempt in range(max_retries):
        try:
            return os.listdir(str(d))
        except OSError as exc:
            last_exc = exc
            sleep_s = base_sleep * (2 ** attempt)
            warn(f"listdir({d.name}) failed (attempt {attempt+1}/{max_retries}, "
                 f"errno={exc.errno}): {exc.__class__.__name__}: {exc}. "
                 f"Sleeping {sleep_s:.1f}s before retry.")
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _build_index_to_image(synsets: list[str]) -> list[tuple[str, str, int]]:
    """For each dataset_idx in 0..49999, return (synset, filename, class_idx).

    Mirrors `torchvision.datasets.ImageFolder` semantics: classes are sorted
    alphabetically, files within each class are sorted by `os.scandir` +
    sort (which `make_dataset` uses internally, matching Python's `sorted`).
    We pre-list every directory once so the dashboard run does ~1000 os.listdir
    calls instead of one per cited image.

    Lustre robustness: each per-class `os.listdir` is wrapped with
    `_list_synset_dir`, which retries with exponential backoff on
    `BrokenPipeError`/`OSError`. We deliberately avoid `Path.iterdir()` +
    `p.is_file()` because each `is_file()` stat()s the entry, and on this
    val tree every entry is a symlink crossing into another filesystem —
    that compound metadata operation is what was failing on bad nodes.
    """
    idx_to_img: list[tuple[str, str, int]] = []
    for cls_idx, syn in enumerate(synsets):
        d = IMAGEFOLDER_ROOT / syn
        names = _list_synset_dir(d)
        # No `is_file()` filter: the val tree contains only JPEG files
        # (or symlinks resolving to JPEGs) — no subdirs or stray entries.
        # Hidden dotfiles are still excluded defensively.
        files = sorted(n for n in names if not n.startswith("."))
        if len(files) != IMGS_PER_CLASS:
            warn(f"{syn}: expected {IMGS_PER_CLASS} files, found {len(files)} — "
                 f"dataset_idx ↔ filename mapping may drift.")
        for fn in files:
            idx_to_img.append((syn, fn, cls_idx))
    return idx_to_img


# ---------------------------------------------------------------------------
# SAE + activation loading
# ---------------------------------------------------------------------------


def _resolve_sae_ckpt(sae_root: Path, condition: str, layer: int,
                      t_bin: int, dit_step: int) -> Path:
    base = sae_root / condition / f"L{layer}_T{t_bin}" / f"step_{dit_step:06d}"
    if not base.exists():
        raise FileNotFoundError(f"SAE step dir missing: {base}")
    finals = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("final_"))
    if not finals:
        raise FileNotFoundError(f"no final_* under {base}")
    return finals[-1]


def _load_matryoshka_sae(ckpt_dir: Path, device: torch.device):
    from sae_lens import MatryoshkaBatchTopKTrainingSAE
    sae = MatryoshkaBatchTopKTrainingSAE.load_from_disk(str(ckpt_dir), device=str(device))
    sae.eval()
    return sae


# ---------------------------------------------------------------------------
# Streaming pass
# ---------------------------------------------------------------------------


@torch.no_grad()
def _stream_encode(
    sae,
    shard_path: Path,
    *,
    batch_images: int,
    device: torch.device,
    density_threshold: float,
) -> dict[str, np.ndarray]:
    """One full pass over the val activation shard.

    Returns a dict with:
      - `max_act`       : (N, d_sae) fp16, per-image max activation
      - `argmax_tok`    : (N, d_sae) int16, token pos where max was reached
      - `density_count` : (d_sae,) int64, # firing tokens across all (image, tok)
      - `sum_active`    : (d_sae,) fp64, sum of activations over firing tokens
      - `n_tokens`      : int, total tokens streamed (N * T)
    """
    sae_dtype = next(sae.parameters()).dtype
    d_sae = int(sae.cfg.d_sae)

    with h5py.File(str(shard_path), "r") as f:
        acts = f["activations"]
        n, t, d = acts.shape
        info(f"shard {shard_path.name}: N={n} T={t} D={d}")

        max_act = np.zeros((n, d_sae), dtype=np.float16)
        argmax_tok = np.zeros((n, d_sae), dtype=np.int16)
        # Reducers stay on GPU as long as memory allows; spill to CPU otherwise.
        density_count = torch.zeros(d_sae, dtype=torch.int64, device=device)
        sum_active = torch.zeros(d_sae, dtype=torch.float64, device=device)
        n_tokens_total = 0

        n_batches = math.ceil(n / batch_images)
        t0 = time.perf_counter()
        for bi in range(n_batches):
            lo = bi * batch_images
            hi = min(lo + batch_images, n)
            # (b, T, D) fp16 on CPU
            chunk = acts[lo:hi]
            # Move to GPU as `sae_dtype` to match SAE weights.
            x = torch.from_numpy(chunk).to(device=device, dtype=sae_dtype, non_blocking=True)
            b = x.shape[0]
            x_flat = x.reshape(b * t, d)
            z = sae.encode(x_flat)  # (b*T, d_sae)
            z32 = z.float()
            # Token-level reducers — masked active.
            active = z32.abs() > density_threshold
            density_count += active.sum(dim=0).to(torch.int64)
            sum_active += (z32 * active).sum(dim=0).to(torch.float64)
            n_tokens_total += int(active.shape[0])

            # Per-image max across T tokens.
            z_b = z32.view(b, t, d_sae)  # (b, T, d_sae)
            mx, am = z_b.max(dim=1)  # both (b, d_sae)
            max_act[lo:hi] = mx.to(torch.float16).cpu().numpy()
            argmax_tok[lo:hi] = am.to(torch.int16).cpu().numpy()

            if bi % 16 == 0 or bi == n_batches - 1:
                rate = (bi + 1) * batch_images / max(time.perf_counter() - t0, 1e-6)
                info(f"  batch {bi+1}/{n_batches}  ~{rate:.0f} img/s")
        ok(f"streaming pass done in {(time.perf_counter() - t0)/60:.2f} min "
           f"(n_tokens={n_tokens_total:,})")

    return {
        "max_act": max_act,
        "argmax_tok": argmax_tok,
        "density_count": density_count.cpu().numpy(),
        "sum_active": sum_active.cpu().numpy(),
        "n_tokens": n_tokens_total,
    }


# ---------------------------------------------------------------------------
# Per-feature aggregation
# ---------------------------------------------------------------------------


def _topk_per_feature(max_act: np.ndarray, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """For each column (feature) of `max_act` (N, d_sae) return the top-k
    image indices and their activations.

    Output:
      top_img_idx : (d_sae, k) int32
      top_act     : (d_sae, k) fp32
    """
    # On CPU torch.topk over (N, d_sae) is fine: peak fp16 = 5000*32768*2 = 327 MB.
    t = torch.from_numpy(max_act)  # fp16
    # topk wants float32 internally on CPU for stability.
    top = torch.topk(t.float(), k=k, dim=0)  # values (k, d_sae), indices (k, d_sae)
    top_act = top.values.t().contiguous().numpy().astype(np.float32)
    top_img_idx = top.indices.t().contiguous().numpy().astype(np.int32)
    return top_img_idx, top_act


def _entropy(class_ids: np.ndarray) -> float:
    """Shannon entropy (nats → bits) of the empirical class distribution.

    For k=9 labels: max entropy is log2(9) ≈ 3.17 bits (all distinct), 0 if
    all 9 labels agree.
    """
    if class_ids.size == 0:
        return 0.0
    _, counts = np.unique(class_ids, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# Thumbnail rendering
# ---------------------------------------------------------------------------


def _read_bytes_resilient(src: Path, *, max_retries: int = 4,
                          base_sleep: float = 2.0) -> bytes:
    """Read a file's bytes with retries on transient Lustre `OSError`s.

    The val-tree files are symlinks crossing into another filesystem
    (`/leonardo_scratch/fast/IscrC_YENDRI/...`), which has produced
    `OSError: [Errno 5] Input/output error` and `BrokenPipeError [Errno 108]`
    on stat/open from specific compute nodes. Reading bytes (rather than
    leaving PIL to lazily stat+open the path) lets us retry the entire
    fetch as one unit.
    """
    last_exc: OSError | None = None
    for attempt in range(max_retries):
        try:
            with open(src, "rb") as fh:
                return fh.read()
        except OSError as exc:
            last_exc = exc
            sleep_s = base_sleep * (2 ** attempt)
            warn(f"read({src.name}) failed (attempt {attempt+1}/{max_retries}, "
                 f"errno={exc.errno}): {exc.__class__.__name__}: {exc}. "
                 f"Sleeping {sleep_s:.1f}s before retry.")
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _build_thumbnails(
    cited_idx: set[int],
    sample_idx: np.ndarray,
    idx_to_img: list[tuple[str, str, int]],
    out_dir: Path,
    size: int = 256,
) -> None:
    """For each cited local image index, decode + center-crop to `size` and save.

    Lustre-resilient: reads each source JPEG via `_read_bytes_resilient`
    (exponential-backoff retry on `OSError`) and feeds an in-memory
    `BytesIO` to PIL. If a single image still fails after retries, we log
    the failure and skip it rather than aborting the entire job — the
    HTML index references `img_<local_i>.jpg`, and a missing thumb only
    affects the rendering of features citing that one image.
    """
    import io
    out_dir.mkdir(parents=True, exist_ok=True)
    info(f"rendering {len(cited_idx)} thumbnails → {out_dir}")
    t0 = time.perf_counter()
    n_failed = 0
    for local_i in sorted(cited_idx):
        out = out_dir / f"img_{local_i:05d}.jpg"
        if out.exists():
            continue
        dataset_idx = int(sample_idx[local_i])
        syn, fn, _cls = idx_to_img[dataset_idx]
        src = IMAGEFOLDER_ROOT / syn / fn
        try:
            raw = _read_bytes_resilient(src)
            with Image.open(io.BytesIO(raw)) as img:
                img = img.convert("RGB")
                w, h = img.size
                # 256² center-crop to match the same preprocessing the SAE saw
                # (precompute_latents uses Resize(256) + CenterCrop(256)).
                s = min(w, h)
                img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
                img = img.resize((size, size), Image.LANCZOS)
                img.save(out, "JPEG", quality=85)
        except OSError as exc:
            n_failed += 1
            warn(f"  skipping thumb {out.name} (src={src}): {exc.__class__.__name__}: {exc}")
    if n_failed:
        warn(f"thumbnails: {n_failed}/{len(cited_idx)} failed after retries — "
             f"HTML will reference missing files for those images.")
    ok(f"thumbnails done in {(time.perf_counter() - t0)/60:.2f} min")


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_HTML_CSS_TMPL = """
:root {
  --bg: {bg};
  --ink: {ink};
  --teal: {teal};
  --amber: {amber};
  --red: {red};
  --cream: {cream};
  --wine: {wine};
}
* { box-sizing: border-box; }
body {
  font-family: ui-monospace, "JetBrains Mono", Menlo, monospace;
  background: var(--bg);
  color: var(--ink);
  margin: 0;
  padding: 16px 24px;
  font-size: 13px;
  line-height: 1.4;
}
h1 { color: var(--wine); margin: 0 0 4px 0; font-size: 20px; }
h2 { color: var(--teal); margin: 16px 0 8px 0; font-size: 15px; }
.subtitle { color: var(--teal); margin-bottom: 18px; }
.kv { color: var(--ink); opacity: 0.85; }
.kv b { color: var(--wine); }
.controls { display: flex; gap: 12px; align-items: baseline; margin: 16px 0; flex-wrap: wrap; }
.controls input, .controls select, .controls button {
  background: var(--cream);
  color: var(--ink);
  border: 1px solid var(--teal);
  padding: 4px 8px;
  font-family: inherit;
  font-size: 12px;
}
.controls button { cursor: pointer; }
.controls button:hover { background: var(--amber); }
table {
  border-collapse: collapse;
  width: 100%;
  background: var(--cream);
}
th, td {
  padding: 4px 8px;
  text-align: left;
  border-bottom: 1px solid rgba(51, 92, 103, 0.25);
  vertical-align: top;
}
th {
  background: var(--teal);
  color: var(--cream);
  cursor: pointer;
  user-select: none;
  position: sticky;
  top: 0;
}
th:hover { background: var(--wine); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.feat-row { cursor: pointer; }
tr.feat-row:hover { background: rgba(224, 159, 62, 0.20); }
tr.detail-row td {
  background: var(--bg);
  padding: 12px;
  border-bottom: 2px solid var(--teal);
}
.grid {
  display: grid;
  grid-template-columns: repeat(3, 168px);
  gap: 8px;
}
.cell {
  background: var(--cream);
  padding: 4px;
  text-align: center;
  border: 1px solid var(--teal);
}
.cell img {
  width: 160px;
  height: 160px;
  object-fit: cover;
  display: block;
  margin-bottom: 2px;
}
.cell .lbl { font-size: 11px; color: var(--wine); }
.cell .act { font-size: 11px; color: var(--teal); font-variant-numeric: tabular-nums; }
.sparkline {
  display: inline-block;
  vertical-align: middle;
}
.dead { opacity: 0.45; }
.bar {
  display: inline-block;
  background: var(--amber);
  height: 8px;
  vertical-align: middle;
}
.legend { font-size: 11px; color: var(--teal); margin-top: 4px; }
.stats { color: var(--teal); }
.vlm {
  margin: 8px 0 4px 0;
  padding: 6px 8px;
  background: var(--cream);
  border-left: 3px solid var(--amber);
  color: var(--ink);
  font-size: 12.5px;
  line-height: 1.35;
}
.vlm-tag {
  display: inline-block;
  background: var(--wine);
  color: var(--cream);
  padding: 1px 6px;
  margin-right: 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: bold;
  letter-spacing: 0.04em;
}
.hist-wrap { margin: 6px 0 8px 0; }
.hist-title {
  font-size: 10px;
  color: var(--wine);
  letter-spacing: 0.06em;
  margin-bottom: 2px;
}
td.vlm-col {
  font-style: italic;
  color: var(--wine);
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
"""


def _render_css() -> str:
    """Substitute Palette-B colours into the CSS template via simple replace.

    `.format(**PB)` would require escaping every `{` in the CSS, which is brittle.
    A literal replace on the seven named placeholders is unambiguous.
    """
    css = _HTML_CSS_TMPL
    for k, v in PB.items():
        css = css.replace("{" + k + "}", v)
    return css


_HTML_JS = r"""
const PER_PAGE_DEFAULT = 200;
let state = {
  sortKey: 'top_act',
  sortDir: -1,
  hideDead: true,
  q: '',
  page: 0,
  perPage: PER_PAGE_DEFAULT,
};

function fmt(x, dp) {
  if (x === null || x === undefined || Number.isNaN(x)) return '-';
  if (typeof x !== 'number') return String(x);
  if (dp === 0) return x.toFixed(0);
  return x.toFixed(dp ?? 4);
}

// Render an inline Neuronpedia-style activation histogram as SVG.
//   h: { bins:[N+1 edges], counts:[N], zero_count, max, n }
// All bars are green (positive — ReLU SAE has no negatives). The negative
// half of the axis is drawn for visual consistency with Neuronpedia but stays
// empty.
function renderHistogramSVG(h, opts) {
  opts = opts || {};
  const W = opts.width  || 320;
  const H = opts.height || 64;
  const PAD_L = 4, PAD_R = 4, PAD_T = 4, PAD_B = 14;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const counts = h.counts || [];
  if (counts.length === 0 || h.max <= 0) {
    return `<svg width="${W}" height="${H}" style="background:var(--cream)"></svg>`;
  }
  const maxC = Math.max(...counts);
  if (maxC <= 0) {
    return `<svg width="${W}" height="${H}" style="background:var(--cream)"></svg>`;
  }
  const n = counts.length;
  const bw = innerW / n;
  // Bars
  let bars = '';
  for (let i = 0; i < n; i++) {
    const c = counts[i];
    const bh = (c / maxC) * innerH;
    const x  = PAD_L + i * bw;
    const y  = PAD_T + innerH - bh;
    bars += `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" `
          + `width="${(bw - 0.5).toFixed(2)}" height="${bh.toFixed(2)}" `
          + `fill="var(--teal)" opacity="0.85"></rect>`;
  }
  // Axis line
  const ax = PAD_T + innerH;
  const axisLine = `<line x1="${PAD_L}" y1="${ax}" x2="${PAD_L + innerW}" y2="${ax}" `
                 + `stroke="var(--wine)" stroke-width="0.5"></line>`;
  // X ticks: 0 and max
  const xTicks =
      `<text x="${PAD_L}" y="${H - 2}" font-size="9" fill="var(--wine)">0</text>` +
      `<text x="${PAD_L + innerW}" y="${H - 2}" font-size="9" fill="var(--wine)" `
      + `text-anchor="end">${h.max.toFixed(2)}</text>`;
  // Y tick (top): max count
  const yTicks =
      `<text x="${PAD_L + 1}" y="${PAD_T + 8}" font-size="9" fill="var(--wine)">`
      + maxC.toLocaleString() + `</text>`;
  return `<svg width="${W}" height="${H}" style="background:var(--cream); border:1px solid var(--teal);">`
       + bars + axisLine + xTicks + yTicks + `</svg>`;
}

function filteredSorted() {
  let rows = FEATURES.slice();
  if (state.hideDead) rows = rows.filter(r => r.density > 0);
  if (state.q) {
    const q = state.q.toLowerCase();
    rows = rows.filter(r => String(r.feature_id).includes(q));
  }
  const k = state.sortKey;
  const dir = state.sortDir;
  rows.sort((a, b) => {
    const va = a[k] ?? -Infinity, vb = b[k] ?? -Infinity;
    if (va < vb) return -1 * dir;
    if (va > vb) return  1 * dir;
    return 0;
  });
  return rows;
}

function render() {
  const rows = filteredSorted();
  const total = rows.length;
  const start = state.page * state.perPage;
  const end = Math.min(start + state.perPage, total);
  const slice = rows.slice(start, end);

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  for (const r of slice) {
    const tr = document.createElement('tr');
    tr.className = 'feat-row';
    tr.dataset.fid = r.feature_id;
    const vlmCell = r.vlm_interpretation
      ? `<td class="vlm-col" title="${escapeHTML(r.vlm_interpretation)}">${escapeHTML(r.vlm_interpretation)}</td>`
      : `<td class="vlm-col"></td>`;
    tr.innerHTML = `
      <td class="num"><b>${r.feature_id}</b></td>
      <td class="num">${fmt(r.density, 6)}</td>
      <td class="num">${fmt(r.density_count, 0)}</td>
      <td class="num">${fmt(r.mean_act, 4)}</td>
      <td class="num">${fmt(r.top_act, 4)}</td>
      <td class="num">${fmt(r.entropy, 3)}</td>
      <td>${(r.top_labels ?? []).slice(0,3).map(escapeHTML).join(', ')}</td>
      ${vlmCell}
    `;
    tr.addEventListener('click', () => toggleDetail(tr, r.feature_id));
    tbody.appendChild(tr);
  }
  document.getElementById('count').textContent =
    `${total.toLocaleString()} features (live: ${LIVE_COUNT.toLocaleString()} / ${TOTAL_COUNT.toLocaleString()})  ·  showing ${start + 1}-${end}`;
  document.getElementById('pageinfo').textContent =
    `page ${state.page + 1} / ${Math.max(1, Math.ceil(total / state.perPage))}`;
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}

async function toggleDetail(tr, fid) {
  const next = tr.nextElementSibling;
  if (next && next.classList.contains('detail-row') && next.dataset.fid == fid) {
    next.remove();
    return;
  }
  // Remove any previously open detail row.
  document.querySelectorAll('tr.detail-row').forEach(n => n.remove());
  const det = document.createElement('tr');
  det.className = 'detail-row';
  det.dataset.fid = fid;
  const td = document.createElement('td');
  td.colSpan = 8;
  td.textContent = 'loading…';
  det.appendChild(td);
  tr.parentNode.insertBefore(det, tr.nextSibling);
  try {
    const res = await fetch(`features/feature_${fid}.json`);
    if (!res.ok) {
      td.textContent = `HTTP ${res.status}: feature_${fid}.json not found`;
      return;
    }
    const data = await res.json();
    const cells = data.top.map(e => `
      <div class="cell">
        <img src="thumbs/img_${String(e.image_local_idx).padStart(5,'0')}.jpg" loading="lazy" />
        <div class="lbl" title="${escapeHTML(e.synset)} (cls ${e.class_idx})">${escapeHTML(e.label)}</div>
        <div class="act">act ${fmt(e.activation, 3)} · tok ${e.token_pos}</div>
      </div>
    `).join('');
    const vlmRow = data.vlm_interpretation
      ? `<div class="vlm"><span class="vlm-tag">VLM</span> ${escapeHTML(data.vlm_interpretation)}</div>`
      : '';
    const histRow = data.activation_histogram
      ? `<div class="hist-wrap">
           <div class="hist-title">ACTIVATIONS DENSITY · max=${fmt(data.activation_histogram.max, 3)}
             · zero=${data.activation_histogram.zero_count.toLocaleString()}/${data.activation_histogram.n.toLocaleString()}</div>
           ${renderHistogramSVG(data.activation_histogram)}
         </div>`
      : '';
    td.innerHTML = `
      <div class="stats">
        feature <b>#${fid}</b>  ·  density <b>${fmt(data.density, 6)}</b>
        (${data.density_count.toLocaleString()} firing tokens of ${data.n_tokens.toLocaleString()})
        ·  mean_act <b>${fmt(data.mean_act, 4)}</b>
        ·  entropy <b>${fmt(data.entropy, 3)}</b> bits
        ·  classes in top-${data.top.length}: <b>${data.unique_classes}</b>
      </div>
      ${vlmRow}
      ${histRow}
      <div class="grid">${cells}</div>
    `;
  } catch (err) {
    td.textContent = 'error: ' + err.message;
  }
}

function setSort(key) {
  if (state.sortKey === key) state.sortDir = -state.sortDir;
  else { state.sortKey = key; state.sortDir = -1; }
  state.page = 0;
  render();
  document.querySelectorAll('th').forEach(t => t.classList.remove('sorted'));
  const el = document.querySelector(`th[data-key="${key}"]`);
  if (el) el.classList.add('sorted');
}

function init() {
  document.querySelectorAll('th[data-key]').forEach(th => {
    th.addEventListener('click', () => setSort(th.dataset.key));
  });
  document.getElementById('q').addEventListener('input', e => {
    state.q = e.target.value; state.page = 0; render();
  });
  document.getElementById('hidedead').addEventListener('change', e => {
    state.hideDead = e.target.checked; state.page = 0; render();
  });
  document.getElementById('perpage').addEventListener('change', e => {
    state.perPage = +e.target.value; state.page = 0; render();
  });
  document.getElementById('prev').addEventListener('click', () => {
    if (state.page > 0) { state.page--; render(); }
  });
  document.getElementById('next').addEventListener('click', () => {
    const total = filteredSorted().length;
    if ((state.page + 1) * state.perPage < total) { state.page++; render(); }
  });
  render();
}

window.addEventListener('DOMContentLoaded', init);
"""


def _write_html(
    out_path: Path,
    *,
    summary: list[dict],
    header: dict,
    live_count: int,
    total_count: int,
) -> None:
    """Write the single-page sortable index."""
    title = (
        f"SAE Feature Dashboard — {header['condition']} L{header['layer']} "
        f"T{header['t_bin']} step {header['dit_step']}"
    )
    subtitle = (
        f"d_sae={header['d_sae']:,} · k={header['k']} · "
        f"N={header['n_images']:,} val images · T={header['n_tokens_per_image']} tokens/img · "
        f"shard={html.escape(header['shard'])}"
    )
    body_kv = (
        f"<span class='kv'><b>SAE ckpt:</b> {html.escape(header['sae_ckpt'])}</span><br>"
        f"<span class='kv'><b>Activation shard:</b> {html.escape(header['shard'])}</span><br>"
        f"<span class='kv'><b>Total tokens scored:</b> {header['n_tokens']:,}</span>"
    )
    # Embed the feature summary as a JSON blob — JS does all sorting / filtering.
    embedded = json.dumps(summary, separators=(",", ":"))
    head = f"""<!DOCTYPE html>
<html lang='en'><head>
<meta charset='utf-8'>
<title>{html.escape(title)}</title>
<style>{_render_css()}</style>
</head><body>
<h1>{html.escape(title)}</h1>
<div class='subtitle'>{subtitle}</div>
<div class='kv-block'>{body_kv}</div>
<div class='controls'>
  <span id='count' class='stats'></span>
  <label>search id <input id='q' size='6' placeholder='e.g. 12345'></label>
  <label><input id='hidedead' type='checkbox' checked> hide dead</label>
  <label>per page
    <select id='perpage'>
      <option>50</option>
      <option selected>200</option>
      <option>500</option>
      <option>1000</option>
    </select>
  </label>
  <button id='prev'>&larr; prev</button>
  <span id='pageinfo'></span>
  <button id='next'>next &rarr;</button>
</div>
<table>
  <thead><tr>
    <th data-key='feature_id'>id</th>
    <th data-key='density'>density</th>
    <th data-key='density_count'># fires</th>
    <th data-key='mean_act'>mean act</th>
    <th data-key='top_act'>top act</th>
    <th data-key='entropy'>entropy (bits)</th>
    <th>top labels</th>
    <th data-key='vlm_interpretation'>VLM interpretation</th>
  </tr></thead>
  <tbody id='tbody'></tbody>
</table>
<div class='legend'>
  Click a row to expand the top-9 image grid. Sort by entropy ascending to surface
  monosemantic features; sort by density ascending to surface specialist features.
</div>
<script>
const FEATURES = {embedded};
const LIVE_COUNT = {live_count};
const TOTAL_COUNT = {total_count};
{_HTML_JS}
</script>
</body></html>
"""
    out_path.write_text(head)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _rebuild_html_only(cell_dir: Path) -> int:
    """Re-emit `features.html` from the existing per-feature JSONs and meta.json.

    Lets us refresh the dashboard layout after `feature_vlm_interp.py` /
    `recompute_histograms.py` have augmented `features/feature_<j>.json`
    without re-running the SAE forward.
    """
    meta_p = cell_dir / "meta.json"
    feat_dir = cell_dir / "features"
    if not meta_p.exists():
        error(f"meta.json missing under {cell_dir}")
        return 1
    if not feat_dir.is_dir():
        error(f"features/ missing under {cell_dir}")
        return 1
    meta = json.loads(meta_p.read_text())
    info(f"Rebuilding HTML from {feat_dir}/ + {meta_p.name}")
    summary: list[dict] = []
    d_sae = int(meta["d_sae"])
    live_files = {int(p.stem.split("_")[1]): p for p in feat_dir.glob("feature_*.json")}
    t0 = time.perf_counter()
    for j in range(d_sae):
        fp = live_files.get(j)
        if fp is None:
            summary.append({
                "feature_id": j, "density": 0.0, "density_count": 0,
                "mean_act": 0.0, "entropy": 0.0, "top_act": 0.0,
                "top_labels": [],
            })
            continue
        rec = json.loads(fp.read_text())
        row = {
            "feature_id": j,
            "density": float(rec.get("density", 0.0)),
            "density_count": int(rec.get("density_count", 0)),
            "mean_act": float(rec.get("mean_act", 0.0)),
            "entropy": float(rec.get("entropy", 0.0)),
            "top_act": float(rec["top"][0]["activation"]) if rec.get("top") else 0.0,
            "top_labels": [t["label"] for t in rec.get("top", [])[:3]],
        }
        if "vlm_interpretation" in rec:
            row["vlm_interpretation"] = rec["vlm_interpretation"]
        summary.append(row)
    live_count = int(meta.get("live_features", len(live_files)))
    header = {
        "condition": meta["condition"],
        "layer": meta["layer"],
        "t_bin": meta["t_bin"],
        "dit_step": meta["dit_step"],
        "d_sae": d_sae,
        "k": meta["k"],
        "shard": meta["shard"],
        "sae_ckpt": meta["sae_ckpt"],
        "n_images": meta["n_images"],
        "n_tokens_per_image": meta["n_tokens_per_image"],
        "n_tokens": meta["n_tokens"],
    }
    html_path = cell_dir / "features.html"
    _write_html(html_path, summary=summary, header=header,
                live_count=live_count, total_count=d_sae)
    ok(f"rebuilt {html_path} in {time.perf_counter()-t0:.2f}s")
    return 0


def main() -> int:
    # Lightweight first-pass: if --rebuild_html_only is set, none of the
    # GPU-dependent args are required. Use a sniff parser to detect the flag,
    # then dispatch to the appropriate branch.
    sniff = argparse.ArgumentParser(add_help=False)
    sniff.add_argument("--rebuild_html_only", type=Path, default=None)
    sniff_args, _ = sniff.parse_known_args()
    if sniff_args.rebuild_html_only is not None:
        return _rebuild_html_only(sniff_args.rebuild_html_only)

    p = argparse.ArgumentParser()
    p.add_argument("--rebuild_html_only", type=Path, default=None,
                   help="Cell dir: re-emit features.html from existing per-feature JSONs "
                        "and exit (no GPU needed).")
    p.add_argument("--condition", required=True, choices=["sd_vae", "eq_vae", "repa_e"])
    p.add_argument("--model_variant", type=str, default="sit_b_2",
                   help="SiT variant id or model name; used for namespaced default roots.")
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--t_bin", required=True, type=int, choices=[0, 1, 2])
    p.add_argument("--dit_step", type=int, default=200_000)
    p.add_argument("--sae_root", type=Path, default=None)
    p.add_argument("--act_root", type=Path, default=None)
    p.add_argument("--out_root", type=Path, default=None)
    p.add_argument("--batch_images", type=int, default=64,
                   help="Images per SAE forward (each → 256 tokens).")
    p.add_argument("--thumb_size", type=int, default=256)
    p.add_argument("--skip_thumbnails", action="store_true",
                   help="Skip thumbnail rendering. Use for analysis-only atlas builds "
                        "when ImageNet JPEG reads are flaky; feature JSON and stats "
                        "are still written.")
    p.add_argument("--density_threshold", type=float, default=DENSITY_THRESHOLD)
    args = p.parse_args()
    spec = model_variant_spec(args.model_variant)
    if args.sae_root is None:
        args.sae_root = (
            SAE_ROOT_DEFAULT if spec.variant_id == "sit_b_2"
            else model_subdir(FAST_PROJECT_ROOT, spec.variant_id, "sae")
        )
    if args.act_root is None:
        namespaced = model_subdir(SCRATCH_PROJECT_ROOT, spec.variant_id, "activations_val")
        args.act_root = namespaced if namespaced.exists() or spec.variant_id != "sit_b_2" else ACTIVATIONS_VAL_ROOT
    if args.out_root is None:
        args.out_root = model_subdir(OUTPUT_ROOT, spec.variant_id, "dashboards", "feature_viz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available — will be slow.")

    # ----- output dir
    cell_tag = f"{args.condition}_L{args.layer}_T{args.t_bin}"
    out_dir = args.out_root / cell_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "features").mkdir(exist_ok=True)
    info(f"out_dir={out_dir}")

    # ----- SAE
    sae_ckpt = _resolve_sae_ckpt(args.sae_root, args.condition, args.layer,
                                 args.t_bin, args.dit_step)
    info(f"Loading SAE from {sae_ckpt}")
    sae = _load_matryoshka_sae(sae_ckpt, device)
    d_sae = int(sae.cfg.d_sae)
    info(f"  d_in={sae.cfg.d_in} d_sae={d_sae} k={sae.cfg.k}")

    # ----- activation shard + manifest
    shard_path = (args.act_root / args.condition /
                  f"step_{args.dit_step:06d}" / f"{args.layer}_{args.t_bin}.h5")
    if not shard_path.exists():
        error(f"activation shard not found: {shard_path}")
        return 1
    manifest_path = shard_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sample_idx = np.asarray(manifest["sample_idx"], dtype=np.int64)
    info(f"manifest: n_samples={manifest['n_samples']} sampling_mode={manifest.get('sampling_mode')}")
    if sample_idx.shape[0] != manifest["n_samples"]:
        error(f"sample_idx length {sample_idx.shape[0]} != manifest['n_samples'] {manifest['n_samples']}")
        return 1

    # ----- imagefolder index
    synsets = _read_synsets()
    lemmas = _read_lemmas()
    info(f"Building ImageFolder index over {len(synsets)} synsets…")
    t0 = time.perf_counter()
    idx_to_img = _build_index_to_image(synsets)
    info(f"  index built ({len(idx_to_img)} entries) in {time.perf_counter()-t0:.2f}s")
    if len(idx_to_img) < int(sample_idx.max()) + 1:
        error(f"ImageFolder size {len(idx_to_img)} < max sample_idx {int(sample_idx.max())+1}")
        return 1

    # ----- per-sample labels (resolved from sample_idx via the ImageFolder index)
    # This is the canonical mapping: dataset_idx → (synset, fn, class_idx).
    sample_class_idx = np.asarray(
        [idx_to_img[int(s)][2] for s in sample_idx], dtype=np.int32,
    )

    # ----- streaming encode pass
    info(f"Streaming encode (batch_images={args.batch_images})…")
    stats = _stream_encode(
        sae, shard_path,
        batch_images=args.batch_images,
        device=device,
        density_threshold=args.density_threshold,
    )

    max_act = stats["max_act"]               # (N, d_sae) fp16
    argmax_tok = stats["argmax_tok"]         # (N, d_sae) int16
    density_count = stats["density_count"]   # (d_sae,)
    sum_active = stats["sum_active"]         # (d_sae,)
    n_tokens = int(stats["n_tokens"])

    # Aggregate scalars per feature.
    info("Computing per-feature top-K + density / entropy…")
    t0 = time.perf_counter()
    top_img_idx, top_act = _topk_per_feature(max_act, k=TOP_K)  # (d_sae, k)
    density = density_count.astype(np.float64) / max(n_tokens, 1)
    mean_act = np.where(density_count > 0,
                        sum_active / np.maximum(density_count, 1),
                        0.0).astype(np.float64)

    # Top-9 class ids per feature → entropy.
    top_class_ids = sample_class_idx[top_img_idx]  # (d_sae, k)
    entropy = np.empty(d_sae, dtype=np.float64)
    n_unique = np.empty(d_sae, dtype=np.int32)
    for j in range(d_sae):
        ids = top_class_ids[j]
        # only consider entries where activation > 0 (top_act sorted desc)
        valid = top_act[j] > args.density_threshold
        ids = ids[valid]
        if ids.size == 0:
            entropy[j] = 0.0
            n_unique[j] = 0
        else:
            entropy[j] = _entropy(ids)
            n_unique[j] = int(np.unique(ids).size)
    info(f"  aggregation in {(time.perf_counter()-t0)/60:.2f} min")

    # ----- collect cited images (only for live features) and render thumbnails
    live_mask = density_count > 0
    live_count = int(live_mask.sum())
    info(f"live features: {live_count:,} / {d_sae:,} "
         f"({100.0 * live_count / d_sae:.2f}%)")

    cited: set[int] = set()
    for j in np.flatnonzero(live_mask):
        for k in range(TOP_K):
            if top_act[j, k] > args.density_threshold:
                cited.add(int(top_img_idx[j, k]))
    info(f"cited image count (live features only): {len(cited):,}")

    if args.skip_thumbnails:
        warn("skipping thumbnail rendering (--skip_thumbnails)")
    else:
        _build_thumbnails(cited, sample_idx, idx_to_img,
                          out_dir / "thumbs", size=args.thumb_size)

    # ----- per-feature JSON records
    info(f"Writing per-feature JSON files to {out_dir/'features'}/")
    feat_dir = out_dir / "features"
    feat_dir.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    summary: list[dict] = []
    for j in range(d_sae):
        is_live = bool(live_mask[j])
        top_list = []
        if is_live:
            for k in range(TOP_K):
                local_i = int(top_img_idx[j, k])
                act = float(top_act[j, k])
                dataset_i = int(sample_idx[local_i])
                syn, fn, cls = idx_to_img[dataset_i]
                lemma = lemmas.get(syn, syn)
                top_list.append({
                    "image_local_idx": local_i,
                    "dataset_idx": dataset_i,
                    "synset": syn,
                    "label": lemma.split(",")[0].strip(),
                    "class_idx": int(cls),
                    "filename": fn,
                    "activation": act,
                    "token_pos": int(argmax_tok[local_i, j]),
                })
        rec = {
            "feature_id": j,
            "live": is_live,
            "density": float(density[j]),
            "density_count": int(density_count[j]),
            "mean_act": float(mean_act[j]),
            "n_tokens": n_tokens,
            "entropy": float(entropy[j]),
            "unique_classes": int(n_unique[j]),
            "top": top_list,
        }
        # Only write JSON for live features (dead features have no top-9).
        if is_live:
            (feat_dir / f"feature_{j}.json").write_text(json.dumps(rec))

        # The summary embedded in features.html keeps every feature so the
        # "hide dead" toggle works without an extra fetch.
        row_summary = {
            "feature_id": j,
            "density": float(density[j]),
            "density_count": int(density_count[j]),
            "mean_act": float(mean_act[j]),
            "entropy": float(entropy[j]),
            "top_act": float(top_act[j, 0]) if is_live else 0.0,
            "top_labels": [t["label"] for t in top_list[:3]],
        }
        # First build of a cell never has VLM or histogram yet; rebuild_html
        # rerun will populate these from the per-feature JSONs.
        summary.append(row_summary)
    info(f"  wrote {live_count:,} feature JSONs in {(time.perf_counter()-t0)/60:.2f} min")

    # ----- HTML
    header = {
        "condition": args.condition,
        "layer": args.layer,
        "t_bin": args.t_bin,
        "t_bin_center": T_BIN_CENTERS[args.t_bin],
        "dit_step": args.dit_step,
        "d_sae": d_sae,
        "k": int(sae.cfg.k),
        "shard": str(shard_path),
        "sae_ckpt": str(sae_ckpt),
        "n_images": int(sample_idx.shape[0]),
        "n_tokens_per_image": 256,
        "n_tokens": n_tokens,
    }
    html_path = out_dir / "features.html"
    _write_html(html_path, summary=summary, header=header,
                live_count=live_count, total_count=d_sae)
    ok(f"Wrote dashboard: {html_path}")

    # ----- top-level metadata for downstream consumers
    meta = {
        **header,
        "live_features": live_count,
        "dead_features": int(d_sae - live_count),
        "density_threshold": args.density_threshold,
        "top_k": TOP_K,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    info("meta.json written")

    # ----- aggregate stats.json for cross-cell analysis (avoid re-reading 30k
    # per-feature JSONs to compute summary stats downstream).
    live_idx = [j for j in range(d_sae) if bool(live_mask[j])]
    live_dens = np.array([density[j] for j in live_idx], dtype=np.float64)
    live_ent = np.array([entropy[j] for j in live_idx], dtype=np.float64)
    live_mean_act = np.array([mean_act[j] for j in live_idx], dtype=np.float64)
    # Monosemantic bands.
    mono_mask = (live_dens >= 1e-4) & (live_dens <= 0.10) & (live_ent < 2.5)
    very_mono_mask = (live_dens >= 1e-4) & (live_dens <= 0.10) & (live_ent < 1.5)
    pure_mono_mask = (live_dens >= 1e-4) & (live_dens <= 0.10) & (live_ent < 1.0)
    stats = {
        **header,
        "live_features": int(live_count),
        "dead_features": int(d_sae - live_count),
        "live_pct": float(live_count / d_sae * 100),
        "monosemantic_count": int(mono_mask.sum()),
        "monosemantic_pct_of_live": float(mono_mask.sum() / max(live_count, 1) * 100),
        "very_monosemantic_count": int(very_mono_mask.sum()),
        "pure_monosemantic_count": int(pure_mono_mask.sum()),
        "monosemantic_filter": {
            "min_density": 1e-4, "max_density": 0.10, "max_entropy": 2.5,
        },
        "entropy_stats": {
            "mean": float(live_ent.mean()) if live_ent.size else 0.0,
            "median": float(np.median(live_ent)) if live_ent.size else 0.0,
            "p10": float(np.percentile(live_ent, 10)) if live_ent.size else 0.0,
            "p90": float(np.percentile(live_ent, 90)) if live_ent.size else 0.0,
        },
        "density_stats": {
            "mean": float(live_dens.mean()) if live_dens.size else 0.0,
            "median": float(np.median(live_dens)) if live_dens.size else 0.0,
            "p10": float(np.percentile(live_dens, 10)) if live_dens.size else 0.0,
            "p90": float(np.percentile(live_dens, 90)) if live_dens.size else 0.0,
            "n_dead": int(d_sae - live_count),
            "n_always_on_50pct": int((live_dens > 0.50).sum()),
        },
        "mean_activation_stats": {
            "mean": float(live_mean_act.mean()) if live_mean_act.size else 0.0,
            "median": float(np.median(live_mean_act)) if live_mean_act.size else 0.0,
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    info("stats.json written (aggregate)")

    # ----- print top-N "interesting" features (high density * low normalized entropy)
    norm_ent = entropy / math.log2(TOP_K) if TOP_K > 1 else 0.0
    score = density * (1.0 - norm_ent) * live_mask
    top_inter = np.argsort(-score)[:10]
    info("Top 10 features by interestingness (density * (1 - entropy_normalized)):")
    for rank, j in enumerate(top_inter, start=1):
        j = int(j)
        lbls = [
            lemmas.get(idx_to_img[int(sample_idx[int(top_img_idx[j, k])])][0], "?").split(",")[0]
            for k in range(min(3, TOP_K))
            if top_act[j, k] > args.density_threshold
        ]
        info(f"  {rank:2d}. feat #{j:5d}  density={density[j]:.4f}  "
             f"entropy={entropy[j]:.3f}  top_act={top_act[j,0]:.3f}  "
             f"labels={lbls}")

    ok(f"Done. open: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
