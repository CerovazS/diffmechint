"""Extract residual-stream activations from a single (condition x ckpt) pair.

For each EMA checkpoint of a SiT run, build x_t = (1-t)*eps + t*x_1 at the
3 Revelio bins {0.025, 0.20, 0.50} on a held-out subset of the cached
ImageNet latents, run forward with `ResidualStreamTap` attached on layers
[3, 6, 9], and flush the per-cell HDF5 shards.

Output layout (one (layer, t_bin) shard per ckpt):

    $SCRATCH/diffmechint/activations/<condition>/step_<NNNNNN>/<layer>_<t_bin>.h5
        ├── activations  (N, T, D)  fp16
        └── labels       (N,)       int64

Usage (inside an `srun --gres=gpu:1`):
    uv run python scripts/extract_activations.py <run_dir> <condition> \
        --n_samples 2000 --layers 3 6 9 --bins 0.025 0.20 0.50

Relies on the same cached latent shard layout as Phase 2 training and the
matching `stats.json` (so the noise prior matches the per-feature z-score
the SiT was trained against — or skips it for the no-zscore conditions).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# HF cache offline (we only need the SiT class here, no adapter download).
_FAST = "/leonardo_scratch/fast/IscrC_YENDRI"
os.environ.setdefault("HF_HOME", f"{_FAST}/lcerovaz/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", f"{_FAST}/lcerovaz/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_FAST}/lcerovaz/hf_cache/transformers")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from diffmechint.hooks import ActivationBuffer, ResidualStreamTap  # noqa: E402
from diffmechint.hooks.timestep_router import timestep_context  # noqa: E402
from diffmechint.sit import build_sit_model, list_ema_checkpoints  # noqa: E402
from diffmechint.training.data import CachedLatentDataset  # noqa: E402
from diffmechint.utils import (  # noqa: E402
    error,
    info,
    model_subdir,
    model_variant_spec,
    ok,
    parse_layers,
    warn,
)

SCRATCH_PROJECT_ROOT = Path("/leonardo_scratch/large/userexternal/lcerovaz/diffmechint")
LATENTS_BASE = SCRATCH_PROJECT_ROOT / "latents"
ACTIVATIONS_BASE = SCRATCH_PROJECT_ROOT / "activations"
NUM_CLASSES = 1000


def _build_sample_indices(
    dataset: CachedLatentDataset,
    n_samples: int,
    stratified_per_class: int | None,
    seed: int,
) -> np.ndarray:
    """Return the dataset indices to extract for this checkpoint.

    Two modes:
      - `stratified_per_class is None` → draw `n_samples` uniformly from val.
      - `stratified_per_class > 0` → for each class label observed in val,
        draw exactly that many indices. Total = (#classes) x stratified_per_class.
        Classes with fewer val samples than requested raise (we want balanced
        coverage; silently truncating would produce uneven cells).
    """
    rng = np.random.default_rng(seed)
    n_total = len(dataset)
    if stratified_per_class is None:
        n = min(n_samples, n_total)
        if n_samples > n_total:
            warn(f"requested n_samples={n_samples} > dataset size {n_total}; clamping")
        return rng.choice(n_total, size=n, replace=False).astype(np.int64)

    # Stratified: build (label → [val_indices]) once, then sample per class.
    # Fast path — read raw labels arrays from HDF5 shards directly instead of
    # going through dataset.__getitem__ (which decodes the full latent each
    # time and would cost ~1 min per ckpt for 64k val samples).
    info(f"  stratified sampling: {stratified_per_class}/class — bucketing labels…")
    import h5py
    all_labels = []
    for shard in dataset.shards:
        with h5py.File(shard, "r") as f:
            all_labels.append(f["labels"][()])
    all_labels = np.concatenate(all_labels)  # global, pre-holdout indexing
    if dataset._index_map is not None:
        # Restrict to the val partition the dataset was constructed with.
        all_labels = all_labels[dataset._index_map]
    label_to_idx: dict[int, list[int]] = {}
    for local_i, lab in enumerate(all_labels):
        label_to_idx.setdefault(int(lab), []).append(local_i)
    info(f"  found {len(label_to_idx)} classes in val partition")

    short = [(lab, len(v)) for lab, v in label_to_idx.items() if len(v) < stratified_per_class]
    if short:
        worst = sorted(short, key=lambda x: x[1])[:5]
        raise ValueError(
            f"{len(short)} classes have fewer than {stratified_per_class} val samples; "
            f"worst examples: {worst}. Lower --stratified or use random sampling."
        )

    out: list[int] = []
    for lab in sorted(label_to_idx):
        choices = rng.choice(label_to_idx[lab], size=stratified_per_class, replace=False)
        out.extend(int(i) for i in choices)
    rng.shuffle(out)  # mix classes within batches so a single batch sees variety
    return np.asarray(out, dtype=np.int64)


@torch.no_grad()
def _extract_one_ckpt(
    model: torch.nn.Module,
    dataset: CachedLatentDataset,
    layers: list[int],
    bins: tuple[float, ...],
    n_samples: int,
    batch_size: int,
    out_dir: Path,
    seed: int,
    device: torch.device,
    stratified_per_class: int | None = None,
    y_null: bool = False,
    metadata: dict | None = None,
) -> None:
    """Forward `n_samples` clean latents through `model` at each `t_bin` and
    flush the captured per-cell activations to HDF5."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = ActivationBuffer(
        max_records_per_cell=0,  # accumulate in RAM, single flush at end
        shard_dir=str(out_dir),
        store_dtype=torch.float16,
    )
    tap = ResidualStreamTap(model, block_indices=layers, buffer=buf, bins=bins, tol=0.01)

    # Held-out indices: deterministic via the dataset's holdout split (already
    # set up by training-time CachedLatentDataModule). Two sampling modes:
    #   - random: draw `n_samples` uniformly from the val partition
    #   - stratified: draw exactly `stratified_per_class` indices per class
    sample_idx = _build_sample_indices(
        dataset, n_samples, stratified_per_class, seed,
    )
    n_samples = len(sample_idx)

    info(f"  {n_samples} samples x {len(bins)} t_bins x {len(layers)} layers; "
         f"batch_size={batch_size}")

    tap.attach()
    try:
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            idxs = sample_idx[batch_start:batch_end]
            x1 = torch.stack(
                [torch.as_tensor(dataset[int(i)]["latent"]) for i in idxs], dim=0
            ).to(device)
            y = torch.tensor(
                [int(dataset[int(i)]["label"]) for i in idxs], dtype=torch.long, device=device
            )
            # When y_null is set, swap the class label with the null token used during
            # CFG-dropout (id = NUM_CLASSES). This removes the y_embedder contribution
            # from the residual stream while keeping x_t / sample order / manifest
            # labels untouched — downstream probing/SAE can still resolve true labels
            # from `manifest.sample_idx`.
            y_for_forward = torch.full_like(y, NUM_CLASSES) if y_null else y
            for t_val in bins:
                eps = torch.randn_like(x1)
                t = torch.full((x1.shape[0],), float(t_val), device=device)
                # Linear FM-OT path: x_t = (1-t)*eps + t*x_1 (matches conf/transport/fm_ot.yaml).
                x_t = (1.0 - t_val) * eps + t_val * x1
                with timestep_context(t_val):
                    _ = model(x_t, t, y_for_forward)
                # Hooks captured one record per (layer x batch). Re-write labels
                # so the first call ends up labelled (the buffer enforces consistency).
                # Workaround: write a no-op labelled batch via direct API path —
                # since the tap doesn't pass labels, we patch them after the fact.
        # Inject labels by re-writing — we need Phase 5 probes to find them.
        # Implementation detail: we re-run the loop *with* explicit labels through
        # a manual write. To keep things simple for v1, we skip label storage at
        # extraction time; the cached dataset's index→label mapping plus the
        # deterministic seed lets Phase 5 reconstruct labels offline.
        # (Document this in the per-ckpt manifest below.)
    finally:
        tap.detach()

    info(f"  buffered {len(buf)} records across {len(buf.keys())} cells; flushing…")
    buf.flush_all()

    # Write a small manifest so downstream code knows seed, sample order, etc.
    manifest = {
        "n_samples": int(n_samples),
        "batch_size": int(batch_size),
        "layers": list(layers),
        "bins": list(bins),
        "seed": int(seed),
        "sample_idx": sample_idx.tolist(),  # restores the (idx → label) mapping
        "store_dtype": "float16",
        "sampling_mode": "stratified" if stratified_per_class else "random",
        "stratified_per_class": int(stratified_per_class) if stratified_per_class else None,
        "y_null": bool(y_null),
        "format_version": "1",
    }
    if metadata:
        manifest.update(metadata)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path,
                   help="Path to a SiT run dir under $SCRATCH/diffmechint/runs/...")
    p.add_argument("condition", type=str,
                   help="Tokenizer / condition name — must match a subdir of $SCRATCH/.../latents/")
    p.add_argument("--model_name", type=str, default="SiT-B/2")
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--layers", nargs="+", default=["auto"],
                   help="Layer indices to tap, or 'auto' for model-aware quartile taps.")
    p.add_argument("--bins", type=float, nargs="+", default=[0.025, 0.20, 0.50])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stratified", type=int, default=None,
                   help="Stratified sampling: draw exactly this many val samples per class. "
                        "Overrides --n_samples. Total = #classes x this value.")
    p.add_argument("--out_root", type=Path, default=None,
                   help="Activation root; defaults to "
                        "$SCRATCH/diffmechint/by_model/<model_variant>/activations. "
                        "Per-ckpt subdirs are <out_root>/<condition>/step_<NNNNNN>/")
    p.add_argument("--latents_root", type=Path, default=LATENTS_BASE,
                   help="Latent shard root; reads from <latents_root>/<condition>/. "
                        "Override to use a parallel tree like 'latents_val/'.")
    p.add_argument("--no_holdout", action="store_true",
                   help="Disable the 5%% train holdout split; use the entire shard set. "
                        "Set when --latents_root points at val-encoded latents (the whole "
                        "val set is already OOD for the SiT, no need to carve a holdout).")
    # Whether to apply per-feature z-score on the cached latents before
    # constructing x_t. Must match the training-time setting of the source run.
    p.add_argument("--no_normalize", action="store_true",
                   help="Skip z-score on cached latents — use for runs trained with +data.normalize=false.")
    p.add_argument("--only_step", type=int, default=None,
                   help="If set, extract only this single ckpt step (for smoke tests).")
    p.add_argument("--y_null", action="store_true",
                   help="Replace the class label with the null token (= NUM_CLASSES) during "
                        "forward — removes the y_embedder contribution from the residual "
                        "stream. Manifest labels (sample_idx) remain the true labels.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("CUDA not available — extraction will be very slow.")

    shard_dir = args.latents_root / args.condition
    if not shard_dir.exists():
        error(f"latent shards not found at {shard_dir}")
        return 1

    info(f"Loading dataset from {shard_dir} (normalize={not args.no_normalize}, "
         f"no_holdout={args.no_holdout})")
    if args.no_holdout:
        # Source already entirely OOD for the SiT (val-encoded latents) — skip
        # the holdout carve; use the full set.
        ds_kwargs = dict(holdout_fraction=0.0, holdout_seed=42, is_val=False)
    else:
        # Use the same holdout split as training so we don't analyze data the
        # SiT trained on. Source run trained with holdout_fraction=0.05.
        ds_kwargs = dict(holdout_fraction=0.05, holdout_seed=42, is_val=True)
    dataset = CachedLatentDataset(
        shard_dir=shard_dir,
        normalize=not args.no_normalize,
        **ds_kwargs,
    )
    info(f"  dataset size: {len(dataset)}")

    # Read latent shape from stats.json for model construction.
    stats = json.loads((shard_dir / "stats.json").read_text())
    in_channels = int(stats["feature_dim"])
    input_size = int(stats["input_size"])
    info(f"Adapter={args.condition} latent_shape={stats['latent_shape']} "
         f"in_ch={in_channels} input_size={input_size}")

    spec = model_variant_spec(args.model_name)
    layers = parse_layers(args.layers, spec)

    info(f"Building model {spec.model_name} (in_ch={in_channels}, input_size={input_size})")
    model = build_sit_model(spec.model_name, in_channels, input_size, device)
    n_blocks = len(model.blocks)
    info(f"  model has {n_blocks} blocks; tapping {layers}")
    bad = [i for i in layers if not (0 <= i < n_blocks)]
    if bad:
        error(f"layer indices {bad} out of range for {n_blocks}-block model")
        return 1

    ckpts = list_ema_checkpoints(args.run_dir)
    if args.only_step is not None:
        ckpts = [(s, p) for s, p in ckpts if s == args.only_step]
        if not ckpts:
            error(f"no EMA checkpoint at step {args.only_step}")
            return 1
    info(f"Found {len(ckpts)} EMA checkpoints to extract from "
         f"({args.run_dir.name})")

    out_root = args.out_root or model_subdir(SCRATCH_PROJECT_ROOT, spec.variant_id, "activations")
    cond_root = out_root / args.condition
    cond_root.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "model_name": spec.model_name,
        "model_variant": spec.variant_id,
        "depth": spec.depth,
        "hidden_size": spec.hidden_size,
        "patch_size": spec.patch_size,
        "tap_layers": list(spec.tap_layers),
        "source_run": str(args.run_dir),
    }
    for step, ema_path in ckpts:
        out_dir = cond_root / f"step_{step:06d}"
        if (out_dir / "manifest.json").exists():
            info(f"  [skip] step {step}: manifest already present at {out_dir}")
            continue
        t0 = time.perf_counter()
        info(f"--- step {step}: loading EMA from {ema_path.name} ---")
        sd = load_file(str(ema_path))
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            warn(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

        _extract_one_ckpt(
            model=model,
            dataset=dataset,
            layers=layers,
            bins=tuple(args.bins),
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            out_dir=out_dir,
            seed=args.seed + step,
            device=device,
            stratified_per_class=args.stratified,
            y_null=args.y_null,
            metadata=run_metadata,
        )
        ok(f"  step {step}: extracted to {out_dir}  ({(time.perf_counter() - t0)/60:.2f} min)")

    ok(f"Done. Activations under {cond_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
