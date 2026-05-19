"""Evaluate every SAE checkpoint under <sae_root> on the held-out val activations.

For each `sae_weights.safetensors` found we:
  1. Read the sibling `cfg.json` → self-describing metadata (condition, layer,
     t_bin, k, d_sae, ...). We trust metadata, not the directory path.
  2. Derive (sweep, dit_step, stage) from the path (paths are stable).
  3. Map to the matching val shard:
       <val_root>/<condition>/step_<dit_step:06d>/<layer>_<t_bin>.h5
  4. Load the SAE via `TopKTrainingSAE.load_from_disk` (handles
     `normalize_activations` folding internally).
  5. Stream the entire val shard through the SAE and compute:
       EV, MSE, recon_cosine, L0 mean, dead_pct, live_features
  6. Write per-SAE JSON + append to a master JSONL + master CSV.

Run with `--dry_run` to print the planned (sae, val_shard) pairs without
loading anything — useful for the coherence-check subagents.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

# HF offline (the SAE load shouldn't touch HF, but be defensive).
_FAST = "/leonardo_scratch/fast/IscrC_YENDRI"
os.environ.setdefault("HF_HOME", f"{_FAST}/lcerovaz/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import h5py  # noqa: E402
import torch  # noqa: E402
from sae_lens import TopKTrainingSAE  # noqa: E402

from diffmechint.utils import error, info, ok, warn  # noqa: E402

# Path regex: matches both E02 (`<sweep>/<cond>/<L>_<T>/step_<N>/<stage>/`) and
# E03 (`<sweep>/<cond>_k<K>/<cond>/<L>_<T>/step_<N>/<stage>/`) layouts. We do
# not rely on this for metadata — only for sweep_id + dit_step + stage.
STEP_RE = re.compile(r"step_(\d{6})")
STAGE_FINAL_RE = re.compile(r"final_\d+")


def discover_saes(sae_root: Path) -> list[dict]:
    """Walk sae_root and return one dict per sae_weights.safetensors found.

    Each dict has: ckpt_dir, sweep_id, dit_step, stage ('mid'|'final'),
    cfg (parsed cfg.json), metadata (from cfg).
    """
    found = []
    for w in sae_root.rglob("sae_weights.safetensors"):
        ckpt_dir = w.parent
        cfg_path = ckpt_dir / "cfg.json"
        if not cfg_path.exists():
            warn(f"  [skip] {w.relative_to(sae_root)} — no sibling cfg.json")
            continue
        cfg = json.loads(cfg_path.read_text())
        md = cfg.get("metadata", {})
        rel = w.relative_to(sae_root)
        # dit_step from the parent path component matching step_NNNNNN.
        step_match = None
        for part in rel.parts:
            m = STEP_RE.fullmatch(part)
            if m:
                step_match = int(m.group(1))
                break
        if step_match is None:
            warn(f"  [skip] {rel} — no step_NNNNNN in path")
            continue
        stage = "final" if STAGE_FINAL_RE.fullmatch(ckpt_dir.name) else "mid"
        sweep_id = rel.parts[0]  # top-level under sae_root
        found.append({
            "ckpt_dir": ckpt_dir,
            "rel_path": str(rel),
            "sweep_id": sweep_id,
            "dit_step": step_match,
            "stage": stage,
            "cfg": cfg,
            "metadata": md,
        })
    return sorted(found, key=lambda d: (d["sweep_id"], d["metadata"].get("condition", ""),
                                        d["metadata"].get("layer", -1),
                                        d["metadata"].get("t_bin", -1),
                                        d["dit_step"], d["stage"]))


def val_shard_path(val_root: Path, condition: str, dit_step: int,
                   layer: int, t_bin: int) -> Path:
    return val_root / condition / f"step_{dit_step:06d}" / f"{layer}_{t_bin}.h5"


@torch.no_grad()
def evaluate_one(
    sae: TopKTrainingSAE,
    shard_path: Path,
    *,
    batch_size: int,
    max_tokens: int | None,
    device: torch.device,
) -> dict:
    """Stream the shard through the SAE; compute EV / MSE / cosine / L0 / dead.

    EV is computed token-wise: `1 - sum||x-x_hat||^2 / sum||x - x_mean||^2`
    where `x_mean` is the global mean over evaluated tokens (estimated streaming).
    """
    with h5py.File(shard_path, "r") as f:
        acts = torch.from_numpy(f["activations"][()]).float()  # (N, T, D)
    n_records, n_tokens_per_rec, d_in = acts.shape
    tokens = acts.reshape(-1, d_in)  # (N*T, D)
    if max_tokens is not None and tokens.shape[0] > max_tokens:
        tokens = tokens[:max_tokens]
    total_tokens = tokens.shape[0]

    # Two passes:
    #   pass 1: compute global mean (for EV denominator) — done in a single
    #   reduction since we already have everything in RAM.
    x_mean = tokens.mean(dim=0, keepdim=True).to(device)  # (1, D)

    feature_active_any = torch.zeros(int(sae.cfg.d_sae), dtype=torch.bool, device=device)
    feature_active_count = torch.zeros(int(sae.cfg.d_sae), dtype=torch.long, device=device)

    sq_err_sum = 0.0
    sq_var_sum = 0.0  # ||x - x_mean||^2 summed
    cos_sum = 0.0
    l0_sum = 0  # total active counts (sum of nonzero entries across batches)

    sae.eval()
    sae_dtype = next(sae.parameters()).dtype
    for start in range(0, total_tokens, batch_size):
        end = min(start + batch_size, total_tokens)
        x = tokens[start:end].to(device=device, dtype=sae_dtype)
        x_hat = sae(x)
        z = sae.encode(x)

        # MSE / EV accumulators (sum of squared per-element distances).
        diff = (x - x_hat).float()
        sq_err_sum += float(diff.pow(2).sum().item())
        x_centered = (x.float() - x_mean.float())
        sq_var_sum += float(x_centered.pow(2).sum().item())

        # Cosine
        x_n = torch.nn.functional.normalize(x.float(), dim=-1)
        xh_n = torch.nn.functional.normalize(x_hat.float(), dim=-1)
        cos_sum += float((x_n * xh_n).sum(dim=-1).sum().item())

        # Sparsity / dead features
        active = (z.abs() > 1e-6)  # (B, d_sae)
        feature_active_any |= active.any(dim=0)
        feature_active_count += active.sum(dim=0).long()
        l0_sum += int(active.sum().item())

    n = max(total_tokens, 1)
    mse = sq_err_sum / (n * d_in)  # mean over per-element entries
    ev = 1.0 - (sq_err_sum / max(sq_var_sum, 1e-12))
    recon_cosine = cos_sum / n
    l0_mean = l0_sum / n
    live = int(feature_active_any.sum().item())
    d_sae = int(sae.cfg.d_sae)
    return {
        "n_tokens": int(n),
        "n_records": int(n_records),
        "tokens_per_record": int(n_tokens_per_rec),
        "d_in": int(d_in),
        "d_sae": d_sae,
        "ev": ev,
        "mse": mse,
        "recon_cosine": recon_cosine,
        "l0_mean": l0_mean,
        "live_features": live,
        "dead_features": d_sae - live,
        "dead_pct": (d_sae - live) / d_sae,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_root", type=Path, required=True,
                   help="Root dir under which to discover sae_weights.safetensors.")
    p.add_argument("--val_root", type=Path,
                   default=Path("/leonardo_scratch/large/userexternal/lcerovaz/"
                                "diffmechint/activations_val"),
                   help="Root of val activation shards "
                        "(<val_root>/<cond>/step_<N>/<L>_<T>.h5).")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Where to write per-SAE JSON + master aggregate files.")
    p.add_argument("--include_mid", action="store_true",
                   help="Also evaluate mid-stage (30M token) checkpoints. "
                        "Default: only the canonical 'final_*' stage.")
    p.add_argument("--batch_size", type=int, default=4096,
                   help="Token batch size during eval forward passes.")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Cap evaluated tokens per SAE (default: all tokens "
                        "in the shard, ~1.28M for 5/class × 256 tok).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the planned (sae_ckpt, val_shard) pairs and exit. "
                        "Used by the coherence-check subagents.")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate at most this many SAEs (for smoke tests).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.dry_run:
        warn("CUDA not available — eval will be very slow.")

    if not args.sae_root.exists():
        error(f"sae_root does not exist: {args.sae_root}")
        return 1
    if not args.val_root.exists():
        error(f"val_root does not exist: {args.val_root}")
        return 1

    info(f"Discovering SAE checkpoints under {args.sae_root} …")
    saes = discover_saes(args.sae_root)
    info(f"  found {len(saes)} sae_weights.safetensors")
    if not args.include_mid:
        saes = [s for s in saes if s["stage"] == "final"]
        info(f"  after filtering to stage=final: {len(saes)}")
    if args.limit is not None:
        saes = saes[: args.limit]
        info(f"  limited to first {len(saes)} (smoke-test mode)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_jsonl = args.output_dir / "aggregate.jsonl"
    aggregate_csv = args.output_dir / "aggregate.csv"

    # Pre-pass: planned pairs (printed for both dry-run and real runs so logs
    # are self-documenting; subagents can grep this).
    plan: list[tuple[dict, Path]] = []
    missing_shards = 0
    for s in saes:
        md = s["metadata"]
        cond = md.get("condition")
        layer = int(md.get("layer", -1))
        t_bin = int(md.get("t_bin", -1))
        shard = val_shard_path(args.val_root, cond, s["dit_step"], layer, t_bin)
        plan.append((s, shard))
        marker = "OK" if shard.exists() else "MISSING"
        if not shard.exists():
            missing_shards += 1
        info(f"  [{marker:7}] sweep={s['sweep_id']:30s} cond={cond:6s} "
             f"L{layer}_T{t_bin} dit_step={s['dit_step']:>6d} "
             f"stage={s['stage']:5s} k={md.get('k')}  →  {shard}")
    info(f"Planned {len(plan)} (SAE, val_shard) pairs; missing shards: {missing_shards}")

    if args.dry_run:
        info("--dry_run set → exiting without loading SAEs / val shards.")
        return 0 if missing_shards == 0 else 2

    if missing_shards > 0:
        error(f"Refusing to run: {missing_shards} planned val shards are missing.")
        return 3

    rows: list[dict] = []
    t_start = time.perf_counter()
    for i, (s, shard) in enumerate(plan):
        md = s["metadata"]
        cond = md.get("condition")
        layer = int(md.get("layer"))
        t_bin = int(md.get("t_bin"))
        k = int(md.get("k"))
        d_sae = int(md.get("d_sae"))
        ckpt_dir = s["ckpt_dir"]
        t0 = time.perf_counter()
        info(f"[{i + 1:3d}/{len(plan)}] sweep={s['sweep_id']} cond={cond} "
             f"L{layer}_T{t_bin} step={s['dit_step']} stage={s['stage']} k={k}")

        try:
            sae = TopKTrainingSAE.load_from_disk(str(ckpt_dir), device=str(device))
        except Exception as exc:
            error(f"  load failed: {exc!r}")
            continue

        # Sanity: metadata vs cfg vs loaded SAE.
        if sae.cfg.k != k:
            warn(f"  k mismatch: metadata={k} vs sae.cfg.k={sae.cfg.k}")
        if int(sae.cfg.d_sae) != d_sae:
            warn(f"  d_sae mismatch: metadata={d_sae} vs sae.cfg.d_sae={sae.cfg.d_sae}")

        try:
            m = evaluate_one(
                sae, shard,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                device=device,
            )
        except Exception as exc:
            error(f"  eval failed on {shard.name}: {exc!r}")
            continue
        finally:
            del sae
            torch.cuda.empty_cache()

        row = {
            "sweep_id": s["sweep_id"],
            "condition": cond,
            "layer": layer,
            "t_bin": t_bin,
            "dit_step": s["dit_step"],
            "stage": s["stage"],
            "k": k,
            "d_sae_meta": d_sae,
            "ckpt_dir": str(ckpt_dir),
            "val_shard": str(shard),
            **m,
        }
        rows.append(row)

        # Per-SAE JSON next to the aggregate (mirrors the SAE tree shape).
        per_sae_dir = args.output_dir / s["sweep_id"] / cond / f"L{layer}_T{t_bin}" / f"step_{s['dit_step']:06d}"
        per_sae_dir.mkdir(parents=True, exist_ok=True)
        (per_sae_dir / f"{s['stage']}.json").write_text(json.dumps(row, indent=2))

        with aggregate_jsonl.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        ok(f"  ev={m['ev']:.4f} mse={m['mse']:.5g} cos={m['recon_cosine']:.4f} "
           f"l0={m['l0_mean']:.1f} dead={m['dead_pct']*100:.2f}% "
           f"  ({time.perf_counter() - t0:.1f}s)")

    # Master CSV.
    if rows:
        with aggregate_csv.open("w", newline="") as fh:
            cols = list(rows[0].keys())
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        ok(f"Wrote {len(rows)} rows to {aggregate_csv}")
    else:
        warn("No rows produced — check earlier errors.")

    info(f"Total wall: {(time.perf_counter() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
