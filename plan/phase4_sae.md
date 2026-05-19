# Phase 4 — SAE Training (multi-checkpoint sweep)

Train TopK SAEs across all 27 cells × 7 DiT checkpoints via SAELens, with cold-start per stage. See [README](README.md) for navigation.

## Toolkit choice (revised 2026-04-29)

The first research pass concluded "avoid SAELens — coupled to
TransformerLens". A direct re-read of SAELens 6.x (3 verification
subagents, recorded in commit history) reverses that conclusion:

> **`SAETrainer` (`sae_lens/training/sae_trainer.py:67`) is genuinely
> model-agnostic.** It accepts `data_provider: Iterator[Tensor]` and
> never instantiates a model. The TransformerLens coupling lives only
> inside `LanguageModelSAETrainingRunner` and `CacheActivationsRunner`,
> which we bypass.

Concrete implications:
- We feed pre-computed activations from our HDF5 shards directly into
  `SAETrainer.fit()`. ~50 LoC adapter.
- TransformerLens is pulled as a transitive dependency but never
  executed on the DiT path. ~200 MB of mostly dead weight in the env.
- We get TopK / BatchTopK / Matryoshka / Gated / JumpReLU / Standard
  SAE variants by config, plus built-in multi-checkpoint
  (`n_checkpoints` arg) — exactly the warm-start primitive we wanted to
  build manually for Xu et al. 2412.17626.
- Save format is the standard SAELens `safetensors + JSON` pair, so
  trained SAEs load directly into `sae_vis`, `sae_dashboard`, and
  Neuronpedia for paper-grade visualization.

**Decision: SAELens is the primary library for Phase 4.** It is the
de-facto standard, the integration cost is ~50 LoC, and the ecosystem
benefits (visualization, format portability, active maintenance) are
real and documented.

`dictionary_learning` (saprmarks) stays as a vendored fallback under
`third_party/dictionary_learning/`. We pin a commit and import only the
TopK / BatchTopK trainer files in case SAELens introduces a DiT-blocking
regression. nnsight remains pinned as a runtime dep for Phase 6 EAP
circuit work — we do not use it for SAE training.

## SAE architecture — via SAELens

`src/diffmechint/sae/topk.py` is a thin wrapper around
`sae_lens.TopKTrainingSAE` and `sae_lens.BatchTopKTrainingSAE`. We do not
re-implement the SAE; we configure it.

```python
from sae_lens import (
    SAETrainer,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.config import SAETrainerConfig

def build_sae(d_in: int, d_sae: int, k: int, variant: str = "batch_topk"):
    if variant == "topk":
        return TopKTrainingSAE(TopKTrainingSAEConfig(
            d_in=d_in, d_sae=d_sae, k=k,
            normalize_activations="expected_average_only_in",
        ))
    elif variant == "batch_topk":
        return BatchTopKTrainingSAE(BatchTopKTrainingSAEConfig(
            d_in=d_in, d_sae=d_sae, k=k,
            normalize_activations="expected_average_only_in",
        ))
    raise ValueError(variant)
```

Default `variant="batch_topk"` (better dead-feature rate per Bussmann
et al.); `topk` for parity with the canonical Anthropic recipe.

## HDF5 → SAELens data provider

`src/diffmechint/sae/data_provider.py` — adapter from our Phase 3
activation buffer shards to a `data_provider: Iterator[Tensor]`:

```python
def hdf5_provider(
    shard_paths: list[Path],
    batch_size: int = 4096,
    device: str = "cuda",
    flatten_tokens: bool = True,
) -> Iterator[torch.Tensor]:
    """Yield (batch_size, D) tensors from Phase 3 HDF5 cells.

    flatten_tokens=True collapses (N, T, D) → (N*T, D); each spatial
    token contributes one SAE training sample. Set False to keep token
    structure when training a token-aware SAE variant.
    """
    for path in shard_paths:
        with h5py.File(path, "r") as f:
            arr = torch.from_numpy(f["activations"][()]).float()  # (N, T, D) fp16→fp32
            if flatten_tokens:
                arr = arr.reshape(-1, arr.shape[-1])              # (N*T, D)
            for batch in arr.split(batch_size):
                yield batch.to(device, non_blocking=True)
```

## Per-checkpoint training strategy — REVISED 2026-05-17

**Decision: cold-start every stage; warm-start removed as a dead path.**

Original plan was to warm-start the SAE for each DiT ckpt from the previous
ckpt's SAE weights, as in Xu et al. 2412.17626 (`Superposition09m/SAE-Track`).
A direct audit of their code revealed that:

- They transfer **only SAE weights** (no optimizer, no LR scheduler, no input
  normalization stats, no firing-rate EMA). The `weights_only` mode in our
  `warm_started_sweep` exactly replicates this.
- Their cold-start budget is **300 M tokens** (16× our 20 M); their DiT-ckpt
  gaps are tiny (Δ=20 ckpts out of 154); the activation drift between stages
  is qualitatively small.
- They never numerically benchmarked warm vs cold — only a qualitative
  Figure 11 in Appendix I shows "warm converges faster". No EV/L0/dead
  comparison.

In our setup (SiT-B/2 DiT, 7 fractional ckpts with very large gaps from 4 k
to 200 k step, 20 M cold budget) the warm-start replication produced
**catastrophic dead-feature collapse**: 94 % dead at stage 1+, EV plateau
at 0.83. We tested a stronger variant ("Fix B") that additionally carries
Adam optimizer state — it mitigated marginally (94 → 91 % dead) but did
not fix the underlying problem (activation drift between DiT ckpts > what
a few-million-token warm budget can recover from).

**Cold-start per stage** (`warm_mode='cold'`, every stage cold with full
20 M token budget) drops dead-features from 94 % to **0.16 %** and lifts
EV from 0.83 to **0.89** on the test cell (`sd_vae / L6 / T1`, k=64).

Implementation (`src/diffmechint/sae/trainer.py:warm_started_sweep`):

```python
warm_started_sweep(
    sae_factory,
    activation_shards_per_dit,   # 7 (label, shards) tuples in DiT-step order
    out_root=...,
    base_total_samples=20_000_000,
    warm_total_samples=20_000_000,   # for cold mode this equals base
    warm_mode='cold',                # canonical; alt: 'weights_only' (diagnostic)
    lr=3e-4,
    lr_warm_up_steps=200,
)
```

Cost trade-off: cold per stage = 7 × cold budget = 7 × 20 M = 140 M tok/chain,
vs the warm-start dream of ~50 M tok/chain. Wall time per chain on 1× A100:
~40 min (cold) vs ~14 min (warm). Full sweep (27 chains) on 3 parallel jobs:
~6 h with cold, ~2 h with warm. The 4× compute overhead is the price for
SAE quality that's actually usable for downstream Phase 5 / Phase 6 analyses.

Earlier warm-start scaffolding (`weights_only` mode) is retained ONLY as a
diagnostic baseline for reproducing the failure mode on demand.

## Sweep dimensions

Per the proposal: per condition, train SAE at `(layer, timestep, k)`
cells where `layer ∈ {25 %, 50 %, 75 % depth}`,
`timestep ∈ {0.025, 0.20, 0.50}`, `k ∈ {16, 32, 64}`,
`d_sae = 16384`. That is 27 SAEs per (condition, checkpoint). With
4 conditions × 7 checkpoints × 27 cells = **756 SAEs total**. Warm-start
brings the 7-checkpoint cost down to ~3× a single-checkpoint cost.

## SAE compute budget

Single SAE: ~10 GPU-h on a single A100, dominated by activation
streaming from disk. Total ~7560 A100-h, but parallelize embarrassingly
across 4 GPUs ⇒ ~320 GPU-h wall-clock for the full sweep on CINECA.
**Plan:** run the canonical cell `(k=32, layer-50 %, t=0.20)` first
across all `(condition, checkpoint)` pairs (28 SAEs); expand to the
full 27-cell grid only when results justify.

## SAE-side metrics

Per `src/diffmechint/sae/eval.py`:
- Reconstruction cosine and L2 (SAELens emits these natively at every
  checkpoint).
- Feature density (fraction non-zero per token).
- Label-σ per feature: per Revelio, std of class-label distribution
  across the top-k activating samples — proxy for monosemanticity.
- Live / dead feature count.
- Per-feature **RIEBench causal-edit score** (One-Step-is-Enough): zero
  out feature, run a small generation, measure CLIP-similarity drop on
  the targeted concept.

Each SAE's eval JSON lands at
`outputs/<run_id>/saes/<vae>/<ckpt>/<layer>_<t>_<k>/metrics.json`.

## Visualization & format portability

SAELens-trained SAEs save as `safetensors + cfg.json`. They drop
straight into:
- **`sae_vis`** — per-feature dashboards, used by Anthropic Circuits
  Updates.
- **`sae_dashboard`** — Neuronpedia-style web viewer.
- **Hugging Face Hub** — public release alongside the paper, per the
  expected-results §6 of the proposal.

This is the principal ergonomic argument for SAELens over a from-
scratch trainer: zero-friction handoff to community visualization
tooling.

## Phase 4 acceptance gate

- 1 SAE trained end-to-end on synthetic activations (smoke) — recon
  cosine > 0.85 within 1 k steps.
- 1 SAE trained on a real DiT-B/2 SD-VAE checkpoint at canonical cell —
  recon cosine > 0.85, density 1–5 %, dead-feature count < 5 %.
- 28-SAE warm-started sweep (4 conditions × 7 ckpts at canonical cell)
  completes in < 320 A100-h on CINECA.

## Phase 4.5 — Causal faithfulness gate (added 2026-05-19)

### Motivation

Phase 4 measures **reconstruction quality on raw residuals** (val EV / MSE /
cosine / dead %). Those numbers say the SAE can predict the activation it
intercepts, not that the SiT's downstream forward survives encode→decode
substitution. A SAE with val EV = 0.97 can still drop the missing 3 % of
variance into directions the SiT actually uses for generation, and downstream
phases (linear probes, EAP circuits) built on top of such an SAE would
interpolate over noise.

We therefore insert a **causal-faithfulness gate** between Phase 4 and Phase 5:
substitute residual-stream activations with their SAE reconstructions at one
(layer, t-bin) cell during sampling, and measure the resulting FID drop.
Cells where ΔFID is small are "faithful" and are admissible bases for
Phase 5 / Phase 6 work; cells where ΔFID is large are flagged and not used
for downstream interpretation.

### 4.5a — Matryoshka substitution-FID grid

Single-shot scan on the production matryoshka SAE family (the Phase 4 winner,
see E04 `cool-scene-6995` and I04 `old-mode-5126`):

- **SAE**: `sae_matryoshka_k256_d32k`, `d_sae = 32 768`, K = 256, prefixes
  `(4 096, 8 192, 16 384, 32 768)` — variant chosen because it dominated
  TopK and BatchTopK on val EV at every depth (E04 / E05).
- **Cells**: 3 conditions × 3 layers × 3 t-bins = **27 cells**, all at the
  fully-converged DiT-200k checkpoint. (Pre-convergence DiT stages are
  excluded — Phase 4 verified they are unstable, see E04 §3.)
- **Reference**: ImageNet val 50 k Clean-FID stats (already prefetched, same
  reference as `post_hoc_fid.py`).
- **Source data for the SAE encode side**: the live SiT sampling trajectory —
  no cached val activations are used, because we are measuring an
  *intervention on the generation forward*. The "val" qualifier refers to
  the FID reference set, not to the activation source.
- **Intervention semantics**: hook the SiT block at the chosen layer; at
  every sampling step, if the current diffusion-time `t` is within ±0.01 of
  the chosen bin centre `{0.025, 0.20, 0.50}`, replace the block output
  with `sae.decode(sae.encode(output))`. Outside the bin, the forward is
  untouched. With a 250-step sampler this means the substitution fires at
  ~5 consecutive sampling steps per generation — a localized
  point-intervention, matching the same narrow band the SAE was trained on
  in Phase 3 / Phase 4.
- **Baselines**: 3 jobs (one per condition) generating with the same SiT,
  same sampler, same seeds, but with the hook disabled — provides the
  baseline FID for the ΔFID delta. The Phase 2 `post_hoc_fid.py` numbers
  cannot be reused as baselines because we want exact seed / batch parity
  with the substitution runs.

### Headline observable

| Symbol | Meaning |
|---|---|
| `FID_baseline(cond)` | Clean-FID of unhooked SiT at DiT-200k, 5 k samples |
| `FID_sub(cond, L, t)` | Clean-FID with matryoshka substitution at (L, t) |
| `ΔFID(cond, L, t)` | `FID_sub − FID_baseline` (positive = quality degraded) |

A 3×3×3 ΔFID tensor per condition is the only headline artifact. Cells with
`ΔFID < 2.0` are flagged "faithful" (green-light for Phase 5); cells with
`ΔFID > 5.0` are flagged "unreliable" (excluded from downstream
interpretation); the band in between is "marginal" (revisit before scaling).

### Compute and runtime

5 000 samples per cell × 30 cells (27 substitution + 3 baselines) × ~18 min
on A100 = **~9 GPU·h** total (∼3 h wall if 4–6 jobs run concurrent).
ImageNet-val 50 k Clean-FID stats are already cached so no internet is
needed from compute nodes.

### Implementation

- `scripts/eval/sae_substitution_fid.py` — single-cell driver. Reuses
  `post_hoc_fid.py`'s sampling loop; adds a substitution hook keyed on
  `(layer, t_bin_center, t_tol)`. Reads the SAE from the production
  `sae_matryoshka_k256_d32k/<cond>/L<L>_T<T>/step_200000/final_*/` dir.
- `slurm/sae_substitution_fid.slurm` — sbatch template, single GPU.
- `slurm/launch_substitution_fid.sh` — submits all 30 jobs in one call.

### Acceptance gate

- All 30 sample directories exist and contain 5 000 PNGs each.
- All 30 FID values written to a single `outputs/phase4_5a_subst_fid/aggregate.csv`.
- A ΔFID heatmap PNG per condition is produced for the experiment node.

### Downstream consequence

The flagged set "faithful cells" produced by 4.5a is the **only** subset of
(cond, L, t) that Phase 5 probes against. If too few cells are faithful,
Phase 5 design is revisited (e.g. select a different SAE prefix, or fall
back to TopK k=128 if matryoshka itself is the failure source — testable
with one additional batch on the worst three cells).
