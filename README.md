# diffmechint

**Semantic Geometry of Diffusability — Mechanistic Atlas of Tokenizer Interventions**

> [!WARNING]
> 🚧 **Repository under active construction (May 2026).**
> Phase 0–5 scaffolding is complete and unit-tested, but the K=4 SiT-B/2 ImageNet
> sweep is currently mid-flight. Numbers, configs and APIs in this repo will
> change over the next ~2 weeks. **Do not** treat current `metrics/` or `samples/`
> outputs as paper-grade. See `CHECKLIST.md` §2.11 for live status.

> [!IMPORTANT]
> Codebase for the paper *"The Semantic Geometry of Diffusability:
> A Mechanistic Atlas of How Latent Geometry Shapes Diffusion Transformer
> Learning"* (Mancusi / Strano, Sapienza — NVIDIA Academic Grant submission,
> ICLR/ICML 2027 target).

The experiment trains a **single SiT (Scalable Interpolant Transformer) backbone
with Flow-Matching + Optimal-Transport** on **K = 5 controlled VAE/tokenizer
variants** on ImageNet-256, then runs a fixed mech-int protocol — k-SAEs,
layer × timestep linear probes, sparse feature circuits via EAP — on each
condition, at 7 fractional training checkpoints.

The output is the first **semantic atlas of diffusability**: which features
emerge in the DiT's residual stream, in what order, and whether the
coarse-to-fine schedule is tokenizer-invariant or tokenizer-shaped.

> [!NOTE]
> See [`PLAN.md`](PLAN.md) for the full implementation plan and
> [`CHECKLIST.md`](CHECKLIST.md) for live progress per phase.

---

## Quick start

```bash
uv sync --extra dev
uv run pytest tests/
```

GPU smoke for the tokenizer adapters (real HF download, ~2 min on 1 A100):

```bash
uv run python scripts/smoke_adapters_gpu.py
```

1k-step SiT-B/2 smoke run with synthetic latents on a single GPU:

```bash
uv run python -m diffmechint.training.train \
    trainer.max_steps=1000 \
    +data.batch_size=32 +data.n_samples=8192 \
    +ckpt_dir=outputs/smoke
```

> [!TIP]
> On CINECA Leonardo, point the HF cache to `$FAST/lcerovaz/hf_cache` (already
> warmed for the 4 working VAE adapters) and the latent / checkpoint dirs to
> `$SCRATCH/diffmechint/`. `$WORK` quota is too small for the precomputed
> latents (~10 GB per condition).

---

## Layout

```
diffmechint/
├── PLAN.md                — single source of truth for design + phases
├── CHECKLIST.md           — per-phase progress tracker
├── pyproject.toml         — uv-managed deps, single source of truth
├── conf/                  — Hydra configs (tokenizer, model, transport, trainer, sae, probe, callbacks)
├── src/diffmechint/
│   ├── tokenizers/        — adapters + registry (sd_vae, eq_vae, repa_e, dc_ae_1_0, rae)
│   ├── sit/               — vendored willisma/SiT @ cbde832, MIT (with diffmechint patches)
│   ├── training/
│   │   ├── sit_module.py        — LightningModule (FM-OT + EMA + resume_from)
│   │   ├── data.py              — CachedLatentDataModule (HDF5 + z-score + holdout split)
│   │   ├── precompute_latents.py — image → VAE → HDF5 + per-feature stats.json
│   │   ├── checkpointing.py     — fractional ckpt callback (7 frac steps + EMA shadow)
│   │   └── callbacks/           — SampleCallback (PNG grids) + MiniFIDCallback (clean-fid)
│   ├── hooks/             — ResidualStreamTap, timestep router, ActivationBuffer
│   ├── sae/               — SAELens-backed SAE training + warm-start sweep
│   ├── probing/           — concepts registry + Revelio-grid linear probes
│   ├── circuits/          — EAP + faithfulness + SHIFT (Phase 6, pending)
│   ├── analysis/          — Hungarian dictionary overlap + temporal atlas (Phase 7, pending)
│   └── utils/             — rich console helpers
├── scripts/
│   ├── prefetch_tokenizers.py     — warm $FAST HF cache on login node
│   ├── precompute_one_imagenet.sh — one tokenizer × ImageNet train → HDF5 shards
│   ├── round_trip_psnr_imagenet.py — adapter acceptance: PSNR > 22 dB on real ImageNet
│   ├── train_sit_full.sh          — DDP 2× A100 SiT training (with optional resume_from)
│   ├── post_hoc_fid.py            — Mini-FID 5k vs ImageNet val 50k for every EMA ckpt
│   ├── prefetch_cleanfid.sh       — one-time Inception cache + reference stats
│   ├── extract_metrics.py         — Lightning CSV → metrics/{train,validation}/*.csv
│   └── plot_run.py                — palette-B PNG plots from metrics/
├── slurm/                 — CINECA SLURM templates (precompute, train, FID)
├── tests/                 — pytest suite (88 tests green)
└── outputs/               — gitignored; symlinked to $FAST/diffmechint/outputs/ on CINECA
```

### Standardized run output layout

> [!IMPORTANT]
> Every `<run_id>/` produced by `scripts/train_sit_full.sh` uses this exact layout:
>
> ```
> runs/<run_id>/
> ├── checkpoints/
> │   ├── step_NNNNNNNN.safetensors          ← live model weights
> │   ├── step_NNNNNNNN_ema.safetensors      ← EMA shadow (analysis target)
> │   └── step_NNNNNNNN_metadata.json
> ├── samples/
> │   └── step_NNNNNNNN_cfg{1p0,4p0}.png     ← 4×4 grid, ODE dopri5 50 steps
> ├── lightning_logs/version_0/metrics.csv   ← raw Lightning log
> ├── metrics/                               ← extract_metrics.py output
> │   ├── train/{loss_step,loss_epoch}.csv
> │   ├── validation/{loss,fid}.csv
> │   └── summary.json
> └── plots/                                 ← plot_run.py output (palette B)
>     ├── train_loss.png  val_loss.png  fid.png  summary.png
> ```
>
> If you write a new training script, **conform to this layout** so
> `extract_metrics.py` and `plot_run.py` work without modification.

---

## Conditions (K = 5)

The four diffusability clusters from the proposal, plus the SD-VAE baseline:

| condition  | cluster                          | hf repo / source                                  | adapter status |
|------------|----------------------------------|---------------------------------------------------|----------------|
| sd_vae     | baseline                         | `stabilityai/sd-vae-ft-mse`                       | 🟢 working      |
| eq_vae     | spectral / equivariance          | `zelaki/eq-vae-ema`                               | 🟢 working      |
| repa_e     | semantic alignment (joint VAE)   | `REPA-E/e2e-sdvae-hf`                             | 🟢 working      |
| dc_ae_1_0  | information-ordered bottleneck   | `mit-han-lab/dc-ae-f32c32-in-1.0-diffusers`       | 🟢 working      |
| rae        | discriminative encoder (DINOv2)  | `nyu-visionx/rae-dinov2-base-vitxl-n08-256`       | 🟡 scaffold     |

DC-AE 1.5 enters as a 6th condition once `dc-ai-projects/DC-Gen` is released.

> [!NOTE]
> Round-trip PSNR on 256 real ImageNet images (run via
> [`scripts/round_trip_psnr_imagenet.py`](scripts/round_trip_psnr_imagenet.py)):
> sd_vae **25.11 dB**, eq_vae **24.14 dB**, repa_e **24.23 dB**,
> dc_ae_1_0 **23.00 dB** — all ≥ 22 dB threshold (PLAN §14).

> [!CAUTION]
> The four working VAEs have **different per-channel σ** on ImageNet
> (sd_vae 0.83, eq_vae **2.66**, repa_e 0.80, dc_ae_1_0 **3.08**).
> Without runtime z-score normalization (`CachedLatentDataModule(normalize=True)`,
> default), DiT training across the K=4 conditions is **not matched-compute
> comparable**. The full per-feature stats live in
> `$SCRATCH/diffmechint/latents/<tok>/stats.json`.

---

## Pipeline conventions

### Latent stats schema (`stats.json` v1)

`stats.json` is the single source of truth for de/normalization and downstream
DiT setup — no consumer should hardcode latent layout:

```json
{
  "kind": "spatial",          // "spatial" (B,C,H,W) | "sequence" (B,T,D)
  "feature_axis": 1,          // axis on the batched tensor
  "feature_dim": 4,           // C for spatial, D for sequence
  "input_size": 32,           // H for spatial, T for sequence
  "scaling_factor": 0.18215,  // applied during VAE.encode (already in HDF5)
  "suggested_patch_size": 2,  // SiT patchify hint
  "per_feature_mean": [...],
  "per_feature_std":  [...]
}
```

### De/normalization chain (reversible)

```
ENCODE   image → vae.encode → mean*scaling_factor → z (HDF5, fp16)
LOAD     z → (z - μ)/σ                                  ← stats.json
SAMPLE   z̃ → z̃*σ + μ → ÷scaling_factor → vae.decode → image
```

> [!IMPORTANT]
> `MiniFIDCallback` is **disabled by default** on K=4 training jobs
> (`callbacks=sample_only`). Reason: `clean-fid` requires Inception weight
> downloads and a built reference stat cache that the compute nodes can't
> reach without the squid proxy, and a partial failure caused 30-min NCCL
> deadlocks. **Use `scripts/post_hoc_fid.py` after training instead** —
> it computes Mini-FID for every saved EMA checkpoint and writes
> `metrics/validation/fid.csv`. Run via `slurm/post_hoc_fid.slurm`.

---

## Stack

Python 3.11 · PyTorch 2.6 · Lightning 2.4+ · Hydra 1.3 · uv · timm · diffusers
0.30+ · SAELens 6.x · sklearn · h5py · safetensors · clean-fid · CUDA 12.x.

Hardware path:
- **CINECA Leonardo** (2× / 4× A100 64 GB) for matched-compute DiT training and the
  full SAE / probe / circuit sweep.
- **Local 3090 / 2080 Ti** (`100.124.107.92` via Tailscale) for adapter smokes
  and short DiT runs.
- **NVIDIA H100 ×8** when the Academic Grant lands; trainer config in
  `conf/trainer/nvidia_8xh100.yaml` is already wired.

---

## Phase status (high-level)

| Phase | Subject                                             | Status                              | Tests |
|-------|-----------------------------------------------------|-------------------------------------|-------|
| 0     | Repo bootstrap + vendor SiT                         | ✅ done                             | 8     |
| 1     | Tokenizer adapters + latent precompute (4 × 1.28 M ImageNet → HDF5) | ✅ done | 17 |
| 2     | SiT training pipeline (FM-OT, fractional ckpts, val + sample callbacks) | 🟡 mid-flight: K=4 training running on Leonardo (DDP 2× A100) | 12 |
| 3     | Activation extraction (hooks + buffer)              | ✅ done                             | 23    |
| 4     | SAE training (SAELens-backed, warm-start)           | ✅ scaffolding done                 | 9     |
| 5     | Linear probes (Revelio grid)                        | ✅ scaffolding done                 | 18    |
| 6     | Sparse feature circuits via EAP                     | 🔴 pending                          | —     |
| 7     | Cross-condition analysis (Hungarian + temporal)     | 🔴 pending                          | —     |
| 8     | Audio extension (deferred)                          | 🔴 pending                          | —     |

**Total: 88/88 unit tests green.** Per-phase verification commands and
acceptance gates live in `PLAN.md` §14.

---

## License

MIT for repo code. Vendored upstream code preserves its own LICENSE files:
- `src/diffmechint/sit/LICENSE.txt` — SiT (Meta, MIT)
- SAELens, transformer-lens, dictionary_learning et al. are runtime / fallback
  deps, used per their own licenses.

---

## TODO — what's left to do

The full plan lives in `PLAN.md` and per-item progress in `CHECKLIST.md`. The
short list:

### Code-only (no GPU / data dependency, can be done locally)

- [ ] **(Phase 6)** EAP via `nnsight`, sparse feature circuits, faithfulness +
      completeness + minimality triplet, SHIFT ablation, RIEBench score.
- [ ] **(Phase 7)** Hungarian-matched cross-tokenizer dictionary overlap +
      temporal-atlas plotting (phase transitions, swing-by, dips).
- [ ] **(1.7-RAE)** Vendor the RAE ViT decoder from `bytetriper/RAE` so
      `RAEAdapter.load/encode/decode` actually run.
- [ ] **(1.10)** `TokenGridAdapter` — only needed once non-grid latents
      (RAE / MAETok) are wired into the training loop.
- [ ] **(2.3)** Optional Laplace-logSNR `t_sampler` — uniform-t works for the
      smoke; lands when real ImageNet runs do.
- [ ] **(2.x)** Fix `FractionalCheckpoint` to also save optimizer + scheduler +
      RNG state (Lightning-native `.ckpt`) so future crashes can be **fully
      resumed** instead of weight-resumed (current `+resume_from=` only loads
      live + EMA weights, optimizer momenta lost).
- [ ] **(2.y)** Fix `MiniFIDCallback` to work in DDP — needs
      `torch.distributed.barrier()` after rank-0-only block to avoid NCCL
      allreduce timeout. For now the callback is replaced by `post_hoc_fid.py`.

### Gated on K = 4 ImageNet runs completing (in flight now)

- [ ] **(2.11)** Full 200 k-step DiT-B/2 runs on **sd_vae · eq_vae · repa_e**
      (resumed from step 50 k after NCCL crash) plus **DC-AE 1.0 with
      patch_size=1** (SiT-B/1 from scratch). All 4 currently RUNNING on Leonardo.
- [ ] **(2.12)** Per-condition Mini-FID curves via `post_hoc_fid.py` once each
      training finishes.
- [ ] **(4.9)** Canonical-cell SAE on a real DiT ckpt: recon cosine > 0.85,
      density 1–5 %, dead-feature count < 5 %.
- [ ] **(4.10)** 28-SAE warm-started sweep (4 conditions × 7 ckpts).
- [ ] **(4.11)** Full 27-cell sweep (756 SAEs) — gated on (4.10) results.
- [ ] **(4.12)** Verify saved SAEs load into `sae_vis` / `sae_dashboard`.
- [ ] **(5.7)** Real 5 × 3 × 3 probe-accuracy heatmap per condition.
- [ ] **(5.8)** Cross-condition probe-peak migration figure for Claim 1.

### Optional / stretch

- [ ] **(1.8 / 1.9)** Add `MAETok` and `VA-VAE` adapters as a 6th and 7th
      condition (their HF checkpoints exist; ~50 LoC each).
- [ ] **(Phase 8)** Audio extension (Semantic-VAE / SALAD-VAE) — separate
      branch, do not block vision.
- [ ] **DC-AE 1.5** condition as soon as `dc-ai-projects/DC-Gen` releases.
