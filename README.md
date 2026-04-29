# diffmechint

**Semantic Geometry of Diffusability — Mechanistic Atlas of Tokenizer Interventions**

Codebase for the paper *"The Semantic Geometry of Diffusability: A Mechanistic
Atlas of How Latent Geometry Shapes Diffusion Transformer Learning"* (Mancusi /
Strano, Sapienza — NVIDIA Academic Grant submission, ICLR/ICML 2027 target).

The experiment trains a **single SiT (Scalable Interpolant Transformer) backbone
with Flow-Matching + Optimal-Transport** on **K = 5 controlled VAE/tokenizer
variants** on ImageNet-256, then runs a fixed mech-int protocol — k-SAEs,
layer × timestep linear probes, sparse feature circuits via EAP — on each
condition, at 7 fractional training checkpoints.

The output is the first **semantic atlas of diffusability**: which features
emerge in the DiT's residual stream, in what order, and whether the
coarse-to-fine schedule is tokenizer-invariant or tokenizer-shaped.

See [`PLAN.md`](PLAN.md) for the full implementation plan and
[`CHECKLIST.md`](CHECKLIST.md) for live progress per phase.

---

## Quick start

```bash
uv sync --extra dev
uv run pytest tests/
```

GPU smoke for the tokenizer adapters (real HF download):

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

---

## Layout

```
diffmechint/
├── PLAN.md                — single source of truth for design + phases
├── CHECKLIST.md           — per-phase progress tracker
├── pyproject.toml         — uv-managed deps, single source of truth
├── conf/                  — Hydra configs (tokenizer, model, transport, trainer, sae, probe)
├── src/diffmechint/
│   ├── tokenizers/        — adapters + registry (sd_vae, eq_vae, repa_e, dc_ae_1_0, rae)
│   ├── sit/               — vendored willisma/SiT @ cbde832, MIT
│   ├── training/          — SiTLightningModule + FM-OT trainer + fractional ckpts
│   ├── hooks/              — ResidualStreamTap, timestep router, ActivationBuffer
│   ├── sae/               — SAELens-backed SAE training + warm-start sweep
│   ├── probing/           — concepts registry + Revelio-grid linear probes
│   ├── circuits/          — EAP + faithfulness + SHIFT (Phase 6, pending)
│   ├── analysis/          — Hungarian dictionary overlap + temporal atlas (Phase 7, pending)
│   └── utils/             — rich console helpers
├── scripts/               — local CLI drivers (smoke_adapters_gpu.py, migrate-session.sh, ...)
├── slurm/                 — CINECA SLURM templates (Phase 2 full run, pending)
├── tests/                 — pytest suite (81 tests green as of latest commit)
└── outputs/               — gitignored; symlinked to $FAST/diffmechint/outputs/ on CINECA
```

---

## Conditions (K = 5)

The four diffusability clusters from the proposal, plus the SD-VAE baseline:

| condition  | cluster                          | hf repo / source                                  | status |
|------------|----------------------------------|---------------------------------------------------|--------|
| sd_vae     | baseline                         | `stabilityai/sd-vae-ft-mse`                       | 🟢      |
| eq_vae     | spectral / equivariance          | `zelaki/eq-vae-ema`                               | 🟢      |
| repa_e     | semantic alignment (joint VAE)   | `REPA-E/e2e-sdvae-hf`                             | 🟢      |
| dc_ae_1_0  | information-ordered bottleneck   | `mit-han-lab/dc-ae-f32c32-in-1.0-diffusers`        | 🟢      |
| rae        | discriminative encoder (DINOv2)  | `nyu-visionx/rae-dinov2-base-vitxl-n08-256`        | 🟡 scaffold |

DC-AE 1.5 enters as a 6th condition once `dc-ai-projects/DC-Gen` is released.

GPU round-trip PSNR > 25 dB on the canonical `pytorch/hub` dog image
(28.10 / 27.30 / 27.03 / 26.20 dB respectively) — see
[`scripts/smoke_adapters_gpu.py`](scripts/smoke_adapters_gpu.py).

---

## Stack

Python 3.11 · PyTorch 2.6 · Lightning 2.4+ · Hydra 1.3 · uv · timm · diffusers
0.30+ · SAELens 6.x · sklearn · h5py · safetensors · CUDA 12.x.

Hardware path:
- **CINECA Leonardo** (4× A100 64 GB) for matched-compute DiT training and the
  full SAE / probe / circuit sweep.
- **Local 3090 / 2080 Ti** (`100.124.107.92` via Tailscale) for adapter smokes
  and short DiT runs.
- **NVIDIA H100 ×8** when the Academic Grant lands; trainer config in
  `conf/trainer/nvidia_8xh100.yaml` is already wired.

---

## Phase status (high-level)

| Phase | Subject                                             | Status    | Tests |
|-------|-----------------------------------------------------|-----------|-------|
| 0     | Repo bootstrap + vendor SiT                         | ✅ done    | 8     |
| 1     | Tokenizer adapters + latent precompute              | ✅ done    | 11    |
| 2     | SiT training pipeline (FM-OT, fractional ckpts)     | ✅ scaffolding done; 1k-step smoke ✓ | 12 |
| 3     | Activation extraction (hooks + buffer)              | ✅ done    | 23    |
| 4     | SAE training (SAELens-backed, warm-start)           | ✅ scaffolding done | 9    |
| 5     | Linear probes (Revelio grid)                        | ✅ scaffolding done | 18   |
| 6     | Sparse feature circuits via EAP                     | 🔴 pending  | —     |
| 7     | Cross-condition analysis (Hungarian + temporal)     | 🔴 pending  | —     |
| 8     | Audio extension (deferred)                          | 🔴 pending  | —     |

**Total: 81/81 unit tests green.** Per-phase verification commands and
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

### Gated on a real DiT-B/2 trained on real ImageNet-256

- [ ] **(2.7)** Matched-compute (gFID-or-budget) stopping criterion.
- [ ] **(2.10)** SLURM driver `slurm/train_sit.slurm` for the full CINECA run.
- [ ] **(2.11)** Full DiT-B/2 run on SD-VAE matching paper gFID ±0.5.
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
