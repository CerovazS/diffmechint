# CHECKLIST — diffmechint

Live, append-only-then-strikethrough tracker of every implementation step
in `PLAN.md`. Always update when starting / completing an item with a
one-line postmortem.

Format: `- [ ] (Phase X.Y) Subject — owner — ETA`. Strike completed items
with a `~~...~~` line plus a `# DONE: ...` postmortem.

## Phase 0 — Repo bootstrap

- [x] (0.1) Create `~/diffmechint/`, copy PLAN, `git init -b main` — claude — DONE
- [x] (0.2) Initial skeleton (`README.md`, `CLAUDE.md`, `CHECKLIST.md`, `.gitignore`) — claude — DONE
- [x] (0.3) Create private GitHub repo `CerovazS/diffmechint` — claude — DONE: github.com/CerovazS/diffmechint
- [x] (0.4) `pyproject.toml` per PLAN §3 + `uv sync` clean — claude — DONE: torch 2.6, lightning 2.4+, hydra 1.3, timm 1.0
- [x] (0.5) Vendor `willisma/SiT` at pinned commit into `src/diffmechint/sit/` — claude — DONE: commit cbde832
- [x] (0.6) Add no-op forward-hook slot in `SiTBlock` — claude — DONE: `block_idx` attr, hooks via PyTorch built-in
- [x] (0.7) Smoke test `tests/test_transport_fm_ot.py` (Linear interpolant) — claude — DONE: 4 tests, ICPlan + alpha=t/sigma=1-t verified
- [x] (0.8) Smoke test `tests/test_sit_forward.py` (B/2 forward + hooks) — claude — DONE: 4 tests, all 12 blocks fire hooks
- [x] (0.9) `uv run pytest tests/` green — claude — DONE: 8/8 passed in 14.58s
- [ ] (0.10) Initial commit + push

## Phase 1 — Tokenizer adapters + latent precompute

- [ ] (1.1) `tokenizers/base.py` — `TokenizerAdapter` ABC contract
- [ ] (1.2) `tokenizers/registry.py` — name → adapter factory
- [ ] (1.3) `tokenizers/sd_vae.py` — `stabilityai/sd-vae-ft-mse`
- [ ] (1.4) `tokenizers/eq_vae.py` — `zelaki/eq-vae-ema`
- [ ] (1.5) `tokenizers/repa_e.py` — `REPA-E/e2e-sdvae-hf`
- [ ] (1.6) `tokenizers/dc_ae_1_0.py` — `mit-han-lab/dc-ae-f32c32-in-1.0`
- [ ] (1.7) `tokenizers/rae.py` — `nyu-visionx/rae-dinov2-base-vitxl-n08`
- [ ] (1.8) `tokenizers/maetok.py` — optional 6th
- [ ] (1.9) `tokenizers/va_vae.py` — optional 7th
- [ ] (1.10) `TokenGridAdapter` for non-grid latents (RAE, MAETok)
- [ ] (1.11) `precompute_latents.py` CLI + Hydra config per adapter
- [ ] (1.12) `tests/test_tokenizer_adapters.py` round-trip PSNR > 25 dB
- [ ] (1.13) Acceptance run on 64 ImageNet-256 images per adapter

## Phase 2 — SiT training pipeline (FM-OT)

- [ ] (2.1) `training/sit_module.py` — `SiTLightningModule`
- [ ] (2.2) `conf/transport/fm_ot.yaml` (Linear + velocity)
- [ ] (2.3) Optional `t_sampler: laplace` extension for logSNR=0
- [ ] (2.4) AdamW + warm-up, EMA wiring
- [ ] (2.5) `training/data.py` — latent-cached ImageNet datamodule
- [ ] (2.6) `training/checkpointing.py` — fractional schedule
- [ ] (2.7) `training/matched_compute.py` — gFID-or-budget stopping
- [ ] (2.8) `tests/test_checkpoint_schedule.py`
- [ ] (2.9) 1k-step smoke run on SD-VAE
- [ ] (2.10) `slurm/train_sit.slurm` driver
- [ ] (2.11) Full DiT-B run on SD-VAE matches paper gFID ±0.5

## Phase 3 — Activation extraction

- [ ] (3.1) `hooks/activation_taps.py` — `ResidualStreamTap`
- [ ] (3.2) `hooks/timestep_router.py` — ContextVar + bin
- [ ] (3.3) `hooks/activation_buffer.py` — ring + shard-to-disk
- [ ] (3.4) `tests/test_hooks.py` — 36-record smoke

## Phase 4 — SAE training

- [ ] (4.1) Vendor `saprmarks/dictionary_learning` at pinned commit
- [ ] (4.2) `sae/topk.py` — TopK + BatchTopK
- [ ] (4.3) `sae/trainer.py` — warm-start across checkpoints
- [ ] (4.4) `sae/eval.py` — cosine, density, label-σ, RIEBench
- [ ] (4.5) `tests/test_sae_topk.py` — recon cosine > 0.85 on synthetic
- [ ] (4.6) Canonical-cell SAE sweep (28 SAEs)
- [ ] (4.7) Full-grid sweep (756 SAEs) — gated on results

## Phase 5 — Linear probes (Revelio grid)

- [ ] (5.1) `probing/concepts.py` — 5 attribute axes
- [ ] (5.2) `probing/revelio_grid.py` — per-cell linear probe
- [ ] (5.3) Probe-peak migration heatmap

## Phase 6 — Sparse feature circuits (EAP)

- [ ] (6.1) Vendor `Aaquib111/edge-attribution-patching` at pinned commit
- [ ] (6.2) `circuits/eap.py` — EAP via nnsight
- [ ] (6.3) `circuits/faithfulness.py` — Wang IOI triplet
- [ ] (6.4) `circuits/shift_ablation.py` — SHIFT validation
- [ ] (6.5) `circuits/riebench.py` — causal-edit score
- [ ] (6.6) 4 concepts × 4 conditions

## Phase 7 — Cross-condition analysis

- [ ] (7.1) `analysis/hungarian_match.py` — cross-tokenizer overlap
- [ ] (7.2) `analysis/temporal_atlas.py` — phase-transition detection
- [ ] (7.3) Headline figure (4 conditions overlaid)

## Phase 8 — Audio extension (deferred branch)

- [ ] (8.1) Branch `audio-ext`, package `src/diffmechint/audio/`
