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
- [x] (0.10) Initial commit + push — claude — DONE: pushed to origin/main

## Phase 1 — Tokenizer adapters + latent precompute

- [x] (1.1) `tokenizers/base.py` — `TokenizerAdapter` ABC contract — claude — DONE: dataclass `TokenizerSpec` + `nn.Module` ABC
- [x] (1.2) `tokenizers/registry.py` — name → adapter factory — claude — DONE: `@register("name")` decorator + `build(name)` dispatch
- [x] (1.3) `tokenizers/sd_vae.py` — `stabilityai/sd-vae-ft-mse` — claude — DONE: 84M params, frozen, round-trip shape-verified
- [x] (1.4) `tokenizers/eq_vae.py` — `zelaki/eq-vae-ema` — agent ac43d8 — DONE: drop-in `AutoencoderKL`
- [x] (1.5) `tokenizers/repa_e.py` — `REPA-E/e2e-sdvae-hf` — agent ac43d8 — DONE: jointly-trained VAE
- [x] (1.6) `tokenizers/dc_ae_1_0.py` — `mit-han-lab/dc-ae-f32c32-in-1.0-diffusers` — agent af4d9a — DONE: diffusers `AutoencoderDC`
- [x] (1.7) `tokenizers/rae.py` — `nyu-visionx/rae-dinov2-base-vitxl-n08-256` — agent af4d9a — SCAFFOLD: registry works; `load()` raises NotImplementedError until upstream decoder is vendored
- [ ] (1.8) `tokenizers/maetok.py` — optional 6th (defer until K=5 results)
- [ ] (1.9) `tokenizers/va_vae.py` — optional 7th (defer until K=5 results)
- [ ] (1.10) `TokenGridAdapter` for non-grid latents — defer until RAE `load()` lands
- [x] (1.11) `precompute_latents.py` CLI + Hydra config per adapter — claude — DONE: HDF5 sharding + ImageFolder loader + Hydra `_target_` instantiate verified
- [x] (1.12) Round-trip PSNR > 25 dB on real image — claude — DONE: GPU smoke (`scripts/smoke_adapters_gpu.py`) on 3090. sd_vae 28.10 / eq_vae 27.30 / repa_e 27.03 / dc_ae_1_0 26.20 dB on the canonical pytorch/hub `dog.jpg`.
- [ ] (1.13) Acceptance run on 64 ImageNet-256 images per adapter — pending real ImageNet mount (deferred to CINECA / pod)

## Phase 2 — SiT training pipeline (FM-OT)

- [x] (2.1) `training/sit_module.py` — `SiTLightningModule` — claude — DONE: SiT-B/2 130.5M params; FM-OT defaults + EMA via on_fit_start hook
- [x] (2.2) `conf/transport/fm_ot.yaml` — claude — DONE: Linear + velocity + null loss_weight
- [ ] (2.3) Optional `t_sampler: laplace` extension for logSNR=0 — deferred (uniform t works for synthetic smoke; Laplace lands when real ImageNet runs)
- [x] (2.4) AdamW + warm-up + EMA — claude — DONE: `LambdaLR` linear warmup, `EMA` wrapper with shadow on the right device
- [x] (2.5) `training/data.py` — claude — DONE: `SyntheticLatentDataModule` (smoke) + `CachedLatentDataModule` (HDF5 from precompute)
- [x] (2.6) `training/checkpointing.py` — fractional schedule — claude — DONE: 7 fractional ckpts at {2,5,10,25,50,75,100}% with safetensors live + EMA + JSON metadata
- [ ] (2.7) `training/matched_compute.py` — gFID-or-budget stopping — deferred (needs `clean-fid` + real samples; lands with full DiT-B run)
- [x] (2.8) `tests/test_checkpoint_schedule.py` — claude — DONE: 8 tests covering target rounding, dedup, dir creation
- [x] (2.9) 1k-step smoke run on SD-VAE — claude — DONE: synthetic-latent run, loss 1.99 → 1.55 monotonic, all 7 fractional ckpts saved (522 MB live + 522 MB EMA each)
- [ ] (2.10) `slurm/train_sit.slurm` driver — deferred until CINECA push
- [ ] (2.11) Full DiT-B run on SD-VAE matches paper gFID ±0.5 — deferred (needs real ImageNet-256 + matched-compute stopping)

## Phase 3 — Activation extraction

- [x] (3.1) `hooks/activation_taps.py` — `ResidualStreamTap` — claude — DONE: ctx-manager API, validates indices, drops on missing/out-of-bin t
- [x] (3.2) `hooks/timestep_router.py` — claude — DONE: `ContextVar` + `timestep_context` + `bin_revelio` (SiT bins {0.025, 0.20, 0.50} ± 0.05 tol, DDPM bins {25, 200, 500} also exposed)
- [x] (3.3) `hooks/activation_buffer.py` — claude — DONE: per-cell records, HDF5 shard `<layer>_<tbin>.h5` with fp16 lzf-compressed `(N, T, D)`, auto-flush on capacity
- [x] (3.4) `tests/test_hooks.py` — 36-record smoke — claude — DONE: PLAN §7.5 acceptance met (4 imgs × 3 layers × 3 timesteps → 36 records, shape `(256, 768)` for SiT-B/2)

## Phase 4 — SAE training (toolkit revised 2026-04-29 → SAELens primary)

- [x] (4.1) `uv add sae-lens` — claude — DONE: sae-lens 6.42.0, transformer-lens 3.0.0 transitive (never executed on DiT path)
- [x] (4.2) `sae/data_provider.py` — claude — DONE: `hdf5_provider` (per-shard drop_last, fp16→fp32 cast, optional flatten) + `synthetic_provider` (K-sparse mixtures for tests)
- [x] (4.3) `sae/builder.py` — claude — DONE: `build_sae(d_in, d_sae, k, variant)` factory for `topk` / `batch_topk` / `matryoshka` over SAELens 6.x classes
- [x] (4.4) `sae/trainer.py` — claude — DONE: `train_sae(...)` wrapping `SAETrainer` + `warm_start_from(prev.safetensors)` + `warm_started_sweep(...)` orchestrator
- [x] (4.5) `sae/eval.py` — claude — DONE: `evaluate_sae(...)` produces recon_cosine / recon_l2 / density / live/dead features. Label-σ + RIEBench deferred to (4.10)
- [x] (4.6) Hydra configs `conf/sae/{topk_k16, topk_k32, topk_k64, batch_topk_k32}.yaml` — claude — DONE
- [ ] (4.7) Vendor `third_party/dictionary_learning/...` as **fallback only** — deferred (only needed if SAELens DiT-blocks)
- [x] (4.8) `tests/test_sae_smoke.py` — claude — DONE: 9 tests including loss-decreases-by-25% on synthetic + Phase 3↔4 e2e integration on a SiT-B/2 forward
- [ ] (4.9) Canonical-cell SAE smoke on a real trained DiT-B/2 SD-VAE checkpoint — recon cosine > 0.85, density 1–5% — gated on Phase 2 full ImageNet run
- [ ] (4.10) Canonical-cell warm-started sweep (28 SAEs = 4 cond × 7 ckpts) — gated on Phase 2 full runs
- [ ] (4.11) Full 27-cell sweep (756 SAEs) — gated on (4.10) results
- [ ] (4.12) Confirm SAELens-saved SAEs load cleanly into `sae_vis` / `sae_dashboard` — quick check at end of (4.10)

## Phase 5 — Linear probes (Revelio grid)

- [x] (5.1) `probing/concepts.py` — 5 attribute axes — claude — DONE: ConceptAxis registry; `object` available via existing HDF5 labels; `scene/color/texture/shape` stubbed with explicit NotImplementedError + docstring TODO
- [x] (5.2) `probing/revelio_grid.py` — per-cell linear probe — claude — DONE: `train_probe` (sklearn LogisticRegression, ≤50k subsample, stratified split), `probe_one_cell`, `evaluate_grid` over (layers × t_bins), `GridResult.peak()` for migration analysis
- [x] (5.3) Probe-peak migration heatmap — claude — DONE: `GridResult.matrix(layers, t_bins)` returns `(L, T)` array (NaN for missing cells); `write_grid_result` produces JSON with peak cell per concept
- [x] (5.4) Buffer label propagation — claude — DONE: `ActivationBuffer.write(..., labels=Tensor | None)` backward-compat; HDF5 emits `labels` dataset alongside `activations` when present
- [x] (5.5) Hydra config `conf/probe/revelio_grid.yaml` — claude — DONE: concept axes, layers, t_bins, pool mode, probe hyperparameters
- [x] (5.6) `tests/test_probing.py` — claude — DONE: 18 tests covering registry, pool helpers, train_probe (linear-separable recovery > 0.9 acc), buffer label round-trip, probe_one_cell, evaluate_grid, e2e SiT-B/2 → labelled-buffer integration
- [ ] (5.7) Real-data acceptance: 5×3×3 accuracy heatmap on a real DiT-B/2 SD-VAE checkpoint — gated on Phase 2 full ImageNet run
- [ ] (5.8) Cross-condition probe-peak migration figure — gated on (5.7) across all 4 conditions

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
