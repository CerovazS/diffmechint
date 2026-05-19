# Workflow Summary for the Implementing Agent

Ordered execution plan, PR discipline, and the current plan status. See [README](README.md) for navigation.

1. **Bootstrap** (Phase 0): `uv init`, vendor SiT, basic tests.
2. **Tokenizers** (Phase 1): write the seven adapters; precompute latents.
3. **SiT pipeline** (Phase 2): Lightning module + matched-compute trainer
   + fractional checkpoints. Smoke-run on SD-VAE 1k steps.
4. **Hooks** (Phase 3): residual-stream tap + activation buffer. Test on
   a checkpoint.
5. **SAE** (Phase 4): SAELens `SAETrainer` + custom HDF5 data provider
   + warm-start across DiT checkpoints. Train one
   `(layer-50%, t=0.20, k=32)` SAE per (condition, checkpoint) — 28
   SAEs, the minimum publishable cell. Save format directly compatible
   with `sae_vis`, `sae_dashboard`, Neuronpedia.
6. **Probes** (Phase 5): Revelio grid for the same cells.
7. **Circuits** (Phase 6): EAP for 4 target concepts on the final
   checkpoint of each condition.
8. **Analysis** (Phase 7): Hungarian overlap + temporal atlas figures.
9. **Audio** (Phase 8): branch off, do not block vision.

**Each phase is a separate PR.** Each PR ships with its tests, a Hydra
config, a SLURM driver, and a one-page `reports/<phase>.md` summary. The
agent should not skip ahead.

---

## Plan Status

- Tokenizer survey verified, K=5 substituted (DC-AE 1.5 → DC-AE 1.0)
- SiT vendor strategy decided (in-repo, not submodule)
- **SAE toolkit revised 2026-04-29**: SAELens (primary, raw
  `SAETrainer` API bypassing TransformerLens) + nnsight (Phase 6 EAP)
  + EAP-Aaquib (Phase 6) + dictionary_learning (vendored fallback)
- CINECA-first, NVIDIA-grant-second compute path defined
- Open verifications: Revelio repo existence, TIDE code release
- Per-phase acceptance gates and verification commands listed
