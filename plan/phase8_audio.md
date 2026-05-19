# Phase 8 — Audio Extension (stretch, M6)

Mirror of the vision protocol on a small audio DiT — gated on upstream tokenizer availability and kept off the critical path. See [README](README.md) for navigation.

Mirror the protocol on a 60M-param audio DiT trained on MTG-Jamendo /
AudioSet-Music subset, with **{vanilla audio VAE, Semantic-VAE,
SALAD-VAE}**. Per
[[Audio Autoencoder Open-Source Availability — Survey 2026-04]]:
- SALAD-VAE is 🔴 (not released) — wait or substitute with X-Codec 1.0
  as the "semantic-anchored" condition (HuBERT pre-RVQ semantics).
- Semantic-VAE is 🟡 — only `dim=16, dim=64` ckpts public; sweep on
  bottleneck dim requires retraining via the public training script.
- Vanilla baseline: DAC 44.1 kHz (the only audio codec with full MIT
  training pipeline) or EnCodec 32 kHz.

**Recommend** moving the audio extension to a **separate branch / future
phase**; do not block the main vision pipeline on audio infrastructure.
Repo layout already supports a future `src/diffmechint/audio/` package.
