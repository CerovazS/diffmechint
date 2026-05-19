# Phase 7 — Cross-Condition Analysis (the deliverable)

Hungarian-matched dictionary overlap, temporal atlas, and the three layers of expected findings. See [README](README.md) for navigation.

## Hungarian-matched dictionary overlap

`src/diffmechint/analysis/hungarian_match.py`:

Given two SAEs trained on different tokenizers at the same (layer, t, k)
cell, compute the cosine-similarity matrix between their feature decoders
and run the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
to produce a 1-to-1 feature pairing. Report:
- mean matched cosine
- distribution of matched cosines
- count of "high-match" features (cos > 0.7)

This is the cross-tokenizer feature-dictionary overlap measure central to
the paper's narrative.

## Temporal atlas

`src/diffmechint/analysis/temporal_atlas.py`:

For each condition, plot per-concept probe accuracy vs training-fraction
checkpoints. Detect:
- **phase transitions** (steep accuracy jumps)
- **swing-by bumps** (Yang et al. 2025 non-monotonic recovery)
- **temporal dips** (Birth of Knowledge, Sawmya et al. 2025)

Emit a single multi-panel figure per condition; overlay all four conditions
on one plot for the headline result.

## Acceptance for the paper

The codebase is "publication-ready" when:
1. The `time × layer × timestep × feature` trajectory CSV exists for all
   4 conditions.
2. The Hungarian-overlap matrix exists for all `C(4, 2) = 6` condition
   pairs.
3. All Level-3 circuits validated by SHIFT ablation (zero-out drops CLIP
   classification > 30%).

## Expected findings — three layers

The class-conditional setup is *not* limiting: it admits three layers of
findings, only the first of which depends on labels. Listed in order of
increasing novelty over Revelio (Kim et al. 2024 / arXiv 2411.16725).

**Layer 1 — Probe-based (uses labels).**
- *Class identity & WordNet super-categories*: which SAE features predict
  class / animal-vehicle-furniture / dog-vs-cat. Direct from ImageNet labels.
- *Temporal class emergence*: at which fractional ckpt does each condition
  start to discriminate dog vs cat; do `eq_vae_noz` features stabilize earlier
  than `sd_vae` features.
- *Revelio-style concept probes* via post-hoc CLIP zero-shot labels (texture,
  color, scene, composition, style) — unlocks the full Revelio §4.1-§4.4
  concept grid without re-training the DiT or adding a text encoder.

**Layer 2 — Cross-condition comparison (label-free).**
The *unique* value of our setup vs Revelio: K=3 fully controlled conditions
(same architecture, dataset, compute — only the tokenizer differs).
- *Universality test*: do the 3 tokenizers learn the same dictionary of
  features (Hungarian-matched, §11.1)? Confirms / refutes Anthropic-style
  universality in the diffusion-transformer regime.
- *Feature drift cross-VAE*: which features are unique to one condition.
  Especially: which features does `eq_vae_noz` learn that `sd_vae` does not
  (and vice-versa).
- *Z-score-destroyed-equivariance, mechanistic version*: compare
  `eq_vae` (z-scored, FID 276) vs `eq_vae_noz` (FID 25). Hypothesis: the
  z-scored variant develops more polysemantic, less class-pure features —
  i.e. the FID gap has a *mechanistic* counterpart, not just a quality one.
- *Compositional vs holistic*: dictionary structure analysis (clustering,
  intrinsic dimensionality) of features per condition. Does the
  joint-trained `repa_e` develop more compositional features than the
  standalone `sd_vae`?

**Layer 3 — Temporal dynamics (label-free).**
Enabled by the 7 fractional checkpoints per condition (Revelio analyses a
single final ckpt and cannot do this).
- *Feature emergence schedule*: at which training fraction features
  stabilize; cross-condition lag detection.
- *Coarse-to-fine layer order*: does layer 3 stabilize before layer 9, and
  is this order tokenizer-invariant?
- *Warm-start transfer efficiency*: how predictive are checkpoint-N features
  of checkpoint-(N+1) features — proxies for the speed of representational
  refinement across training.

**Layer 4 (stretch / future) — Generalization vs memorization (in-domain vs OOD).**
*Spunto*, non in scope per la prima submission. Probare SAE / linear probes
sia su attivazioni **in-domain** (ImageNet val, già held-out di Phase 3) sia
su attivazioni **out-of-domain** (es. encoding via SD-VAE di un dataset
diverso: COCO, Open Images, CIFAR, oppure immagini generate da modelli
text-to-image). Domanda: le feature SAE che emergono sono *generalizzano*
(stessa firing distribution su OOD) o sono *memorizzate* (collassano /
sparano su qualsiasi cosa o non sparano su nulla)? Connessione naturale a
PLAN §11.4 Layer 2 (universality cross-tokenizer estesa a cross-data).
Costo: re-encoding di un dataset OOD via VAE già usato (~1 h compute),
re-extraction attivazioni (~30 min), inference SAE/probe (~10 min).

**Candidate headline result** (to be refuted or confirmed):

> Different tokenizers do not just change FID — they cause the DiT to learn
> **fundamentally different feature dictionaries**. Tokenizers that produce
> lower FID (e.g. `eq_vae_noz`) yield more class-pure, less polysemantic SAE
> features. The mechanistic divergence emerges by ~50k training steps,
> correlates with the FID gap, and is *not* predictable from PSNR
> (reconstruction quality alone).

This is a finding Revelio cannot make: they have 4 uncontrolled models,
we have K=3 with controlled tokenizer-as-only-variable.
