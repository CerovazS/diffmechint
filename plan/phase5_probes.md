# Phase 5 — Linear Probes (Revelio Grid)

Per-(layer × timestep × concept) linear probes built on cached activations. See [README](README.md) for navigation.

## Concept set

`src/diffmechint/probing/concepts.py`:

ImageNet-derived attribute axes:
- **Object**: 1000-way classification head (linear) on residual stream
- **Scene**: 365-way Places365 alignment via CLIP labels
- **Color**: 11-way (red, blue, green, ...) via WordNet attributes
- **Texture**: DTD 47-way
- **Shape**: ShapeNet-derived 12-way via ImageNet-Sketch alignment

For each concept axis, train a linear probe per (layer, timestep) cell.
Probes use scikit-learn's `LogisticRegression(max_iter=1000)` on
≤ 50k activation samples per cell. Activations cached under
`$FAST/diffmechint/probes/<run>/`.

## Output

Per condition, a 5 (concept) × 3 (depth) × 3 (timestep) accuracy heatmap.
Per concept, a single (depth, timestep) cell where the probe peaks. The
**probe-peak migration** across conditions is the target observable for
Claim 1 of the proposal.

## Implementation reference

Borrow patterns from Revelio (verify the canonical repo at
`https://github.com/revelio-diffusion/revelio` — if it does not exist,
fall back to the paper's algorithm description; the
`one-step-is-enough` repo by Surkov et al. or the
`Concept Steerers` repo by Kim/Ghadiyaram are likely better-maintained
substitutes for hook + probe code).
