# Phase 6 — Sparse Feature Circuits via EAP

Edge Attribution Patching with SAE-feature nodes to extract per-concept circuits and validate them via SHIFT ablation. See [README](README.md) for navigation.

## Method choice

Edge Attribution Patching (Aaquib111/edge-attribution-patching) is the
canonical scalable method (Wang et al. + Conmy et al.) and outperforms
ACDC on AUC. Combine with **Sparse Feature Circuits**
(saprmarks/feature-circuits) for the SAE-feature-as-node circuit
formulation, per Marks et al. ICLR 2025.

## Pipeline

Per (condition, target concept):

1. Identify the target concept and a *contrast* concept (e.g. "dog" vs
   "cat").
2. Curate ~500 generation prompts/seeds split between target and contrast.
3. Replace MLP/attention nodes with their SAE-feature reconstructions
   (from Phase 4). Use SAEs trained on `(layer-50%, t=200, k=32)` as
   the canonical feature dictionary.
4. Compute edge attribution scores via EAP using `nnsight` traces (one
   forward + one backward per dataset).
5. Threshold to retain top-N edges; build the circuit graph.
6. Compute the **faithfulness / completeness / minimality** triplet
   (Wang et al. IOI):
   - Faithfulness: circuit alone reproduces target behavior
   - Completeness: removing circuit breaks behavior
   - Minimality: no proper subset suffices
7. SHIFT-style ablation validation: zero out circuit-internal SAE
   features; verify target concept disappears in generations (via CLIP
   classifier on samples).

## Compute

Per (concept, condition): ~30 A100-h. With 4 concepts × 4 conditions =
**480 A100-h**. Per the proposal's expected-results table.

## Output

`outputs/<run_id>/circuits/<vae>/<concept>/{circuit.json,
faithfulness.json, shift_eval.png}`.
