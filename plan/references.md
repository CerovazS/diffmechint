# Critical Files / References

External resources, canonical upstream repos, open verification items, and the explicit non-goals of this plan. See [README](README.md) for navigation.

## From the user's environment

- **Vision AE survey** (drop-in availability):
  `Research/Diffusability/Vision Autoencoder Open-Source Availability — Survey 2026-04.md`
- **Audio AE survey** (for Phase 8):
  `Research/Diffusability/Audio Autoencoder Open-Source Availability — Survey 2026-04.md`
- **Project proposal** (Notion):
  `https://www.notion.so/34daae54ea0c81448d77d4fcd52ac0da`
- **Stack rules:** `~/.claude/rules/swe-stack.md`,
  `~/.claude/rules/ml-training.md`,
  `~/.claude/rules/flywheel.md`
- **CINECA SLURM:** `~/.claude/rules/slurm.md`

## Canonical upstream repos

| Component | Repo | Pin |
|---|---|---|
| SiT backbone | https://github.com/willisma/SiT | latest main, vendor commit hash |
| REPA reference | https://github.com/sihyun-yu/REPA | for VAE-projection patterns |
| REPA-E (joint VAE) | https://github.com/End2End-Diffusion/REPA-E | for end-to-end loop |
| LightningDiT (VA-VAE) | https://github.com/hustvl/LightningDiT | for VAE abstraction patterns |
| EQ-VAE | https://github.com/zelaki/eqvae | adapter reference |
| DC-AE 1.0 | https://github.com/mit-han-lab/efficientvit | adapter reference |
| RAE | https://github.com/bytetriper/RAE | adapter reference |
| MAETok | https://github.com/Hhhhhhao/continuous_tokenizer | adapter reference |
| **SAELens (primary)** | https://github.com/jbloomAus/SAELens | runtime dep, `>=6.x`, pin transformer-lens with it |
| Dictionary learning (fallback) | https://github.com/saprmarks/dictionary_learning | vendor commit, only if SAELens DiT-blocks |
| sae_vis | https://github.com/callummcdougall/sae_vis | viz of SAELens-format SAEs |
| sae_dashboard | https://github.com/jbloomAus/SAEDashboard | Neuronpedia-style web viewer |
| Sparse feature circuits | https://github.com/saprmarks/feature-circuits | EAP + circuit reference |
| EAP | https://github.com/Aaquib111/edge-attribution-patching | EAP scaffold |
| nnsight | https://github.com/ndif-team/nnsight | runtime dep, Phase 6 EAP only |
| SAeUron | https://github.com/cywinski/SAeUron | timestep-aware SAE reference |
| Birth of Knowledge | https://arxiv.org/abs/2505.19440 | checkpoint-sweep methodology |
| Tracking Feature Dynamics | https://arxiv.org/abs/2412.17626 | warm-start methodology |

## Open verification items (not blocking, but worth checking before Phase 4)

- Does `revelio-diffusion/revelio` actually exist on GitHub? Subagent
  reported it; both my earlier surveys did not encounter the canonical
  Revelio code. If absent, reimplement Revelio's k-SAE protocol from the
  paper (it is a thin wrapper around standard TopK SAE training + label-σ
  monosemanticity scoring).
- TIDE (Huang et al. arXiv:2503.07050) — paper exists, code release
  status uncertain. Plan for re-implementation: TIDE = TopK SAE + a
  timestep-conditioned encoder (one MLP layer that takes `t` as input and
  shifts the encoder pre-activation). ~50 LoC delta on top of `topk.py`.

## What this plan deliberately does *not* do

- No DC-AE 1.5 condition (gated). When the upstream releases (track
  `dc-ai-projects/DC-Gen`), add a `dc_ae_1_5.yaml` config and rerun the
  pipeline.
- No HookedTransformer subclass for SiT — verified empirically too
  costly. We bypass TransformerLens via SAELens's lower-level
  `SAETrainer` API which takes pre-computed activations directly.
- No diffusers-based training loop — SiT is a small enough codebase that
  vendoring keeps full control.
- No mid-flight architecture sweep — all 4 conditions use the *same*
  SiT-B/L/XL backbone hyperparameters, by experimental design.
