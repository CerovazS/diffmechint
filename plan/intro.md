# Intro — Context, Repository Architecture, Stack

Context for the diffmechint codebase, the monorepo layout, and the pinned Python stack. See [README](README.md) for navigation.

## Context

The paper is the first systematic mechanistic-interpretability analysis of
latent-diffusion **tokenizer interventions**. The diffusability literature
has produced 22+ autoencoder modifications evaluated only at the task level
(gFID). The mech-int literature has matured to the point where SAE +
layer/timestep probing + EAP circuit discovery can answer the next-level
question: *which features the DiT learns, in what order, and whether the
coarse-to-fine schedule is tokenizer-invariant*.

The codebase must support a **K=4 (or K=5) controlled tokenizer sweep on a
matched-compute SiT backbone**, with **7 intermediate checkpoints per
condition**, plus a SAE/probing/EAP toolkit applied per
(condition × checkpoint × layer × timestep) cell.

While the NVIDIA grant is pending, work runs on **CINECA Leonardo (4× A100
64GB per node, $WORK / $FAST / $SCRATCH)**. The plan must therefore target
CINECA today and scale up to 8× H100 when the grant lands without rewriting
the training loop.

The original Notion proposal targets DC-AE 1.5 as the
"information-ordered bottleneck" condition, but
[[Vision Autoencoder Open-Source Availability — Survey 2026-04]] confirmed
DC-AE 1.5 is **🔴 not yet released** (DC-Gen repo gated on legal review).
Recommended replacements from the survey's 🟢 list:
- **DC-AE 1.0** (released, MIT, Apache-2.0) as a near-substitute for the
  information-ordered cluster
- **RAE** (NYU VisionX, MIT, full ckpts) as a discriminative-encoder alternative
- **MAETok** (CMU+HKU, MIT, ckpts on HF) as the redesign-cluster cleanest
- **VA-VAE** (hustvl, MIT, paired with LightningDiT) as the VFM-alignment
  reference

**Recommended K=5 today:** {SD-VAE, EQ-VAE, REPA-E, DC-AE 1.0, RAE} — covers
all four diffusability clusters, all 🟢 in the survey, all MIT/Apache. Add
MAETok as a 6th if compute permits. DC-AE 1.5 enters as a "future condition"
once released.

---

## Repository Architecture

Single monorepo, `diffusability-mechint/`. SiT vendored, not submoduled, so
it can be Hydra-ified and cleanly extended. SAE / probing / EAP live as
sibling packages that import from the SiT module. No git submodules — they
break reproducibility on CINECA.

```
diffusability-mechint/
├── pyproject.toml                   # uv-managed, single source of truth
├── uv.lock
├── README.md
├── CLAUDE.md                        # repo-local agent context
├── conf/                            # Hydra configs
│   ├── config.yaml                  # top-level entrypoint
│   ├── tokenizer/
│   │   ├── sd_vae.yaml
│   │   ├── eq_vae.yaml
│   │   ├── repa_e.yaml
│   │   ├── dc_ae_1_0.yaml
│   │   ├── rae.yaml
│   │   ├── maetok.yaml             # optional 6th
│   │   └── va_vae.yaml             # optional 7th
│   ├── model/
│   │   ├── sit_b_2.yaml            # ~130M
│   │   ├── sit_l_2.yaml            # ~450M
│   │   └── sit_xl_2.yaml           # ~670M
│   ├── transport/
│   │   └── fm_ot.yaml              # Linear interpolant + velocity
│   ├── trainer/
│   │   ├── cineca_4xa100.yaml
│   │   └── nvidia_8xh100.yaml      # for grant phase
│   ├── sae/
│   │   ├── topk_k16.yaml
│   │   ├── topk_k32.yaml
│   │   └── topk_k64.yaml
│   └── probe/
│       └── revelio_grid.yaml
├── src/diffmechint/
│   ├── __init__.py
│   ├── tokenizers/                  # one adapter per VAE family
│   │   ├── base.py                  # TokenizerAdapter ABC
│   │   ├── registry.py              # name → adapter factory
│   │   ├── sd_vae.py
│   │   ├── eq_vae.py
│   │   ├── repa_e.py
│   │   ├── dc_ae_1_0.py
│   │   ├── rae.py
│   │   ├── maetok.py
│   │   └── va_vae.py
│   ├── sit/                         # vendored from willisma/SiT, modified
│   │   ├── models.py                # SiT + hook-instrumented SiTBlock
│   │   ├── transport/               # unchanged from upstream (Linear/GVP/VP)
│   │   ├── train.py                 # upstream argparse loop, kept for reference
│   │   └── LICENSE                  # upstream MIT
│   ├── training/
│   │   ├── sit_module.py            # LightningModule wrapping SiT
│   │   ├── data.py                  # latent-cached ImageNet datamodule
│   │   ├── checkpointing.py         # fractional schedule {2%,5%,10%,25%,50%,75%,100%}
│   │   ├── precompute_latents.py    # CLI: encode ImageNet through each VAE → HDF5
│   │   └── matched_compute.py       # gFID-or-budget stopping criterion
│   ├── hooks/
│   │   ├── activation_taps.py       # nn.Module forward-hook utilities
│   │   ├── timestep_router.py       # bin t∈{25,200,500} for cell collection
│   │   └── activation_buffer.py     # streaming buffer with optional shard-to-disk
│   ├── sae/
│   │   ├── topk.py                  # TopK / BatchTopK SAE
│   │   ├── trainer.py               # warm-start across checkpoints (Xu et al. 2412.17626)
│   │   └── eval.py                  # reconstruction cosine, label-σ, monosemanticity
│   ├── probing/
│   │   ├── revelio_grid.py          # layer × timestep × concept linear probe
│   │   └── concepts.py              # ImageNet concept set (object/scene/color/texture/shape)
│   ├── circuits/
│   │   ├── eap.py                   # Edge Attribution Patching for DiT residual circuits
│   │   ├── faithfulness.py          # Wang et al. IOI triplet
│   │   ├── shift_ablation.py        # SHIFT-style causal-edit validation
│   │   └── riebench.py              # One-Step-is-Enough causal-edit score
│   ├── analysis/
│   │   ├── hungarian_match.py       # cross-tokenizer dictionary overlap
│   │   └── temporal_atlas.py        # time × layer × timestep × feature trajectories
│   └── utils/
│       ├── console.py               # rich-based ok/info/warn/error
│       ├── seed.py
│       ├── distributed.py
│       └── io.py                    # safetensors / HDF5 helpers
├── scripts/
│   ├── precompute_latents.sh        # SLURM driver
│   ├── train_sit.sh                 # SLURM driver per condition
│   ├── train_saes.sh                # SLURM SAE sweep (warm-start)
│   ├── run_probes.sh
│   └── run_eap.sh
├── slurm/
│   ├── precompute.slurm
│   ├── train_sit.slurm
│   └── sweep_sae.slurm
├── tests/
│   ├── test_tokenizer_adapters.py
│   ├── test_transport_fm_ot.py
│   ├── test_hooks.py
│   ├── test_sae_topk.py
│   ├── test_eap_smoke.py
│   └── test_checkpoint_schedule.py
└── outputs/                         # symlink → $FAST/diffmechint/outputs
    └── <run_id>/
        ├── artifacts/
        ├── checkpoints/
        ├── activations/
        ├── saes/
        ├── probes/
        ├── circuits/
        ├── plots/
        └── reports/
```

---

## Stack Pin and Environment

`pyproject.toml` is the single source of truth.

```toml
[project]
name = "diffmechint"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "torch>=2.5.0,<2.7",
    "torchvision",
    "diffusers>=0.30",
    "transformers>=4.45",
    "accelerate>=1.0",
    "lightning>=2.4",
    "hydra-core>=1.3",
    "omegaconf",
    "timm",
    "torchdiffeq",                 # ODE for FM-OT sampling
    "huggingface-hub",
    "safetensors",
    "h5py",
    "einops",
    "rich",                        # console
    "wandb",
    "scikit-learn",                # linear probes, Hungarian
    "scipy",
    "matplotlib",
    "nnsight>=0.3",                # activation patching for EAP
]

[project.optional-dependencies]
flash = [
    # FA3 cu128 abi3 wheel — see CLAUDE.md, do NOT pip install flash-attn
]
dev = ["pytest", "ruff", "mypy"]
```

**Hard rules from `~/.claude/rules/swe-stack.md`:**
- `uv add` only — never `uv pip install`
- Configs use Hydra `_target_`, instantiated via `hydra.utils.instantiate`
- Console output via the shared `rich` helpers (`ok / info / warn / error`)
- Checkpoints + datasets live under `$FAST` or `$SCRATCH`, never `$WORK`

**FlashAttention-3 install** (after `uv sync`):
```bash
pip install --no-cache-dir \
  "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
# Note: FA3 prebuilt wheel needs cu128 on H100; CINECA A100s use FA2 — see swe-stack.md
```

For CINECA A100, fall back to FA2:
```bash
module load cuda/12.2
uv add setuptools
uv add flash-attn --no-build-isolation
```
