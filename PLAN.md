# Implementation Plan — Semantic Geometry of Diffusability

A self-contained, agent-executable plan for building the codebase behind
**"The Semantic Geometry of Diffusability: A Mechanistic Atlas of How Latent
Geometry Shapes Diffusion Transformer Learning"** (Mancusi/Strano, Sapienza —
NVIDIA Academic Grant submission, ICLR/ICML 2027 target).

---

## 1. Context

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

## 2. Repository Architecture

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

## 3. Stack Pin and Environment

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

---

## 4. Phase 0 — Repo Bootstrap

Goals: working `uv sync`, vendored SiT, smoke tests for the SiT forward
pass and the FM-OT interpolant.

1. `uv init`, populate `pyproject.toml`, run `uv sync`.
2. Vendor `willisma/SiT` (commit-pinned) into `src/diffmechint/sit/`. Keep
   `LICENSE` (MIT) intact. Do **not** keep upstream `train.py` as the
   training entry — it stays for reference; the Lightning module is canonical.
3. Add minimal hook hooks to `models.py:SiTBlock`: a registered
   `forward_hook` slot keyed by block index that the activation tap module
   can attach to. Do not change forward semantics; the tap is a no-op when
   no listener is attached.
4. Smoke tests:
   - `tests/test_transport_fm_ot.py`: instantiate `create_transport(path_type='Linear', prediction='velocity', loss_weight=None)`. Assert it returns the ICPlan with `alpha_t = t, sigma_t = 1 - t`.
   - `tests/test_sit_forward.py`: SiT-B/2 forward on a `(2, 4, 32, 32)` tensor produces `(2, 4, 32, 32)`. Hooks fire 12 times.

**Acceptance:** `uv run pytest tests/` green; `uv run python -c "from diffmechint.sit.models import SiT_models; m = SiT_models['SiT-B/2']()"` works.

---

## 5. Phase 1 — Tokenizer Adapters

Every VAE/tokenizer becomes a `TokenizerAdapter` exposing the same
interface so the Lightning training module is tokenizer-agnostic.

### 5.1 The adapter contract

`src/diffmechint/tokenizers/base.py`:

```python
class TokenizerAdapter(abc.ABC):
    name: str
    latent_shape: tuple[int, int, int]   # (C, H, W) at 256² input
    scaling_factor: float
    license: str                         # for audit
    
    @abc.abstractmethod
    def encode(self, x: Tensor) -> Tensor: ...   # x: (B, 3, 256, 256)
    
    @abc.abstractmethod
    def decode(self, z: Tensor) -> Tensor: ...
    
    @property
    def in_channels(self) -> int:        # for SiT in_channels arg
        return self.latent_shape[0]
```

### 5.2 Adapter implementations

Each concrete adapter wraps the upstream loader. All weights pulled from
HuggingFace Hub at run time, cached under `$FAST/hf_cache/`.

| Adapter | Loader call | Latent shape (256²) | Scaling | Notes |
|---|---|---|---|---|
| `sd_vae` | `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")` | (4, 32, 32) | 0.18215 | baseline |
| `eq_vae` | `AutoencoderKL.from_pretrained("zelaki/eq-vae-ema")` | (4, 32, 32) | 0.18215 | drop-in `diffusers` |
| `repa_e` | `AutoencoderKL.from_pretrained("REPA-E/e2e-sdvae-hf")` | (4, 32, 32) | 0.18215 | jointly-trained VAE |
| `dc_ae_1_0` | from `mit-han-lab/dc-ae-f32c32-in-1.0` via custom loader | (32, 8, 8) | learned | non-diffusers, see efficientvit |
| `rae` | `nyu-visionx/rae-dinov2-base-vitxl-n08` | high-dim token grid | learned | flow-matching DiT pairing |
| `maetok` | `MAETok/maetok-b-128` | 128 tokens × D | learned | continuous tokenizer |
| `va_vae` | `hustvl/imagenet256-latents-vave-f16d32-dinov2` | (32, 16, 16) | learned | DINOv2-aligned |

### 5.3 Latent shape adaptation in SiT

SiT's `in_channels` is hard-coded by config. The Lightning module reads
`adapter.in_channels` and instantiates the SiT with that value. For
non-grid latents (RAE, MAETok), use a **canonical-token-grid view** — the
adapter exposes `(B, T, D)` and the Lightning module reshapes/projects to
SiT's `(B, C, H', W')` expectation, or swaps in a 1D-token SiT variant.
**Decision rule:** if `latent_shape` has ≤ 32 channels and a grid layout,
use the standard SiT 2D model; otherwise wrap with a `TokenGridAdapter`
(linear projection + spatial reshape) before SiT entry.

### 5.4 Latent precomputation

Encoding ImageNet through every VAE on every epoch is wasteful. Pre-encode
once to `$FAST`:

`src/diffmechint/training/precompute_latents.py`:
- Reads ImageNet-256 (mounted from `$FAST/datasets/imagenet256/`).
- For each registered tokenizer, dumps encoded latents to
  `$FAST/diffmechint/latents/{tokenizer_name}/{shard:05d}.h5` with
  `(z, label)` pairs in fp16.
- ~150 GB per VAE per ImageNet split (raw + cached).
- Single-A100 ~2 h per VAE per 1M images.

**Acceptance:** for each adapter, encode→decode round-trip on a held-out
batch produces images with PSNR > 25 dB; latent statistics
(mean/std/min/max) logged to `outputs/precompute/<vae>/stats.json`.

---

## 6. Phase 2 — SiT Training Pipeline (FM-OT)

### 6.1 LightningModule

`src/diffmechint/training/sit_module.py`:

```python
class SiTLightningModule(pl.LightningModule):
    def __init__(self, model_cfg, transport_cfg, tokenizer, optimizer_cfg, ema_decay=0.9999):
        super().__init__()
        self.model = hydra.utils.instantiate(model_cfg, in_channels=tokenizer.in_channels)
        self.transport = create_transport(**transport_cfg)
        self.ema = EMA(self.model, decay=ema_decay)
        # Tokenizer is frozen; latents arrive pre-encoded from datamodule
    
    def training_step(self, batch, batch_idx):
        z, labels = batch                          # latents, class labels
        loss_dict = self.transport.training_losses(self.model, z, dict(y=labels))
        return loss_dict["loss"].mean()
    
    def configure_optimizers(self): ...           # AdamW + warm-up
```

### 6.2 FM-OT interpolant

Hydra config `conf/transport/fm_ot.yaml`:

```yaml
_target_: diffmechint.sit.transport.create_transport
path_type: Linear         # alpha_t = t, sigma_t = 1 - t  → optimal-transport coupling
prediction: velocity       # most stable, used by REPA, LightningDiT
loss_weight: null          # uniform weighting
train_eps: 0
sample_eps: 0
```

The proposal mentions a *Laplace logSNR schedule concentrated around
logSNR=0*. SiT's transport layer does not implement a Laplace logSNR
sampling; default is uniform t ∈ [0, 1]. **Action:** add a
`t_sampler: uniform | laplace` field to the transport config, implement
Laplace sampling in `src/diffmechint/sit/transport/sampling.py`, and wire
it into `training_losses`. Default `uniform` keeps parity with SiT
upstream; switch to `laplace` for the proposal's stated schedule.

### 6.3 Optimizer + matched compute

- AdamW lr 1e-4, β=(0.9, 0.999), wd 0, warm-up 10k steps (matches SiT
  paper). Same across all conditions.
- **Stopping criterion** (`matched_compute.py`): trainer halts when
  *either* the compute budget is reached *or* gFID drops below the
  matched-target window. Reuse `clean-fid` package for FID computation.
  Per condition, log `(steps, wall-clock GPU-h, gFID)` to
  `outputs/<run_id>/training_curve.csv`.

### 6.4 Fractional checkpoint schedule

`src/diffmechint/training/checkpointing.py`:

The proposal requires checkpoints at training fractions
`{2%, 5%, 10%, 25%, 50%, 75%, 100%}`. With matched-compute stopping the
total step count `S` is known per run, so checkpoint steps are
`{0.02·S, 0.05·S, 0.10·S, 0.25·S, 0.50·S, 0.75·S, 1.00·S}`. Implement as
a Lightning `ModelCheckpoint` with a custom `filename` and
`every_n_train_steps=None` plus a manual `on_train_batch_end` hook that
triggers a save when `global_step` crosses the next fractional target.

Each checkpoint writes:
- `step_{step:08d}.safetensors` (model weights)
- `step_{step:08d}_ema.safetensors` (EMA weights — these are the analysis
  target, since SAE work historically uses EMA)
- `step_{step:08d}_metadata.json` (loss, gFID, lr, fractional position)

Stored under `$FAST/diffmechint/runs/<vae>/<seed>/checkpoints/`.

### 6.5 Distributed training on CINECA

- 4× A100 64GB per node, DDP via Lightning, batch 32/GPU = 128 global.
- `bf16-mixed` precision; FA2 attention (until grant unlocks H100+FA3).
- `torch.compile` enabled by default; fall back to eager if compile bugs.
- SLURM template lives at `slurm/train_sit.slurm` with placeholders for
  `--vae`, `--model`, `--seed`. Sets `http_proxy/https_proxy` to login01
  per CINECA rules.
- Per-condition wall-clock estimate (DiT-B, 256², matched compute):
  ~480 H100-h ≈ 850 A100-h ≈ 9 days on 4× A100.
- **Strategy on CINECA:** sequential conditions, not concurrent — fits the
  $FAST quota and the 24-h SLURM partition limit (use checkpoint resume
  across SLURM jobs).

### 6.6 Acceptance gate for Phase 2

- 1k-step smoke run on SD-VAE produces decreasing loss, valid samples.
- Full DiT-B run on SD-VAE matches the SiT paper's gFID at 400k steps
  within 0.5 points.
- Checkpoint files at the 7 fractional steps land in `$FAST` with the
  correct schema.

---

## 7. Phase 3 — Activation Extraction

### 7.1 Hook utilities

`src/diffmechint/hooks/activation_taps.py`:

```python
class ResidualStreamTap:
    """Attaches forward hooks to a chosen subset of SiTBlocks and
    streams (B, T, D) activations into an ActivationBuffer."""
    
    def __init__(self, model, block_indices: list[int], buffer: ActivationBuffer):
        self.handles = [
            model.blocks[i].register_forward_hook(self._capture(i))
            for i in block_indices
        ]
    
    def _capture(self, idx):
        def hook(module, input, output):
            self.buffer.write(layer=idx, t=current_t(), x=output.detach())
        return hook
    
    def detach(self): for h in self.handles: h.remove()
```

The `current_t()` helper reads the timestep from a `ContextVar` that the
training/inference loop sets before each model call — avoids threading a
`t` argument through every hook.

### 7.2 Timestep routing

`src/diffmechint/hooks/timestep_router.py`:

The Revelio grid uses three discrete timesteps `t ∈ {25, 200, 500}` (out
of 1000). The router bins continuous `t ∈ [0, 1]` into these targets when
sampling activations for analysis. For SAE training, sample uniformly in
`t ∈ [0, 1]` *or* stratify into the three bins — both supported via a
`stratify: uniform | revelio` config flag.

### 7.3 Activation buffer

`src/diffmechint/hooks/activation_buffer.py`:

In-memory ring buffer plus optional shard-to-disk. Default capacity 1M
tokens × `D_max`. When full, flushes to
`$FAST/diffmechint/activations/<run_id>/<ckpt>/<layer>_<t_bin>.h5`.

### 7.4 Where to tap

For SiT-B (12 blocks), tap at depths `{25%, 50%, 75%}` ⇒ blocks
`{3, 6, 9}` for SAE. For SiT-L (24) ⇒ `{6, 12, 18}`. For SiT-XL (28) ⇒
`{7, 14, 21}`. Reads the Revelio convention. Configurable via
`probe.layers: [3, 6, 9]`.

### 7.5 Acceptance

Smoke: run inference with hooks attached on a 4-image batch; verify
`activation_buffer` contains `3 layers × 3 timesteps × 4 batch = 36`
records of shape `(T, D)` with the expected `D = 768/1024/1152` for
B/L/XL.

---

## 8. Phase 4 — SAE Training (Multi-Checkpoint Sweep)

### 8.1 Toolkit choice (revised 2026-04-29)

The first research pass concluded "avoid SAELens — coupled to
TransformerLens". A direct re-read of SAELens 6.x (3 verification
subagents, recorded in commit history) reverses that conclusion:

> **`SAETrainer` (`sae_lens/training/sae_trainer.py:67`) is genuinely
> model-agnostic.** It accepts `data_provider: Iterator[Tensor]` and
> never instantiates a model. The TransformerLens coupling lives only
> inside `LanguageModelSAETrainingRunner` and `CacheActivationsRunner`,
> which we bypass.

Concrete implications:
- We feed pre-computed activations from our HDF5 shards directly into
  `SAETrainer.fit()`. ~50 LoC adapter.
- TransformerLens is pulled as a transitive dependency but never
  executed on the DiT path. ~200 MB of mostly dead weight in the env.
- We get TopK / BatchTopK / Matryoshka / Gated / JumpReLU / Standard
  SAE variants by config, plus built-in multi-checkpoint
  (`n_checkpoints` arg) — exactly the warm-start primitive we wanted to
  build manually for Xu et al. 2412.17626.
- Save format is the standard SAELens `safetensors + JSON` pair, so
  trained SAEs load directly into `sae_vis`, `sae_dashboard`, and
  Neuronpedia for paper-grade visualization.

**Decision: SAELens is the primary library for Phase 4.** It is the
de-facto standard, the integration cost is ~50 LoC, and the ecosystem
benefits (visualization, format portability, active maintenance) are
real and documented.

`dictionary_learning` (saprmarks) stays as a vendored fallback under
`third_party/dictionary_learning/`. We pin a commit and import only the
TopK / BatchTopK trainer files in case SAELens introduces a DiT-blocking
regression. nnsight remains pinned as a runtime dep for Phase 6 EAP
circuit work — we do not use it for SAE training.

### 8.2 SAE architecture — via SAELens

`src/diffmechint/sae/topk.py` is a thin wrapper around
`sae_lens.TopKTrainingSAE` and `sae_lens.BatchTopKTrainingSAE`. We do not
re-implement the SAE; we configure it.

```python
from sae_lens import (
    SAETrainer,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.config import SAETrainerConfig

def build_sae(d_in: int, d_sae: int, k: int, variant: str = "batch_topk"):
    if variant == "topk":
        return TopKTrainingSAE(TopKTrainingSAEConfig(
            d_in=d_in, d_sae=d_sae, k=k,
            normalize_activations="expected_average_only_in",
        ))
    elif variant == "batch_topk":
        return BatchTopKTrainingSAE(BatchTopKTrainingSAEConfig(
            d_in=d_in, d_sae=d_sae, k=k,
            normalize_activations="expected_average_only_in",
        ))
    raise ValueError(variant)
```

Default `variant="batch_topk"` (better dead-feature rate per Bussmann
et al.); `topk` for parity with the canonical Anthropic recipe.

### 8.3 HDF5 → SAELens data provider

`src/diffmechint/sae/data_provider.py` — adapter from our Phase 3
activation buffer shards to a `data_provider: Iterator[Tensor]`:

```python
def hdf5_provider(
    shard_paths: list[Path],
    batch_size: int = 4096,
    device: str = "cuda",
    flatten_tokens: bool = True,
) -> Iterator[torch.Tensor]:
    """Yield (batch_size, D) tensors from Phase 3 HDF5 cells.

    flatten_tokens=True collapses (N, T, D) → (N*T, D); each spatial
    token contributes one SAE training sample. Set False to keep token
    structure when training a token-aware SAE variant.
    """
    for path in shard_paths:
        with h5py.File(path, "r") as f:
            arr = torch.from_numpy(f["activations"][()]).float()  # (N, T, D) fp16→fp32
            if flatten_tokens:
                arr = arr.reshape(-1, arr.shape[-1])              # (N*T, D)
            for batch in arr.split(batch_size):
                yield batch.to(device, non_blocking=True)
```

### 8.4 Warm-start across checkpoints (Xu et al. 2412.17626)

Implemented via SAELens's `n_checkpoints` config plus our own
post-processing wrapper that re-uses the previous checkpoint's
encoder+decoder weights when training the SAE for the next DiT
checkpoint:

```python
def train_with_warm_start(
    dit_ckpt_paths: list[Path],   # 7 fractional ckpts per (condition)
    activation_shards_per_dit: dict[Path, list[Path]],
    sae_kwargs: dict,
    trainer_kwargs: dict,
    out_root: Path,
) -> None:
    prev_sae_path: Path | None = None
    for dit_ckpt in dit_ckpt_paths:
        sae = build_sae(**sae_kwargs)
        if prev_sae_path is not None:
            sae.load_state_dict(load_safetensors(prev_sae_path), strict=False)
        provider = hdf5_provider(activation_shards_per_dit[dit_ckpt])
        trainer = SAETrainer(SAETrainerConfig(**trainer_kwargs), sae, provider)
        trainer.fit()
        prev_sae_path = trainer.checkpoint_path / "final.safetensors"
```

After the first checkpoint is trained from scratch (~10 GPU-h), each
subsequent checkpoint warm-starts and converges in ~3 GPU-h. ~70 % wall-
clock saving across the 7-checkpoint trajectory.

### 8.5 Sweep dimensions

Per the proposal: per condition, train SAE at `(layer, timestep, k)`
cells where `layer ∈ {25 %, 50 %, 75 % depth}`,
`timestep ∈ {0.025, 0.20, 0.50}`, `k ∈ {16, 32, 64}`,
`d_sae = 16384`. That is 27 SAEs per (condition, checkpoint). With
4 conditions × 7 checkpoints × 27 cells = **756 SAEs total**. Warm-start
brings the 7-checkpoint cost down to ~3× a single-checkpoint cost.

### 8.6 SAE compute budget

Single SAE: ~10 GPU-h on a single A100, dominated by activation
streaming from disk. Total ~7560 A100-h, but parallelize embarrassingly
across 4 GPUs ⇒ ~320 GPU-h wall-clock for the full sweep on CINECA.
**Plan:** run the canonical cell `(k=32, layer-50 %, t=0.20)` first
across all `(condition, checkpoint)` pairs (28 SAEs); expand to the
full 27-cell grid only when results justify.

### 8.7 SAE-side metrics

Per `src/diffmechint/sae/eval.py`:
- Reconstruction cosine and L2 (SAELens emits these natively at every
  checkpoint).
- Feature density (fraction non-zero per token).
- Label-σ per feature: per Revelio, std of class-label distribution
  across the top-k activating samples — proxy for monosemanticity.
- Live / dead feature count.
- Per-feature **RIEBench causal-edit score** (One-Step-is-Enough): zero
  out feature, run a small generation, measure CLIP-similarity drop on
  the targeted concept.

Each SAE's eval JSON lands at
`outputs/<run_id>/saes/<vae>/<ckpt>/<layer>_<t>_<k>/metrics.json`.

### 8.8 Visualization & format portability

SAELens-trained SAEs save as `safetensors + cfg.json`. They drop
straight into:
- **`sae_vis`** — per-feature dashboards, used by Anthropic Circuits
  Updates.
- **`sae_dashboard`** — Neuronpedia-style web viewer.
- **Hugging Face Hub** — public release alongside the paper, per the
  expected-results §6 of the proposal.

This is the principal ergonomic argument for SAELens over a from-
scratch trainer: zero-friction handoff to community visualization
tooling.

### 8.9 Phase 4 acceptance gate

- 1 SAE trained end-to-end on synthetic activations (smoke) — recon
  cosine > 0.85 within 1 k steps.
- 1 SAE trained on a real DiT-B/2 SD-VAE checkpoint at canonical cell —
  recon cosine > 0.85, density 1–5 %, dead-feature count < 5 %.
- 28-SAE warm-started sweep (4 conditions × 7 ckpts at canonical cell)
  completes in < 320 A100-h on CINECA.

---

## 9. Phase 5 — Linear Probes (Revelio Grid)

### 9.1 Concept set

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

### 9.2 Output

Per condition, a 5 (concept) × 3 (depth) × 3 (timestep) accuracy heatmap.
Per concept, a single (depth, timestep) cell where the probe peaks. The
**probe-peak migration** across conditions is the target observable for
Claim 1 of the proposal.

### 9.3 Implementation reference

Borrow patterns from Revelio (verify the canonical repo at
`https://github.com/revelio-diffusion/revelio` — if it does not exist,
fall back to the paper's algorithm description; the
`one-step-is-enough` repo by Surkov et al. or the
`Concept Steerers` repo by Kim/Ghadiyaram are likely better-maintained
substitutes for hook + probe code).

---

## 10. Phase 6 — Sparse Feature Circuits via EAP

### 10.1 Method choice

Edge Attribution Patching (Aaquib111/edge-attribution-patching) is the
canonical scalable method (Wang et al. + Conmy et al.) and outperforms
ACDC on AUC. Combine with **Sparse Feature Circuits**
(saprmarks/feature-circuits) for the SAE-feature-as-node circuit
formulation, per Marks et al. ICLR 2025.

### 10.2 Pipeline

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

### 10.3 Compute

Per (concept, condition): ~30 A100-h. With 4 concepts × 4 conditions =
**480 A100-h**. Per the proposal's expected-results table.

### 10.4 Output

`outputs/<run_id>/circuits/<vae>/<concept>/{circuit.json,
faithfulness.json, shift_eval.png}`.

---

## 11. Phase 7 — Cross-Condition Analysis (the deliverable)

### 11.1 Hungarian-matched dictionary overlap

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

### 11.2 Temporal atlas

`src/diffmechint/analysis/temporal_atlas.py`:

For each condition, plot per-concept probe accuracy vs training-fraction
checkpoints. Detect:
- **phase transitions** (steep accuracy jumps)
- **swing-by bumps** (Yang et al. 2025 non-monotonic recovery)
- **temporal dips** (Birth of Knowledge, Sawmya et al. 2025)

Emit a single multi-panel figure per condition; overlay all four conditions
on one plot for the headline result.

### 11.3 Acceptance for the paper

The codebase is "publication-ready" when:
1. The `time × layer × timestep × feature` trajectory CSV exists for all
   4 conditions.
2. The Hungarian-overlap matrix exists for all `C(4, 2) = 6` condition
   pairs.
3. All Level-3 circuits validated by SHIFT ablation (zero-out drops CLIP
   classification > 30%).

---

## 12. Phase 8 — Audio Extension (stretch, M6)

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

---

## 13. Engineering Practices and Robustness

These are the rules that keep the code maintainable past the publication
window.

### 13.1 Configuration

- Hydra `_target_` everywhere — no manual `if vae == "sd_vae"` branches
  in business logic.
- Per-run config snapshot saved alongside checkpoints
  (`outputs/<run>/config.yaml`).
- Deterministic seeds: `seed_everything(42)` at run start; per-condition
  seed overrides via Hydra multirun.

### 13.2 Reproducibility

- Every run writes a `reproducibility.md` to its output dir, per
  `~/.claude/rules/flywheel.md`. Lists git SHA, branch, exact CLI
  command, env (`uv pip freeze`), hardware, seeds, and SLURM job IDs.
- A `commit.txt` is the last artifact written.
- Outputs/checkpoints never overwrite a previous run's directory; run
  IDs are `{vae}_{seed}_{timestamp}` and are unique by construction.

### 13.3 Logging

- Rich console for human; CSV/JSON for machines; WandB for dashboards.
- All three modes always-on; CSV is the source of truth for paper plots.

### 13.4 Testing

- `pytest` unit tests for every adapter (encode→decode round-trip, shape
  invariants, scaling factor application).
- A 1k-step smoke run on SD-VAE in CI proves the pipeline is alive.
- An `eap_smoke` test validates EAP returns non-empty edges on a tiny
  synthetic circuit.

### 13.5 Code style

- Ruff for linting + formatting.
- mypy strict on `src/diffmechint/tokenizers/` and
  `src/diffmechint/sae/` (the contract-heavy layers); permissive
  elsewhere.
- One-line module docstring per file; no decorative comments; no
  multi-paragraph docstrings.

### 13.6 License compliance

The repo bundles MIT/Apache code. Each tokenizer adapter logs the upstream
license to its config. **VFM-VAE** weights are NVIDIA Non-Commercial —
flag `commercial_use=False` in its adapter so an automated audit can
filter out non-commercial conditions.

### 13.7 Long-term maintenance

- Pin every dependency to a major + minor version, leave patch open.
- Pin upstream commit SHAs for vendored code (SiT, optional
  dictionary_learning fallback).
- Pin SAELens + transformer-lens versions in `pyproject.toml` (the SAE
  toolkit pair is the most version-sensitive part of the stack).
- Re-run `tests/` before any `uv lock --upgrade`.
- Keep `CLAUDE.md` in repo root with project-specific conventions for
  future agents.

---

## 14. Verification / Test Plan

End-to-end checks, in order, that an implementing agent should run.

| Stage | Command | Acceptance |
|---|---|---|
| Env | `uv sync && uv run python -c "import diffmechint"` | imports cleanly |
| Tests | `uv run pytest tests/` | all green |
| Adapter round-trip | `uv run python -m diffmechint.training.precompute_latents tokenizer=sd_vae max_images=64` | PSNR > 25 dB |
| All adapters | repeat for each tokenizer | all PSNR > 22 dB |
| FM-OT smoke | `uv run python -m diffmechint.training.train_sit tokenizer=sd_vae model=sit_b_2 trainer.max_steps=1000` | loss decreases, no NaN |
| Hooks smoke | `uv run pytest tests/test_hooks.py -v` | activation buffer contains expected shapes |
| Checkpoint schedule | `uv run python -m diffmechint.training.train_sit ... trainer.max_steps=10000 +ckpt_fractions=[0.1,0.5,1.0]` | 3 ckpt files written at expected steps |
| SAE smoke | `uv run python -m diffmechint.sae.train sae=topk_k32 layer=6 t_bin=200 ckpt=<path>` | recon cosine > 0.85, density 1-5% |
| Probe smoke | `uv run python -m diffmechint.probing.run_probes ckpt=<path>` | per-cell accuracy emitted |
| EAP smoke | `uv run python -m diffmechint.circuits.eap concept=dog ckpt=<path>` | non-empty edge list |
| Hungarian smoke | `uv run python -m diffmechint.analysis.hungarian_match sae_a=... sae_b=...` | matrix produced |

Full-pipeline sanity at the end of M3: 1 condition (SD-VAE), 1 checkpoint
(50% fraction), full Level-1 + Level-2 + Level-3 readout in < 24 h on
4× A100. If that works, scale.

---

## 15. Critical Files / References

### From the user's environment

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

### Canonical upstream repos

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

### Open verification items (not blocking, but worth checking before Phase 4)

- Does `revelio-diffusion/revelio` actually exist on GitHub? Subagent
  reported it; both my earlier surveys did not encounter the canonical
  Revelio code. If absent, reimplement Revelio's k-SAE protocol from the
  paper (it is a thin wrapper around standard TopK SAE training + label-σ
  monosemanticity scoring).
- TIDE (Huang et al. arXiv:2503.07050) — paper exists, code release
  status uncertain. Plan for re-implementation: TIDE = TopK SAE + a
  timestep-conditioned encoder (one MLP layer that takes `t` as input and
  shifts the encoder pre-activation). ~50 LoC delta on top of `topk.py`.

### What this plan deliberately does *not* do

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

---

## 16. Workflow Summary for the Implementing Agent

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

## 17. Plan Status

- Tokenizer survey verified, K=5 substituted (DC-AE 1.5 → DC-AE 1.0)
- SiT vendor strategy decided (in-repo, not submodule)
- **SAE toolkit revised 2026-04-29**: SAELens (primary, raw
  `SAETrainer` API bypassing TransformerLens) + nnsight (Phase 6 EAP)
  + EAP-Aaquib (Phase 6) + dictionary_learning (vendored fallback)
- CINECA-first, NVIDIA-grant-second compute path defined
- Open verifications: Revelio repo existence, TIDE code release
- Per-phase acceptance gates and verification commands listed
