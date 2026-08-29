# diffmechint

## The Semantic Geometry of Diffusability

**A controlled mechanistic-interpretability study of how an image tokenizer's
latent representation shapes feature learning in diffusion transformers.**

`diffmechint` trains matched **SiT** models on ImageNet-256: one model per latent
tokenizer, with the backbone scale, data, optimization budget, and evaluation
protocol matched across conditions. The tokenizer and its latent layout define
each experimental condition. Every trained model is then analyzed with sparse
autoencoders (SAEs), layer × diffusion-timestep probes, representation alignment,
and causal interventions.

The project asks a concrete question:

> Do diffusion transformers recover the same denoising abstractions in different
> coordinate systems, or does the tokenizer change the features and causal
> computations that the model learns?

## Research objectives

| Objective | Operational test | Intended conclusion |
|---|---|---|
| Map feature emergence | Compare SAE features and probe accuracy across training checkpoints, layers, and diffusion timesteps | Identify when and where semantic and spectral information becomes readable |
| Test representational invariance | Match dictionaries and compare residual streams with probe transfer, CKA/RSA, and activation similarity | Distinguish shared structure from tokenizer-specific organization |
| Test causal equivalence | Patch matched features and feature families across tokenizer-conditioned models, then measure activation and sample-level effects | Determine whether aligned representations perform the same computation |
| Validate the measurement layer | Measure held-out SAE reconstruction, feature usage, and FID under SAE substitution | Establish where SAE-based conclusions are faithful enough to support intervention claims |

## Claims and evidence status

> [!IMPORTANT]
> This is an active research repository. The design includes five tokenizer
> adapters, but the current comparative evidence is primarily based on three
> trained conditions: `sd_vae`, `repa_e`, and `eq_vae`. Most causal analyses use
> matched SiT-B/2 checkpoints at 200k steps. Results should not be generalized to
> every tokenizer family or diffusion architecture without further validation.

The current evidence supports the following bounded claims:

| Supported claim | Decisive evidence | Boundary |
|---|---|---|
| The selected Matryoshka SAEs are sufficiently faithful for the tested intervention grid | All 27 tested layer × timestep × tokenizer cells stayed below the defined ΔFID = 2 substitution gate (mean +0.59; maximum +1.80) | This rules out catastrophic SAE substitution error; it does not make every learned feature interpretable |
| Cross-tokenizer feature families do not show reliable source-specific transfer in the tested activation-space interventions | Across 110 directed family-patching tasks, mean coefficient R² was 0.012 and transfer exceeded shuffled pairing by only 3.69e-7 explained variance on average | This is a negative result for the tested K=3 models, cells, and matching procedure—not proof that universal diffusion features never exist |
| The current sparse dictionaries do not capture the full concept-relevant computation | In 36 concept-margin circuits, the top 100 SAE features retained about 0.020× of the clean margin gap without the reconstruction-error node and 0.273× with it | The error node is carrying material signal, so feature-only circuit claims remain incomplete |
| Matryoshka improves held-out reconstruction over plain TopK on the tested null-label distribution | Matryoshka won explained variance in all 27 matched cells (0.97779 vs 0.94737 mean EV), with a higher dead-feature rate (12.89% vs 7.55%) | This motivates the SAE choice; it is not a claim about generative quality or universal SAE superiority |

The broader thesis—that latent geometry determines the learning dynamics and
causal organization of a diffusion transformer—remains a hypothesis under test.
The current results establish measurement validity in the tested grid and expose
strong limits on cross-tokenizer transfer, but they do not yet isolate latent
geometry as the sole cause of those differences.

For the evolving run record and implementation state, see [`program.md`](program.md)
and [`CHECKLIST.md`](CHECKLIST.md). Curated experiment evidence is maintained in the
project's Flywheel research graph.

---

## Setup

Dependencies are managed with [**uv**](https://docs.astral.sh/uv/) — `pyproject.toml`
is the single source of truth.

```bash
uv sync --extra dev          # create .venv and install everything
source .venv/bin/activate    # or prefix commands with `uv run`
uv run pytest tests/         # sanity-check the install
uv run ruff check .          # lint
```

> [!NOTE]
> Add dependencies only with `uv add <pkg>` — never `uv pip install` (it breaks the
> `pyproject.toml` / `uv.lock` lockstep).

**HuggingFace token.** Tokenizer adapters download from the Hub. Export `HF_TOKEN`
before any download (on compute nodes it is not picked up from `~/.bashrc`
automatically):

```bash
export HF_TOKEN=hf_...
```

**Storage (CINECA Leonardo).** Code and lightweight outputs live under `$WORK`;
**latents, activations and checkpoints must go to `$FAST` or `$SCRATCH`** (`$WORK`
quota is too small). `outputs/` is a symlink into `$FAST`/`$SCRATCH`; never commit
weights or large artifacts.

---

## How the code is organized

The repo separates **library code**, **thin entry-point scripts**, **configs**, and
**cluster job templates**:

```
src/diffmechint/   ← all the real logic (importable package)
scripts/           ← thin CLI shells that import from src/diffmechint and expose argparse
conf/              ← Hydra configs (composed for SiT training)
slurm/             ← CINECA SLURM batch templates (one per stage)
tests/             ← pytest suite
flywheel/          ← curated per-node evidence uploaded to the research graph
index.md           ← local mirror of the Flywheel graph (one entry per node)
```

Most `scripts/analysis/*.py` and `scripts/eval/*.py` files are **6-line wrappers**:
the implementation lives in the package, e.g. `scripts/analysis/feature_dashboard.py`
→ `src/diffmechint/analysis/dashboard.py`. Run a script with `--help` to see its
exact flags.

```
src/diffmechint/
├── tokenizers/     adapters + registry (sd_vae, eq_vae, repa_e, dc_ae_1_0, rae)
├── sit/            vendored SiT model + transport (FM-OT) + sampling + train entry-point
├── training/       LightningModule, datamodule, latent precompute, checkpointing, callbacks
├── hooks/          residual-stream taps, timestep router, activation buffer
├── sae/            SAELens-backed SAE builder / trainer / eval / checkpoint loaders
├── probing/        concept registry + Revelio-grid linear probes
├── circuits/       EAP concept-margin attribution
├── analysis/       alignment, atlas, dashboard, latent_atlas, latent_probe,
│                   patching/ (bank·match·activation·families·sampling), spectral_probe, ...
├── spectral.py     DCT / octave-band utilities for the spectral analyses
└── utils/          rich console helpers, IO, plotting palette
```

---

## Running experiments

All stages follow the same pattern: a **library script** you can run locally for a
smoke / small slice, plus a **matching `slurm/<stage>.slurm`** template for the full
run on Leonardo. Use `--help` on any script for the full flag set.

### 0. Smoke-test the tokenizer adapters (real HF download, ~2 min on 1 GPU)

```bash
uv run python scripts/util/smoke_adapters_gpu.py
```

### 1. Precompute latents (image → VAE → HDF5 + per-feature `stats.json`)

```bash
# local, one condition
uv run python -m diffmechint.training.precompute_latents tokenizer=sd_vae
# full sweep on CINECA
sbatch slurm/precompute_all.slurm        # train split
sbatch slurm/precompute_val_all.slurm    # val split
```

### 2. Train the SiT backbone (Hydra entry-point, FM-OT)

```bash
# 1k-step smoke with synthetic latents on a single GPU
uv run python -m diffmechint.training.train \
    model=sit_b_2 transport=fm_ot \
    trainer.max_steps=1000 +ckpt_dir=outputs/smoke

# full run on CINECA
sbatch slurm/train_sit_full.slurm
```

Configs compose from `conf/config.yaml` and the subgroups `conf/{model,transport,
trainer,tokenizer,callbacks}/`. Override anything on the command line
(`key=value`, `+key=value` to add). Objects are built with Hydra `_target_` /
`instantiate` — there are no `if tokenizer == ...` branches.

### 3. Extract residual-stream activations (hooks + buffer)

```bash
uv run python scripts/extraction/extract_activations.py --help
sbatch slurm/extract_activations.slurm <condition> <step_NNNNNN>
```

`--y_null` re-extracts counterfactual activations with the class label replaced by
the null token (for class-conditional ablations).

### 4. Train SAEs over the (condition × layer × timestep × checkpoint) grid

```bash
# image-latent DiT activations (TopK / BatchTopK / Matryoshka)
uv run python scripts/training/train_sae.py \
    --conditions sd_vae repa_e eq_vae --variant matryoshka
sbatch slurm/train_sae.slurm

# SAEs directly on tokenizer latent tokens
sbatch slurm/train_latent_sae.slurm
```

### 5. Evaluate SAEs

```bash
# held-out reconstruction EV / dead-feature % per cell
uv run python scripts/eval/eval_sae_on_val.py --help
sbatch slurm/eval_sae_on_val.slurm

# causal-faithfulness: drop the SAE into the sampler, measure FID shift
sbatch slurm/sae_substitution_fid.slurm
```

### 6. Analysis

| What | Script (→ package module) | SLURM |
|---|---|---|
| Feature dashboard + monosemanticity atlas | `scripts/analysis/feature_dashboard.py` → `analysis/dashboard.py`; `scripts/analysis/atlas/*` → `analysis/atlas.py` | `feature_dashboard.slurm` |
| Linear probes (Revelio grid) | `scripts/analysis/run_revelio_grid.py` → `probing/` | `revelio_grid.slurm` |
| Cross-tokenizer alignment (probe-transfer / CKA-RSA / activation-proxy) | `scripts/analysis/tokenizer_dictionary_validation.py` → `analysis/alignment.py` (subcommands) | `tokenizer_dictionary_validation.slurm` |
| Feature / family activation patching | `scripts/analysis/sae_feature_patching.py` → `analysis/patching/*` (subcommands) | `feature_activation_patching.slurm` |
| Concept-margin EAP circuits | `scripts/analysis/sae_concept_eap.py` → `circuits/eap.py` | `concept_eap_array.slurm` |
| Latent-token atlas / probe | `scripts/analysis/latent_feature_atlas.py`, `latent_probe.py` → `analysis/latent_*` | `latent_probe.slurm` |
| Spectral (PSD / band alignment / inheritance / probe) | `scripts/analysis/{latent_psd,band_alignment,latent_dit_inheritance,spectral_probe}.py` → `analysis/*` + `spectral.py` | `band_alignment.slurm`, `band_inheritance.slurm`, `spectral_probe.slurm` |

The subcommand-style scripts expose their stages explicitly, e.g.:

```bash
uv run python scripts/analysis/tokenizer_dictionary_validation.py probe-transfer --help
uv run python scripts/analysis/sae_feature_patching.py build-bank --help
```

### Plotting

`scripts/plotting/` turns run outputs into figures + CSVs (`plot_run.py`,
`plot_fid_compare.py`, the per-experiment `plot_*.py`). All plots use the project's
Palette-B on a white background.

---

## Running on CINECA Leonardo

**Validate interactively before submitting batch jobs.** Grab a GPU session, test,
then `sbatch` the matching template:

```bash
srun -p boost_usr_prod -A <ACCOUNT> --gres=gpu:1 --mem=40G --time=2:00:00 --pty bash
module load cuda/12.2
source .venv/bin/activate
# compute nodes have no internet — set the squid proxy if you need downloads:
export http_proxy='http://login01:<port>'; export https_proxy="$http_proxy"
```

The `slurm/*.slurm` templates read their parameters from environment variables /
positional args (see the header of each file); scheduler `.out`/`.err` logs are
written under `slurm/logs/` (gitignored).

---

## Run-output conventions

Every run writes to its own uniquely-named directory under `outputs/` and never
overwrites a previous run. A training `<run_id>/` uses:

```
outputs/<pipeline>/<run_id>/
├── checkpoints/   step_*.safetensors  +  step_*_ema.safetensors (analysis target)  +  *_metadata.json
├── samples/       step_*_cfg{1p0,4p0}.png
├── metrics/       train/ · validation/ · summary.json   (extract_metrics.py)
├── plots/         loss / fid / summary PNGs              (plot_run.py)
└── reports/       reproducibility.md + commit.txt + summary.md
```

If you add a training script, conform to this layout so `extract_metrics.py` and
`plot_run.py` work unchanged.

**Latent normalization.** `stats.json` (written next to each latent set) is the
single source of truth for de/normalization and DiT setup — no consumer should
hardcode latent layout. Conditions have very different per-channel σ, so runtime
z-scoring (`CachedLatentDataModule(normalize=True)`, default) is required for
matched-compute comparisons.

```
ENCODE  image → vae.encode → mean·scaling_factor → z (HDF5, fp16)
LOAD    z → (z − μ)/σ                                   ← stats.json
SAMPLE  z̃ → z̃·σ + μ → ÷scaling_factor → vae.decode → image
```

---

## Experimental conditions

The full design spans five tokenizer conditions. The first three form the current
matched-comparison set; the remaining two extend the design but are not part of the
headline claims above.

| condition | experimental role | hf repo / source | current evidence status |
|---|---|---|---|
| `sd_vae` | conventional latent-diffusion baseline | `stabilityai/sd-vae-ft-mse` | production comparison |
| `eq_vae` | spectrally regularized / equivariant latent space | `zelaki/eq-vae-ema` | production comparison |
| `repa_e` | semantically aligned, jointly trained VAE | `REPA-E/e2e-sdvae-hf` | production comparison |
| `dc_ae_1_0` | high-compression, information-ordered bottleneck | `mit-han-lab/dc-ae-f32c32-in-1.0-diffusers` | adapter and latent precompute validated; excluded from current headline claims |
| `rae` | planned discriminative DINOv2 representation with learned decoder | `nyu-visionx/rae-dinov2-base-vitxl-n08-256` | adapter scaffold only |

Round-trip PSNR on 256 real ImageNet images (`scripts/eval/round_trip_psnr_imagenet.py`):
sd_vae 25.1 dB · eq_vae 24.1 dB · repa_e 24.2 dB · dc_ae_1_0 23.0 dB.

---

## Stack

Python 3.11 · PyTorch 2.6 · Lightning · Hydra · uv · timm · diffusers · SAELens ·
scikit-learn · h5py · safetensors · clean-fid · CUDA 12.x.

Hardware paths: CINECA Leonardo (A100 64 GB) for matched-compute training and the full
SAE / probe / circuit sweeps; a local 3090 / 2080 Ti for adapter smokes and short runs;
H100 pods for burst capacity (`conf/trainer/` has the multi-GPU configs).

---

## License

MIT for repo code. Vendored upstream code keeps its own LICENSE files
(`src/diffmechint/sit/LICENSE.txt` — SiT, Meta, MIT). SAELens, transformer-lens,
dictionary_learning et al. are used per their own licenses.
