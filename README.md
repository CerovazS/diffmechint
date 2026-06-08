# diffmechint

**Semantic Geometry of Diffusability — a mechanistic-interpretability study of
latent-diffusion tokenizers.**

The project trains a single **SiT** (Scalable Interpolant Transformer, Flow-Matching +
Optimal-Transport) backbone on **K controlled VAE / tokenizer variants** over
ImageNet-256, then runs a fixed mech-int protocol on each condition: sparse
autoencoders (SAEs), layer × diffusion-timestep linear probes, and sparse
feature-circuit attribution. The goal is to read out *which* features the diffusion
transformer learns, *in what order*, and whether that is tokenizer-invariant or
tokenizer-shaped.

This README explains **how the repo is organized and how to run the files and
experiments**. The curated research record (hypotheses, results, evidence) lives in
the Flywheel graph; [`index.md`](index.md) is a local mirror of it.

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

## Conditions (K = 5)

| condition | cluster | hf repo / source | adapter |
|---|---|---|---|
| `sd_vae`    | baseline                       | `stabilityai/sd-vae-ft-mse`                 | 🟢 working |
| `eq_vae`    | spectral / equivariance        | `zelaki/eq-vae-ema`                         | 🟢 working |
| `repa_e`    | semantic alignment (joint VAE) | `REPA-E/e2e-sdvae-hf`                        | 🟢 working |
| `dc_ae_1_0` | information-ordered bottleneck | `mit-han-lab/dc-ae-f32c32-in-1.0-diffusers` | 🟢 working |
| `rae`       | discriminative encoder (DINOv2)| `nyu-visionx/rae-dinov2-base-vitxl-n08-256` | 🟡 scaffold |

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
