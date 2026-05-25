# SiT L/2 Refactor Branch Plan

## Objective

Refactor the existing pipeline so that SiT-B/2 and SiT-L/2 can coexist without
colliding in run outputs, activation caches, SAE checkpoints, probe outputs, or
analysis artifacts.

A separate git branch is not technically required. The safer implementation is
to make the code model-variant aware and keep all existing SiT-B/2 artifacts
readable in their current legacy locations.

The first larger model target is SiT-L/2:

- already configured in `conf/model/sit_l_2.yaml`;
- compatible with the existing precomputed `4x32x32` latents for `sd_vae`,
  `eq_vae`, and `repa_e`;
- approximately 458M parameters, versus approximately 130M for SiT-B/2;
- uses the same `/2` patching and therefore the same 256-token latent grid for
  SD-VAE-like tokenizers.

SiT-XL/2 exists in the codebase, but should remain a later second scale jump.
DC-AE should be handled separately because its `32x8x8` latent shape makes
`SiT-L/1` the more comparable patching choice.

## Current Repository State

The SiT training path is already mostly parametrized:

- `scripts/training/train_sit_full.sh` accepts `MODEL`, defaulting to `sit_b_2`;
- `slurm/train_sit_full.slurm` forwards `MODEL`;
- `src/diffmechint/training/train.py` instantiates the model through Hydra;
- `src/diffmechint/sit/models.py` already exposes `SiT-L/2`;
- `src/diffmechint/hooks/activation_taps.py` already provides
  `default_tap_layers(depth)`;
- current expected tap layers are:
  - SiT-B depth 12: `[3, 6, 9]`;
  - SiT-L depth 24: `[6, 12, 18]`;
  - SiT-XL depth 28: `[7, 14, 21]`.

The downstream path is still implicitly SiT-B/2:

- `scripts/extraction/extract_activations.py` defaults to `SiT-B/2`, layers
  `[3, 6, 9]`, and `$SCRATCH/diffmechint/activations`;
- `scripts/training/train_sae.py` defaults to `d_in=768`, layers `[3, 6, 9]`,
  and an SAE root without a model namespace;
- `scripts/analysis/run_revelio_grid.py`,
  `scripts/analysis/tokenizer_dictionary_validation.py`,
  `scripts/analysis/sae_feature_patching.py`, atlas scripts, dashboard scripts,
  and several eval scripts assume B/2 through hardcoded layers, B/2-specific
  roots, or `SiT-B/2` defaults;
- local outputs are phase-based (`outputs/phase4_*`, `outputs/phase5_*`) rather
  than model-based;
- large outputs on `$SCRATCH` are type/tokenizer-based (`activations/eq_vae/...`)
  rather than model-based.

Existing outputs must not be moved, deleted, or overwritten. They remain legacy
SiT-B/2 artifacts.

## Implementation Changes

### Model Variant Utility

Add a central utility, for example `src/diffmechint/utils/model_variants.py`,
that owns:

- stable ids such as `sit_b_2`, `sit_l_2`, `sit_xl_2`;
- conversion from Hydra config names and SiT model names:
  - `sit_l_2` -> `SiT-L/2`;
  - `SiT-L/2` -> `sit_l_2`;
- construction of a small model metadata object:
  - `model_name`;
  - `variant_id`;
  - `depth`;
  - `hidden_size`;
  - `patch_size`;
  - `tap_layers`;
- reusable path helpers for model-namespaced roots.

The implementation should derive metadata from the existing SiT model registry
instead of duplicating a large manual table. A small explicit map is acceptable
only for name normalization.

### Output Layout

Use the following layout for new outputs:

```text
outputs/by_model/
  sit_b_2/
    analysis/
    probes/
    atlas/
    dashboards/
    patching/
  sit_l_2/
    analysis/
    probes/
    atlas/
    dashboards/
    patching/

$SCRATCH/diffmechint/by_model/
  sit_b_2/
    runs/
    activations/
    activations_val/
    activations_ynull/
  sit_l_2/
    runs/
    activations/
    activations_val/
    activations_ynull/

$FAST/.../diffmechint/by_model/
  sit_b_2/
    sae/
  sit_l_2/
    sae/
```

The old roots remain supported for reading:

- `outputs/phase4_*`;
- `outputs/phase5_*`;
- `$SCRATCH/diffmechint/runs`;
- `$SCRATCH/diffmechint/activations*`;
- `$FAST/.../sae_matryoshka_k256_d32k`.

Do not perform an automatic migration of old artifacts. If needed later, add a
separate non-destructive indexing or symlink script.

### Training

Update `scripts/training/train_sit_full.sh` so that new runs are written under:

```text
$SCRATCH/diffmechint/by_model/<model_variant>/runs/<run_id>
```

Keep the run id explicit:

```text
sit_l_2_eq_vae_l2_noz_full_<timestamp>
```

Write or extend per-run metadata so that every run records:

- `model_variant`;
- `model_name`;
- tokenizer condition;
- normalization mode;
- latent shape;
- output directory;
- git commit and branch when available.

Keep backward compatibility for existing run directories by accepting explicit
paths everywhere downstream.

### Activation Extraction

Update `scripts/extraction/extract_activations.py`:

- default `--layers` should become `auto`;
- `auto` resolves through model depth to `[3, 6, 9]` for B/2 and
  `[6, 12, 18]` for L/2;
- add or derive `model_variant`;
- default output root should be model-namespaced for new runs;
- keep explicit `--out_root` as an override for legacy compatibility.

Every extraction `manifest.json` should include:

- `model_name`;
- `model_variant`;
- `depth`;
- `hidden_size`;
- `patch_size`;
- `layers`;
- `bins`;
- `source_run`;
- existing sample and seed fields.

### SAE Training

Update `scripts/training/train_sae.py`:

- support `--model_variant`;
- support `--layers auto`;
- support `--d_in auto`, inferred from the first activation HDF5 shard;
- write new SAE chains under:

```text
$FAST/.../diffmechint/by_model/<model_variant>/sae/<sae_variant>/<condition>/L<layer>_T<t_bin>/
```

Do not overwrite or reuse the existing B/2 root
`sae_matryoshka_k256_d32k` for L/2.

The first L/2 SAE configuration should be named separately, even if the
hyperparameters match B/2 for an initial comparability pass.

### Probe, Atlas, Dashboard, and Analysis

Update the main downstream scripts to accept model-aware roots:

- `scripts/analysis/run_revelio_grid.py`;
- `scripts/analysis/tokenizer_dictionary_validation.py`;
- `scripts/analysis/sae_feature_patching.py`;
- `scripts/analysis/feature_dashboard.py`;
- `scripts/analysis/recompute_histograms.py`;
- atlas scripts under `scripts/analysis/atlas/`;
- relevant eval scripts under `scripts/eval/`.

For these scripts:

- replace hardcoded `LAYERS = (3, 6, 9)` with explicit CLI/config layers or
  manifest-derived layers;
- write new outputs under `outputs/by_model/<model_variant>/...`;
- keep current defaults usable for legacy B/2 paths where practical;
- mark scripts as B/2-only in docstrings if they are not generalized in this
  refactor.

### Target SiT-L/2 Commands

Initial L/2 training commands:

```bash
sbatch --export=TOK=sd_vae,MODEL=sit_l_2,NORMALIZE=true,RUN_SUFFIX=l2,CALLBACKS=sample_only slurm/train_sit_full.slurm
sbatch --export=TOK=repa_e,MODEL=sit_l_2,NORMALIZE=true,RUN_SUFFIX=l2,CALLBACKS=sample_only slurm/train_sit_full.slurm
sbatch --export=TOK=eq_vae,MODEL=sit_l_2,NORMALIZE=false,RUN_SUFFIX=l2_noz,CALLBACKS=sample_only slurm/train_sit_full.slurm
```

Expected SiT-L/2 tap layers:

```text
[6, 12, 18]
```

## Test Plan

Add or update tests for:

- variant normalization:
  - `SiT-B/2 -> sit_b_2`;
  - `SiT-L/2 -> sit_l_2`;
  - `sit_l_2 -> SiT-L/2`;
- model metadata:
  - B/2 depth 12, hidden size 768, layers `[3, 6, 9]`;
  - L/2 depth 24, hidden size 1024, layers `[6, 12, 18]`;
- output path helpers;
- extraction layer auto-resolution;
- SAE `d_in auto` from synthetic HDF5 with `D=1024`.

Run:

```bash
uv run pytest tests/
uv run ruff check .
```

## Assumptions

- The implementation must preserve all existing outputs.
- New outputs should be model-namespaced by default.
- Existing scripts should remain able to read legacy B/2 artifacts.
- `sit_b_2` and `sit_l_2` are the only operational targets for this refactor.
- `sit_xl_2` should work through the same abstractions but is not required for
  the first experiment.
- DC-AE remains outside this refactor.
- Any pre-existing dirty worktree changes must be preserved.
