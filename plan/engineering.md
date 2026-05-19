# Engineering Practices and Robustness

Conventions for configuration, reproducibility, logging, testing, style, licensing, and long-term maintenance. See [README](README.md) for navigation.

These are the rules that keep the code maintainable past the publication
window.

## Configuration

- Hydra `_target_` everywhere — no manual `if vae == "sd_vae"` branches
  in business logic.
- Per-run config snapshot saved alongside checkpoints
  (`outputs/<run>/config.yaml`).
- Deterministic seeds: `seed_everything(42)` at run start; per-condition
  seed overrides via Hydra multirun.

## Reproducibility

- Every run writes a `reproducibility.md` to its output dir, per
  `~/.claude/rules/flywheel.md`. Lists git SHA, branch, exact CLI
  command, env (`uv pip freeze`), hardware, seeds, and SLURM job IDs.
- A `commit.txt` is the last artifact written.
- Outputs/checkpoints never overwrite a previous run's directory; run
  IDs are `{vae}_{seed}_{timestamp}` and are unique by construction.

## Logging

- Rich console for human; CSV/JSON for machines; WandB for dashboards.
- All three modes always-on; CSV is the source of truth for paper plots.

## Testing

- `pytest` unit tests for every adapter (encode→decode round-trip, shape
  invariants, scaling factor application).
- A 1k-step smoke run on SD-VAE in CI proves the pipeline is alive.
- An `eap_smoke` test validates EAP returns non-empty edges on a tiny
  synthetic circuit.

## Code style

- Ruff for linting + formatting.
- mypy strict on `src/diffmechint/tokenizers/` and
  `src/diffmechint/sae/` (the contract-heavy layers); permissive
  elsewhere.
- One-line module docstring per file; no decorative comments; no
  multi-paragraph docstrings.

## License compliance

The repo bundles MIT/Apache code. Each tokenizer adapter logs the upstream
license to its config. **VFM-VAE** weights are NVIDIA Non-Commercial —
flag `commercial_use=False` in its adapter so an automated audit can
filter out non-commercial conditions.

## Long-term maintenance

- Pin every dependency to a major + minor version, leave patch open.
- Pin upstream commit SHAs for vendored code (SiT, optional
  dictionary_learning fallback).
- Pin SAELens + transformer-lens versions in `pyproject.toml` (the SAE
  toolkit pair is the most version-sensitive part of the stack).
- Re-run `tests/` before any `uv lock --upgrade`.
- Keep `CLAUDE.md` in repo root with project-specific conventions for
  future agents.
