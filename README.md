# diffmechint

**Semantic Geometry of Diffusability — Mechanistic Atlas of Tokenizer Interventions**

Codebase for a controlled K=5 tokenizer sweep on a matched-compute SiT
backbone, with per-checkpoint SAE training, layer × timestep linear
probing, and EAP-based sparse feature circuits.

See [`PLAN.md`](PLAN.md) for the full implementation plan and
[`CHECKLIST.md`](CHECKLIST.md) for tracked progress.

## Quick start

```bash
uv sync
uv run pytest tests/
```

## Layout

- `src/diffmechint/` — Python package
- `conf/` — Hydra configs
- `scripts/`, `slurm/` — driver scripts
- `tests/` — pytest suite
- `outputs/` — symlink to `$FAST/diffmechint/outputs/` on CINECA, gitignored

## Stack

Python 3.11 · PyTorch 2.5 · Lightning · Hydra · uv · FA2/FA3 · CUDA 12.x.
See `PLAN.md` §3 for pinned versions.

## License

MIT — vendored upstream code (SiT, dictionary_learning) carries its own
LICENSE files preserved in-tree.
