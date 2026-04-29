# CLAUDE.md — diffmechint repo conventions

Repo-local agent context. Defers to the user's global rules at
`~/.claude/rules/{coding-discipline,swe-stack,ml-training,flywheel,autonomous-research,safety}.md`.

## Project goal

Implementation of *"The Semantic Geometry of Diffusability"* — see
[`PLAN.md`](PLAN.md) for the full specification.

## Hard rules

- **Read `PLAN.md` before any code change.** Every PR must reference the
  phase + sub-section it implements. No off-plan work.
- **Update `CHECKLIST.md`** when starting / completing a checklist item.
  Strikethrough completed items with a one-line postmortem.
- **Hydra `_target_` everywhere.** No `if vae == "sd_vae"` branches —
  `hydra.utils.instantiate` is the canonical dispatch.
- **`uv add` only.** Never `uv pip install` (breaks pyproject lockstep).
- **Outputs → `$FAST` / `$SCRATCH`, never `$WORK`** on CINECA. Locally,
  `outputs/` is a symlink. Never commit weights.
- **Console**: use `diffmechint.utils.console` (`ok / info / warn / error`).
- **Reproducibility**: every run writes `reproducibility.md` + `commit.txt`
  to its output dir per `~/.claude/rules/flywheel.md`.

## Conditions (current)

K=5: `sd_vae`, `eq_vae`, `repa_e`, `dc_ae_1_0`, `rae`. DC-AE 1.5 enters
when upstream releases (`dc-ai-projects/DC-Gen`).

## Phases

| Phase | Subject | Status |
|---|---|---|
| 0 | Repo bootstrap | done |
| 1 | Tokenizer adapters + latent precompute | done (RAE scaffold; M/V optional) |
| 2 | SiT training (FM-OT) | smoke ✓; full run pending GPU/data |
| 3 | Activation extraction (hooks) | done |
| 4 | SAE training (multi-checkpoint) | scaffolding done; full sweep pending real DiT ckpts |
| 5 | Linear probes (Revelio grid) | pending |
| 6 | Sparse feature circuits (EAP) | pending |
| 7 | Cross-condition analysis | pending |
| 8 | Audio extension (deferred) | pending |

## Verification before merging any PR

1. `uv run pytest tests/` green.
2. `uv run ruff check .` clean.
3. The relevant `CHECKLIST.md` items are checked off with one-line postmortems.
4. New files have a one-line module docstring; no decorative comments.

## Hardware availability (live)

- **Local CPU only on this machine.** Two GPU machines reachable via SSH:
  - `100.124.107.92` (Tailscale): RTX 3090 24GB, RTX 2080 Ti 11GB —
    use only when free; check before launching.
- **Cloud pods**: existing aliases under `~/.ssh/config` for Prime
  Intellect H100/H200 — invoke `/gpu-cloud` skill if a pod is needed.
- **CINECA Leonardo**: `ssh leonardo` — reserved for full-scale Phase 2 +
  Phase 4 runs. Do not push there for smoke tests.
