# Phase 0 — Repo Bootstrap

Working `uv sync`, vendored SiT, smoke tests for the SiT forward pass and the FM-OT interpolant. See [README](README.md) for navigation.

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
