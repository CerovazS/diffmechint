"""Timestep router — propagates the current diffusion `t` into forward hooks.

PLAN §7.2: hooks need to know which timestep produced the activation. SiT's
forward(x, t, y) doesn't pass t to its sub-modules, so we route it via a
`ContextVar` set by the inference / training loop:

    with timestep_context(t_value):
        out = model(x, t, y)
    # hooks fired during forward see `current_t() == t_value`.

Revelio analyses use a discrete timestep grid {25, 200, 500} (out of 1000
DDPM steps). SiT uses continuous `t ∈ [0, 1]`, so the canonical mapping is
{0.025, 0.20, 0.50}. Both are exposed via `bin_revelio`.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

# (B,)-shaped tensor or scalar — whichever the loop happens to set.
_current_t: ContextVar[float | None] = ContextVar("diffmechint_current_t", default=None)

# Continuous-t bin centers (SiT convention). Override per analysis if needed.
DEFAULT_T_BINS_SIT: tuple[float, ...] = (0.025, 0.20, 0.50)
DEFAULT_T_TOL: float = 0.05  # accepts t within ±0.05 of a bin center

# Discrete-t bin centers (DDPM convention, t in 0..1000).
DEFAULT_T_BINS_DDPM: tuple[int, ...] = (25, 200, 500)


def current_t() -> float | None:
    """Return the t value set by the surrounding `timestep_context`, or None."""
    return _current_t.get()


def set_t(t: float | None) -> None:
    _current_t.set(t)


@contextmanager
def timestep_context(t: float | None) -> Iterator[None]:
    """Temporarily bind `current_t()` to `t` for the duration of the block."""
    token = _current_t.set(t)
    try:
        yield
    finally:
        _current_t.reset(token)


def bin_revelio(
    t: float,
    bins: tuple[float, ...] = DEFAULT_T_BINS_SIT,
    tol: float = DEFAULT_T_TOL,
) -> int | None:
    """Return index of nearest bin if within `tol`, else None.

    A None return means the activation should be dropped — only collect on cells
    that closely match an analysis bin.
    """
    if t is None:
        return None
    best_idx, best_dist = None, float("inf")
    for i, c in enumerate(bins):
        d = abs(float(t) - float(c))
        if d <= tol and d < best_dist:
            best_idx, best_dist = i, d
    return best_idx
