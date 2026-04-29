"""Unit tests for FractionalCheckpoint targeting + filename schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffmechint.training.checkpointing import DEFAULT_FRACTIONS, FractionalCheckpoint


def test_default_fractions_sorted_unique() -> None:
    assert list(DEFAULT_FRACTIONS) == sorted(set(DEFAULT_FRACTIONS))
    assert DEFAULT_FRACTIONS[0] >= 0.0
    assert DEFAULT_FRACTIONS[-1] <= 1.0
    assert len(DEFAULT_FRACTIONS) == 7  # 2/5/10/25/50/75/100


@pytest.mark.parametrize(
    ("max_steps", "fractions", "expected"),
    [
        (1000, (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00), [20, 50, 100, 250, 500, 750, 1000]),
        (200, (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00), [4, 10, 20, 50, 100, 150, 200]),
        (100, (0.5, 1.0), [50, 100]),
        (
            400_000,
            (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00),
            [8000, 20000, 40000, 100000, 200000, 300000, 400000],
        ),
    ],
)
def test_target_steps_from_fractions(
    tmp_path: Path,
    max_steps: int,
    fractions: tuple[float, ...],
    expected: list[int],
) -> None:
    cb = FractionalCheckpoint(out_dir=str(tmp_path), max_steps=max_steps, fractions=fractions)
    assert cb.targets == expected


def test_no_zero_step_target(tmp_path: Path) -> None:
    """Even fraction=0.0 must be clamped to step 1 (callback never saves at step 0)."""
    cb = FractionalCheckpoint(out_dir=str(tmp_path), max_steps=1000, fractions=(0.0, 1.0))
    assert min(cb.targets) >= 1


def test_dedup_when_two_fractions_round_to_same_step(tmp_path: Path) -> None:
    """E.g. on max_steps=100, fractions 0.02 and 0.05 both round to 2 and 5; check uniqueness."""
    cb = FractionalCheckpoint(
        out_dir=str(tmp_path),
        max_steps=100,
        fractions=(0.025, 0.025, 0.05, 0.05),
    )
    assert cb.targets == sorted(set(cb.targets))


def test_outdir_created(tmp_path: Path) -> None:
    new_dir = tmp_path / "nested" / "ckpts"
    FractionalCheckpoint(out_dir=str(new_dir), max_steps=10)
    assert new_dir.is_dir()
