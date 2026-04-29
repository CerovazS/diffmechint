"""Phase 3 acceptance tests for hooks + timestep router + activation buffer."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import torch

from diffmechint.hooks import (
    ActivationBuffer,
    ResidualStreamTap,
    bin_revelio,
    current_t,
    default_tap_layers,
    timestep_context,
)
from diffmechint.sit import SiT_models


# ---------------------------------------------------------------------------
# timestep_router
# ---------------------------------------------------------------------------
def test_current_t_default_none() -> None:
    assert current_t() is None


def test_timestep_context_sets_and_restores() -> None:
    assert current_t() is None
    with timestep_context(0.5):
        assert current_t() == 0.5
        with timestep_context(0.025):
            assert current_t() == 0.025
        assert current_t() == 0.5
    assert current_t() is None


@pytest.mark.parametrize(
    ("t", "expected_bin"),
    [
        (0.025, 0),
        (0.04, 0),
        (0.20, 1),
        (0.18, 1),
        (0.50, 2),
        (0.46, 2),
        (0.10, None),  # between bin 0 (0.025) and bin 1 (0.20), tol=0.05 → None
        (0.99, None),
    ],
)
def test_bin_revelio_default_grid(t: float, expected_bin: int | None) -> None:
    assert bin_revelio(t) == expected_bin


def test_bin_revelio_none_input() -> None:
    assert bin_revelio(None) is None


# ---------------------------------------------------------------------------
# ActivationBuffer
# ---------------------------------------------------------------------------
def test_buffer_write_splits_batch() -> None:
    buf = ActivationBuffer()
    x = torch.randn(4, 16, 32)
    buf.write(layer=0, t_bin=1, x=x)
    records = buf.get(0, 1)
    assert len(records) == 4
    assert records[0].shape == (16, 32)


def test_buffer_rejects_non_3d() -> None:
    buf = ActivationBuffer()
    with pytest.raises(ValueError, match=r"\(B, T, D\)"):
        buf.write(layer=0, t_bin=0, x=torch.randn(4, 16))


def test_buffer_keys_sorted() -> None:
    buf = ActivationBuffer()
    buf.write(2, 1, torch.randn(1, 8, 4))
    buf.write(0, 0, torch.randn(1, 8, 4))
    buf.write(1, 2, torch.randn(1, 8, 4))
    assert buf.keys() == [(0, 0), (1, 2), (2, 1)]


def test_buffer_flush_to_disk(tmp_path: Path) -> None:
    buf = ActivationBuffer(shard_dir=tmp_path)
    buf.write(layer=3, t_bin=1, x=torch.randn(2, 16, 32))
    buf.write(layer=3, t_bin=1, x=torch.randn(3, 16, 32))
    buf.flush_all()
    shard = tmp_path / "3_1.h5"
    assert shard.is_file()
    with h5py.File(shard, "r") as f:
        assert f["activations"].shape == (5, 16, 32)
        assert f["activations"].dtype == "float16"
    # After flush the in-memory cell should be empty.
    assert buf.get(3, 1) == []


def test_buffer_auto_flush_on_capacity(tmp_path: Path) -> None:
    buf = ActivationBuffer(max_records_per_cell=4, shard_dir=tmp_path)
    buf.write(layer=0, t_bin=0, x=torch.randn(5, 8, 4))  # 5 > 4 triggers flush
    shard = tmp_path / "0_0.h5"
    assert shard.is_file()
    assert buf.get(0, 0) == []


# ---------------------------------------------------------------------------
# ResidualStreamTap end-to-end on SiT-B/2 (PLAN §7.5 acceptance)
# ---------------------------------------------------------------------------
def test_default_tap_layers_for_each_size() -> None:
    assert default_tap_layers(12) == [3, 6, 9]   # SiT-B
    assert default_tap_layers(24) == [6, 12, 18]  # SiT-L
    assert default_tap_layers(28) == [7, 14, 21]  # SiT-XL


def test_phase3_acceptance_36_records() -> None:
    """PLAN §7.5: 4-image batch × 3 layers × 3 timesteps → 36 records, shape (T, D)."""
    model = SiT_models["SiT-B/2"](input_size=32, in_channels=4, num_classes=1000)
    model.eval()
    buf = ActivationBuffer()
    block_indices = default_tap_layers(len(model.blocks))  # [3, 6, 9]

    B = 4
    x = torch.randn(B, 4, 32, 32)
    y = torch.randint(0, 1000, (B,))

    with ResidualStreamTap(model, block_indices, buf):
        for t_value in (0.025, 0.20, 0.50):
            t = torch.full((B,), t_value)
            with timestep_context(t_value), torch.no_grad():
                model(x, t, y)

    # 3 layers × 3 timesteps = 9 cells, each with B=4 records.
    assert len(buf.keys()) == 9
    for k in buf.keys():
        assert len(buf.get(*k)) == B
    assert len(buf) == 36
    # Check shape: (B, T, D) where T = (32/2)² = 256, D = 768 for SiT-B.
    sample = buf.get(3, 0)[0]
    assert sample.shape == (256, 768)


def test_tap_drops_when_t_outside_bins() -> None:
    """If `current_t()` is not near any bin, hook records nothing."""
    model = SiT_models["SiT-B/2"](input_size=32, in_channels=4, num_classes=10)
    model.eval()
    buf = ActivationBuffer()
    with ResidualStreamTap(model, [3, 6, 9], buf):
        with timestep_context(0.99), torch.no_grad():  # not near 0.025/0.20/0.50
            model(torch.randn(2, 4, 32, 32), torch.full((2,), 0.99), torch.randint(0, 10, (2,)))
    assert len(buf) == 0


def test_tap_drops_when_t_unset() -> None:
    model = SiT_models["SiT-B/2"](input_size=32, in_channels=4, num_classes=10)
    model.eval()
    buf = ActivationBuffer()
    with ResidualStreamTap(model, [3, 6, 9], buf):
        with torch.no_grad():
            # No timestep_context wrapping → current_t() returns None.
            model(torch.randn(2, 4, 32, 32), torch.full((2,), 0.5), torch.randint(0, 10, (2,)))
    assert len(buf) == 0


def test_tap_attach_detach_cleanup() -> None:
    model = SiT_models["SiT-B/2"](input_size=32, in_channels=4)
    tap = ResidualStreamTap(model, [3], ActivationBuffer())
    tap.attach()
    assert len(tap._handles) == 1
    tap.detach()
    assert tap._handles == []
    # After detach, the hook should NOT fire any more.
    buf = ActivationBuffer()
    with timestep_context(0.20), torch.no_grad():
        model(torch.randn(1, 4, 32, 32), torch.full((1,), 0.20), torch.zeros(1, dtype=torch.long))
    assert len(buf) == 0


def test_tap_re_attach_disallowed_without_detach() -> None:
    model = SiT_models["SiT-B/2"](input_size=32, in_channels=4)
    tap = ResidualStreamTap(model, [3], ActivationBuffer())
    tap.attach()
    with pytest.raises(RuntimeError, match="detach"):
        tap.attach()
    tap.detach()


def test_tap_invalid_block_index() -> None:
    model = SiT_models["SiT-B/2"](input_size=32, in_channels=4)
    with pytest.raises(IndexError):
        ResidualStreamTap(model, [99], ActivationBuffer())
