"""ResidualStreamTap — register forward hooks on chosen SiTBlock indices."""

from __future__ import annotations

from typing import Sequence

from torch import nn
from torch.utils.hooks import RemovableHandle

from .activation_buffer import ActivationBuffer
from .timestep_router import bin_revelio, current_t


class ResidualStreamTap:
    """Attach forward-hooks to selected SiT blocks; route activations into a buffer.

    Usage:
        tap = ResidualStreamTap(model, block_indices=[3, 6, 9], buffer=buf)
        tap.attach()
        with timestep_context(0.20):
            model(x, t=torch.full((B,), 0.20), y=y)
        tap.detach()
        # buf now holds activations for layers 3/6/9 at t-bin matching 0.20.

    Args:
      model: any module exposing a `.blocks` ModuleList with `.block_idx` attrs
             (true for our vendored SiT, see `src/diffmechint/sit/models.py`).
      block_indices: which block indices to tap.
      buffer: ActivationBuffer to write into.
      bins: continuous-t bin centers used to label activations. Hooks only
            record when `bin_revelio(current_t())` returns a non-None index.
            If `bins=None` records every time the t is set, with t_bin=0.
    """

    def __init__(
        self,
        model: nn.Module,
        block_indices: Sequence[int],
        buffer: ActivationBuffer,
        bins: tuple[float, ...] | None = (0.025, 0.20, 0.50),
        tol: float = 0.05,
    ) -> None:
        if not hasattr(model, "blocks"):
            raise AttributeError("model has no `.blocks` attribute (expected SiT-style backbone).")
        self.model = model
        self.block_indices = list(block_indices)
        self.buffer = buffer
        self.bins = bins
        self.tol = tol
        self._handles: list[RemovableHandle] = []
        # Validate indices up-front.
        n_blocks = len(model.blocks)
        for idx in self.block_indices:
            if not (0 <= idx < n_blocks):
                raise IndexError(f"block index {idx} out of range [0, {n_blocks}).")

    def _make_hook(self, idx: int):
        bins = self.bins
        tol = self.tol
        buffer = self.buffer

        def hook(module, inputs, output):  # noqa: ARG001
            t = current_t()
            if t is None:
                return  # no timestep set → drop
            if bins is None:
                t_bin = 0
            else:
                t_bin = bin_revelio(float(t), bins=bins, tol=tol)
                if t_bin is None:
                    return  # outside any analysis bin → drop
            # output: (B, T, D)
            buffer.write(layer=idx, t_bin=t_bin, x=output)

        return hook

    def attach(self) -> "ResidualStreamTap":
        if self._handles:
            raise RuntimeError("attach() called while previous hooks still active. detach() first.")
        for idx in self.block_indices:
            block = self.model.blocks[idx]
            self._handles.append(block.register_forward_hook(self._make_hook(idx)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "ResidualStreamTap":
        return self.attach()

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.detach()


def default_tap_layers(depth: int) -> list[int]:
    """Return the {25%, 50%, 75%} block indices for a given backbone depth.

    Per PLAN §7.4:
      SiT-B  (depth 12) → {3, 6, 9}
      SiT-L  (depth 24) → {6, 12, 18}
      SiT-XL (depth 28) → {7, 14, 21}
    """
    return [
        max(1, depth // 4),
        depth // 2,
        max(depth // 2 + 1, (3 * depth) // 4),
    ]
