"""Lightweight in-place exponential-moving-average wrapper for nn.Module."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class EMA:
    """Holds a shadow copy of a model and updates it after each optimizer step.

    The shadow weights are saved alongside the live weights in the checkpoint;
    SAE / probing pipelines load the EMA copy by convention (see PLAN §6.4).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        # Buffers (BN running stats etc.) are copied 1:1 — no smoothing.
        for sb, pb in zip(self.shadow.buffers(), model.buffers(), strict=True):
            sb.copy_(pb)

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        """Re-sync the shadow to live weights (used on initial epoch / resume)."""
        for s, p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            s.copy_(p.detach())
        for sb, pb in zip(self.shadow.buffers(), model.buffers(), strict=True):
            sb.copy_(pb)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        self.shadow.load_state_dict(sd["shadow"])
