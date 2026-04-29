"""Abstract tokenizer contract — every VAE / autoencoder family implements this."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TokenizerSpec:
    """Static metadata an adapter exposes without loading weights."""

    name: str
    latent_shape: tuple[int, int, int]  # (C, H, W) at 256² input
    scaling_factor: float                # multiplicative pre-DiT scaling, fixed or learned
    license: str                         # for compliance audit
    commercial_use: bool = True
    paper_arxiv: str = ""                # optional arXiv id

    @property
    def in_channels(self) -> int:
        return self.latent_shape[0]


class TokenizerAdapter(nn.Module, abc.ABC):
    """One adapter per VAE/tokenizer family.

    Contract (matches PLAN.md §5.1):
      - encode(x: (B, 3, H, W)) -> z: (B, C, H', W')   # already scaled, ready for DiT
      - decode(z: (B, C, H', W')) -> x_hat: (B, 3, H, W)
      - The adapter holds the upstream model in eval/no-grad mode by default.
      - Adapters are *not* trained — their weights are frozen.
    """

    spec: TokenizerSpec

    def __init__(self, spec: TokenizerSpec) -> None:
        super().__init__()
        self.spec = spec
        self._loaded = False

    @abc.abstractmethod
    def load(self) -> None:
        """Lazy-load upstream model weights (HF Hub, local cache, etc.)."""

    @abc.abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """RGB image batch → DiT-ready scaled latent."""

    @abc.abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """DiT-ready scaled latent → RGB image batch."""

    @property
    def in_channels(self) -> int:
        return self.spec.in_channels

    def freeze(self) -> None:
        """Freeze all parameters and put module in eval mode."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def round_trip(self, x: Tensor) -> Tensor:
        """Encode then decode (debug / acceptance helper)."""
        with torch.no_grad():
            return self.decode(self.encode(x))

    def __repr__(self) -> str:  # noqa: D401
        return f"{self.__class__.__name__}(name={self.spec.name}, in_ch={self.in_channels})"
