"""Abstract tokenizer contract — every VAE / autoencoder family implements this."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TokenizerSpec:
    """Static metadata an adapter exposes without loading weights.

    `kind` + `feature_axis` make the spec self-describing for downstream
    consumers (precompute stats, DiT input head, SAE, probes) so that none of
    them need to hardcode the layout per VAE family.
    """

    name: str
    latent_shape: tuple[int, ...]  # (C, H, W) for spatial; (T, D) for sequence
    scaling_factor: float                # multiplicative pre-DiT scaling, fixed or learned
    license: str                         # for compliance audit
    commercial_use: bool = True
    paper_arxiv: str = ""                # optional arXiv id
    # Layout-awareness — keeps the codebase honest when sequence-token VAEs land.
    kind: str = "spatial"                # "spatial" (B,C,H,W) | "sequence" (B,T,D)
    feature_axis: int = 1                # axis (in the batched tensor) holding features
    suggested_patch_size: int | None = 2  # SiT patchify hint; None for sequence layouts

    @property
    def in_channels(self) -> int:
        # For spatial latents the feature dim is the channel count (axis 1 → idx 0
        # in the post-batch shape). For sequence latents it's the last dim. The
        # feature_axis field encodes both cases: post-batch index = feature_axis - 1.
        idx = self.feature_axis - 1 if self.feature_axis >= 0 else self.feature_axis
        return self.latent_shape[idx]

    @property
    def feature_dim(self) -> int:
        return self.in_channels

    @property
    def input_size(self) -> int:
        """Spatial side length for grid latents; sequence length for token latents."""
        if self.kind == "spatial":
            return self.latent_shape[1]  # H (assumes H == W, true for all 4 active VAEs)
        # Sequence: latent_shape == (T, D), so axis 0 (post-batch) is T.
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
