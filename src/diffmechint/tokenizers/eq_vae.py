"""EQ-VAE (equivariance-regularized VAE) — drop-in replacement for SD-VAE."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import TokenizerAdapter, TokenizerSpec
from .registry import register

EQ_VAE_SPEC = TokenizerSpec(
    name="eq_vae",
    latent_shape=(4, 32, 32),
    scaling_factor=0.18215,
    license="not declared (verify before commercial use)",
    commercial_use=False,
    paper_arxiv="2502.09509",
)


@register("eq_vae")
class EQVAEAdapter(TokenizerAdapter):
    """Wraps `zelaki/eq-vae-ema` (4-channel f8 latent, 256² → 32², SD-VAE-compatible)."""

    HF_REPO_ID = "zelaki/eq-vae-ema"

    def __init__(self, hf_repo_id: str | None = None) -> None:
        super().__init__(EQ_VAE_SPEC)
        self.hf_repo_id = hf_repo_id or self.HF_REPO_ID
        self.vae = None  # populated by load()

    def load(self) -> None:
        if self._loaded:
            return
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(self.hf_repo_id)
        self.freeze()
        self._loaded = True

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        if not self._loaded:
            self.load()
        # Use posterior mean (not sample) for deterministic encoding.
        latent_dist = self.vae.encode(x).latent_dist
        z = latent_dist.mean * self.spec.scaling_factor
        return z

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        if not self._loaded:
            self.load()
        x_hat = self.vae.decode(z / self.spec.scaling_factor).sample
        return x_hat
