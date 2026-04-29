"""DC-AE 1.0 (MIT-Han-Lab, f32c32) tokenizer — 32× spatial, 32-channel latent."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import TokenizerAdapter, TokenizerSpec
from .registry import register

# DC-AE 1.0: paper arXiv 2410.10733, Apache-2.0, MIT-Han-Lab.
# Diffusers >= 0.30 ships `AutoencoderDC` and supports the `-diffusers` HF variant
# directly. Verified locally on diffusers 0.37.1: `from diffusers import AutoencoderDC`
# imports cleanly. We default to the diffusers-compatible repo id; the efficientvit
# package is the documented fallback if the diffusers loader is ever unavailable.
DC_AE_1_0_SPEC = TokenizerSpec(
    name="dc_ae_1_0",
    latent_shape=(32, 8, 8),  # 256² input → 32× spatial compression, 32-channel latent
    scaling_factor=1.0,  # DC-AE applies learned scaling internally; no post-encode multiply
    license="Apache-2.0",
    commercial_use=True,
    paper_arxiv="2410.10733",
)


@register("dc_ae_1_0")
class DCAE10Adapter(TokenizerAdapter):
    """Wraps `mit-han-lab/dc-ae-f32c32-in-1.0-diffusers` via diffusers `AutoencoderDC`."""

    HF_REPO_ID = "mit-han-lab/dc-ae-f32c32-in-1.0-diffusers"

    def __init__(self, hf_repo_id: str | None = None) -> None:
        super().__init__(DC_AE_1_0_SPEC)
        self.hf_repo_id = hf_repo_id or self.HF_REPO_ID
        self.vae = None  # populated by load()

    def load(self) -> None:
        if self._loaded:
            return
        from diffusers import AutoencoderDC

        self.vae = AutoencoderDC.from_pretrained(self.hf_repo_id)
        self.freeze()
        self._loaded = True

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        if not self._loaded:
            self.load()
        # AutoencoderDC.encode returns an EncoderOutput with `latent` (deterministic;
        # DC-AE is not a VAE — no posterior). Learned scaling is baked into the model.
        z = self.vae.encode(x).latent * self.spec.scaling_factor
        return z

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        if not self._loaded:
            self.load()
        x_hat = self.vae.decode(z / self.spec.scaling_factor).sample
        return x_hat
