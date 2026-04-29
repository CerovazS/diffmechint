"""RAE (Representation Autoencoder, NYU VisionX) tokenizer — DINOv2 token-grid latent."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import TokenizerAdapter, TokenizerSpec
from .registry import register

# RAE: paper arXiv 2510.11690, MIT, NYU VisionX (https://github.com/bytetriper/RAE).
# RAE pairs a frozen pretrained discriminative encoder (DINOv2-base by default) with
# a trained ViT decoder. The latent is the DINOv2 token grid with the CLS token
# dropped: for 256² input and DINOv2-base patch16, that is (B, 256, 768), which we
# expose to SiT as the spatial layout (768, 16, 16). RAE uses no global multiplicative
# scaling — the encoder is fixed and the decoder is trained against its raw outputs.
RAE_SPEC = TokenizerSpec(
    name="rae",
    latent_shape=(768, 16, 16),  # DINOv2-base: 16×16 patches, D=768, CLS dropped
    scaling_factor=1.0,
    license="MIT",
    commercial_use=True,
    paper_arxiv="2510.11690",
)


@register("rae")
class RAEAdapter(TokenizerAdapter):
    """Wraps `nyu-visionx/rae-dinov2-base-vitxl-n08-256` (frozen DINOv2 encoder + ViT decoder).

    Loading is non-trivial — RAE is not a standard diffusers model. The full
    implementation must:
      1. Load the frozen encoder via `AutoModel.from_pretrained("facebook/dinov2-base")`.
      2. Instantiate the upstream RAE ViT decoder class (see github.com/bytetriper/RAE)
         and load weights from the `nyu-visionx/rae-dinov2-base-vitxl-n08-256` checkpoint.
      3. In `encode`: forward through DINOv2, drop the CLS token, reshape (B, 256, 768)
         to (B, 768, 16, 16) for SiT compatibility.
      4. In `decode`: reverse the spatial reshape to (B, 256, 768) and run the ViT decoder.
    """

    HF_REPO_ID = "nyu-visionx/rae-dinov2-base-vitxl-n08-256"
    ENCODER_REPO_ID = "facebook/dinov2-base"

    def __init__(
        self,
        hf_repo_id: str | None = None,
        encoder_repo_id: str | None = None,
    ) -> None:
        super().__init__(RAE_SPEC)
        self.hf_repo_id = hf_repo_id or self.HF_REPO_ID
        self.encoder_repo_id = encoder_repo_id or self.ENCODER_REPO_ID
        self.encoder = None  # frozen DINOv2
        self.decoder = None  # trained ViT decoder

    def load(self) -> None:
        # TODO: implement full loader. See model card at
        # https://huggingface.co/nyu-visionx/rae-dinov2-base-vitxl-n08-256 and the
        # upstream repo https://github.com/bytetriper/RAE for the decoder class and
        # checkpoint layout. Required: vendor (or import) the RAE ViT decoder
        # definition, load DINOv2-base via transformers `AutoModel`, then load the
        # decoder state_dict from the HF checkpoint.
        raise NotImplementedError(
            "RAE loader not yet implemented — see "
            "https://huggingface.co/nyu-visionx/rae-dinov2-base-vitxl-n08-256 "
            "and https://github.com/bytetriper/RAE for the decoder definition."
        )

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        if not self._loaded:
            self.load()
        # TODO: forward through frozen DINOv2, drop CLS token, reshape
        # (B, 256, 768) → (B, 768, 16, 16).
        raise NotImplementedError("RAE.encode awaits load() implementation.")

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        if not self._loaded:
            self.load()
        # TODO: reshape (B, 768, 16, 16) → (B, 256, 768) and run ViT decoder.
        raise NotImplementedError("RAE.decode awaits load() implementation.")
