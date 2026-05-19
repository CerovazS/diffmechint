# Phase 1 — Tokenizer Adapters

One adapter per VAE family with a shared interface so the Lightning training module is tokenizer-agnostic. See [README](README.md) for navigation.

Every VAE/tokenizer becomes a `TokenizerAdapter` exposing the same
interface so the Lightning training module is tokenizer-agnostic.

## The adapter contract

`src/diffmechint/tokenizers/base.py`:

```python
class TokenizerAdapter(abc.ABC):
    name: str
    latent_shape: tuple[int, int, int]   # (C, H, W) at 256² input
    scaling_factor: float
    license: str                         # for audit
    
    @abc.abstractmethod
    def encode(self, x: Tensor) -> Tensor: ...   # x: (B, 3, 256, 256)
    
    @abc.abstractmethod
    def decode(self, z: Tensor) -> Tensor: ...
    
    @property
    def in_channels(self) -> int:        # for SiT in_channels arg
        return self.latent_shape[0]
```

## Adapter implementations

Each concrete adapter wraps the upstream loader. All weights pulled from
HuggingFace Hub at run time, cached under `$FAST/hf_cache/`.

| Adapter | Loader call | Latent shape (256²) | Scaling | Notes |
|---|---|---|---|---|
| `sd_vae` | `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")` | (4, 32, 32) | 0.18215 | baseline |
| `eq_vae` | `AutoencoderKL.from_pretrained("zelaki/eq-vae-ema")` | (4, 32, 32) | 0.18215 | drop-in `diffusers` |
| `repa_e` | `AutoencoderKL.from_pretrained("REPA-E/e2e-sdvae-hf")` | (4, 32, 32) | 0.18215 | jointly-trained VAE |
| `dc_ae_1_0` | from `mit-han-lab/dc-ae-f32c32-in-1.0` via custom loader | (32, 8, 8) | learned | non-diffusers, see efficientvit |
| `rae` | `nyu-visionx/rae-dinov2-base-vitxl-n08` | high-dim token grid | learned | flow-matching DiT pairing |
| `maetok` | `MAETok/maetok-b-128` | 128 tokens × D | learned | continuous tokenizer |
| `va_vae` | `hustvl/imagenet256-latents-vave-f16d32-dinov2` | (32, 16, 16) | learned | DINOv2-aligned |

## Latent shape adaptation in SiT

SiT's `in_channels` is hard-coded by config. The Lightning module reads
`adapter.in_channels` and instantiates the SiT with that value. For
non-grid latents (RAE, MAETok), use a **canonical-token-grid view** — the
adapter exposes `(B, T, D)` and the Lightning module reshapes/projects to
SiT's `(B, C, H', W')` expectation, or swaps in a 1D-token SiT variant.
**Decision rule:** if `latent_shape` has ≤ 32 channels and a grid layout,
use the standard SiT 2D model; otherwise wrap with a `TokenGridAdapter`
(linear projection + spatial reshape) before SiT entry.

## Latent precomputation

Encoding ImageNet through every VAE on every epoch is wasteful. Pre-encode
once to `$FAST`:

`src/diffmechint/training/precompute_latents.py`:
- Reads ImageNet-256 (mounted from `$FAST/datasets/imagenet256/`).
- For each registered tokenizer, dumps encoded latents to
  `$FAST/diffmechint/latents/{tokenizer_name}/{shard:05d}.h5` with
  `(z, label)` pairs in fp16.
- ~150 GB per VAE per ImageNet split (raw + cached).
- Single-A100 ~2 h per VAE per 1M images.

**Acceptance:** for each adapter, encode→decode round-trip on a held-out
batch produces images with PSNR > 25 dB; latent statistics
(mean/std/min/max) logged to `outputs/precompute/<vae>/stats.json`.
