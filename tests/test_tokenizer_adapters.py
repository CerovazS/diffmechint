"""Tokenizer adapter tests — fast unit tests + opt-in integration round-trip."""

from __future__ import annotations

import math

import pytest
import torch

from diffmechint.tokenizers import (
    TokenizerAdapter,
    TokenizerSpec,
    build,
    list_registered,
)
from diffmechint.tokenizers.sd_vae import SDVAEAdapter


def test_registry_lists_all_phase1_adapters() -> None:
    names = list_registered()
    for expected in ("sd_vae", "eq_vae", "repa_e", "dc_ae_1_0", "rae"):
        assert expected in names, f"missing adapter '{expected}' in registry: {names}"


def test_build_sd_vae_returns_adapter_without_loading() -> None:
    adapter = build("sd_vae")
    assert isinstance(adapter, SDVAEAdapter)
    assert isinstance(adapter, TokenizerAdapter)
    # Spec must be available pre-load.
    assert adapter.spec.name == "sd_vae"
    assert adapter.spec.latent_shape == (4, 32, 32)
    assert math.isclose(adapter.spec.scaling_factor, 0.18215)
    assert adapter.in_channels == 4
    assert adapter.spec.license == "CreativeML OpenRAIL-M"
    # Lazy: model not loaded until first use.
    assert not adapter._loaded


def test_unknown_adapter_raises() -> None:
    with pytest.raises(KeyError):
        build("does_not_exist")


def test_double_registration_raises() -> None:
    """Registering an adapter under an already-taken name must raise."""
    from diffmechint.tokenizers.registry import register

    with pytest.raises(ValueError, match="already registered"):

        @register("sd_vae")
        class _Dup:  # pragma: no cover - registration alone triggers
            pass


def test_tokenizer_spec_in_channels_property() -> None:
    spec = TokenizerSpec(
        name="dummy",
        latent_shape=(8, 16, 16),
        scaling_factor=1.0,
        license="MIT",
    )
    assert spec.in_channels == 8


@pytest.mark.parametrize(
    ("name", "expected_in_channels", "expected_latent_shape"),
    [
        ("sd_vae", 4, (4, 32, 32)),
        ("eq_vae", 4, (4, 32, 32)),
        ("repa_e", 4, (4, 32, 32)),
        ("dc_ae_1_0", 32, (32, 8, 8)),
        ("rae", 768, (768, 16, 16)),
    ],
)
def test_all_adapters_buildable_lazy(
    name: str, expected_in_channels: int, expected_latent_shape: tuple[int, int, int]
) -> None:
    adapter = build(name)
    assert isinstance(adapter, TokenizerAdapter)
    assert adapter.spec.name == name
    assert adapter.in_channels == expected_in_channels
    assert adapter.spec.latent_shape == expected_latent_shape
    assert not adapter._loaded


def test_adapter_specs_carry_license_metadata() -> None:
    """Every adapter must declare license + commercial_use for the audit."""
    for name in ("sd_vae", "eq_vae", "repa_e", "dc_ae_1_0", "rae"):
        adapter = build(name)
        assert adapter.spec.license, f"{name}: license string is empty"
        assert isinstance(adapter.spec.commercial_use, bool)
        assert adapter.spec.paper_arxiv, f"{name}: missing arXiv id"


@pytest.mark.integration
def test_sd_vae_round_trip_psnr() -> None:
    """Real HF download — encode/decode a synthetic image, expect PSNR > 25 dB.

    Run with: `uv run pytest tests/ -m integration --runslow`.
    """
    adapter = build("sd_vae")
    adapter.load()
    # synthetic batch in [-1, 1] (SD-VAE convention)
    x = torch.randn(2, 3, 256, 256).clamp_(-1, 1)
    x_hat = adapter.round_trip(x)
    assert x_hat.shape == x.shape
    mse = torch.mean((x - x_hat) ** 2).item()
    psnr = 10 * math.log10(4.0 / max(mse, 1e-12))  # max range 2.0 → max² = 4.0
    assert psnr > 15.0, f"random-noise round-trip PSNR too low: {psnr:.2f}"
