"""SiT forward-pass smoke + hook self-describe smoke."""

from __future__ import annotations

import torch

from diffmechint.sit import SiT_models, SiTBlock


def test_sit_b_2_forward_shape() -> None:
    model = SiT_models["SiT-B/2"](input_size=32, in_channels=4, num_classes=1000)
    model.eval()
    B = 2
    x = torch.randn(B, 4, 32, 32)
    t = torch.rand(B)
    y = torch.randint(0, 1000, (B,))
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == (B, 4, 32, 32)


def test_sit_block_idx_assigned() -> None:
    model = SiT_models["SiT-B/2"](input_size=32, in_channels=4)
    for i, block in enumerate(model.blocks):
        assert isinstance(block, SiTBlock)
        assert hasattr(block, "block_idx")
        assert block.block_idx == i
    # SiT-B has 12 blocks
    assert len(model.blocks) == 12


def test_forward_hook_fires_per_block() -> None:
    """Register a forward_hook on every block; verify it fires `depth` times."""
    model = SiT_models["SiT-B/2"](input_size=32, in_channels=4)
    model.eval()
    fired: list[int] = []

    def make_hook(idx: int):
        def hook(module, inputs, output):  # noqa: ARG001
            assert getattr(module, "block_idx", None) == idx
            fired.append(idx)

        return hook

    handles = [b.register_forward_hook(make_hook(b.block_idx)) for b in model.blocks]
    try:
        x = torch.randn(1, 4, 32, 32)
        t = torch.rand(1)
        y = torch.randint(0, 1000, (1,))
        with torch.no_grad():
            model(x, t, y)
    finally:
        for h in handles:
            h.remove()

    assert fired == list(range(len(model.blocks)))


def test_sit_with_8_in_channels_for_dc_ae_like_latents() -> None:
    """In-channels swap (dc_ae_1_0 has 32-ch latents) — verify SiT accepts it."""
    model = SiT_models["SiT-B/2"](input_size=8, in_channels=32, num_classes=1000)
    B = 2
    x = torch.randn(B, 32, 8, 8)
    t = torch.rand(B)
    y = torch.randint(0, 1000, (B,))
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == (B, 32, 8, 8)
