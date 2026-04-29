"""Sanity checks for the Lightning SiT module — instantiate + training_step + EMA."""

from __future__ import annotations

import torch

from diffmechint.training.sit_module import SiTLightningModule


def test_sit_module_instantiates_b2_with_sd_vae_shape() -> None:
    m = SiTLightningModule(
        model_name="SiT-B/2",
        input_size=32,
        in_channels=4,
        num_classes=10,
        warmup_steps=10,
    )
    n = m.n_parameters()
    assert 100e6 < n < 200e6, f"SiT-B/2 should be ~130M params, got {n / 1e6:.1f}M"


def test_sit_module_training_step_returns_finite_loss() -> None:
    m = SiTLightningModule(
        model_name="SiT-B/2",
        input_size=32,
        in_channels=4,
        num_classes=10,
        warmup_steps=10,
    )
    m.train()
    z = torch.randn(2, 4, 32, 32)
    y = torch.randint(0, 10, (2,))
    loss = m.training_step({"latent": z, "label": y}, batch_idx=0)
    assert torch.isfinite(loss)
    assert loss.ndim == 0  # scalar


def test_ema_updates_after_training_step() -> None:
    m = SiTLightningModule(
        model_name="SiT-B/2",
        input_size=32,
        in_channels=4,
        num_classes=10,
        warmup_steps=10,
        ema_decay=0.99,
    )
    m.train()
    # Manually trigger the on_fit_start hook to construct the EMA.
    m.on_fit_start()
    assert m.ema is not None

    # Capture an EMA shadow weight before any update.
    first_param = next(m.ema.shadow.parameters()).clone()

    # Mutate live model params so EMA update has something to track.
    with torch.no_grad():
        for p in m.model.parameters():
            p.add_(0.1)

    m.on_train_batch_end(outputs=None, batch=None, batch_idx=0)
    second_param = next(m.ema.shadow.parameters())
    assert not torch.allclose(first_param, second_param), "EMA shadow did not update."


def test_fm_ot_default_in_module() -> None:
    """Module must default to Linear/velocity (FM-OT) without explicit transport_cfg."""
    m = SiTLightningModule(
        model_name="SiT-B/2",
        input_size=32,
        in_channels=4,
        num_classes=10,
        warmup_steps=10,
    )
    # Linear interpolant maps to ICPlan in transport.
    from diffmechint.sit.transport import path

    assert isinstance(m.transport.path_sampler, path.ICPlan)
