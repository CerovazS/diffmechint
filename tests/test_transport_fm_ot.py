"""FM-OT (Linear interpolant + velocity prediction) sanity checks."""

from __future__ import annotations

import torch

from diffmechint.sit import create_transport
from diffmechint.sit.transport import path
from diffmechint.sit.transport.transport import ModelType, WeightType


def test_fm_ot_factory_returns_linear_velocity_no_weight() -> None:
    """Linear path-type maps to ICPlan; velocity prediction; uniform weighting."""
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=0,
        sample_eps=0,
    )
    assert isinstance(transport.path_sampler, path.ICPlan)
    assert transport.model_type == ModelType.VELOCITY
    assert transport.loss_type == WeightType.NONE
    # For velocity + Linear, train/sample eps stay at 0 (factory clamps them).
    assert transport.train_eps == 0
    assert transport.sample_eps == 0


def test_linear_path_marginal_shapes() -> None:
    """Linear interpolant defines alpha_t = t, sigma_t = 1 - t.
    Verify the path sampler returns (t, x0, x1) with consistent shapes."""
    transport = create_transport(path_type="Linear", prediction="velocity")
    B, C, H, W = 2, 4, 8, 8
    x_1 = torch.randn(B, C, H, W)
    t, x_0, x_1_out = transport.sample(x_1)
    assert t.shape == (B,)
    assert x_0.shape == x_1.shape == x_1_out.shape
    assert x_0.dtype == x_1.dtype


def test_linear_path_alpha_sigma_at_t() -> None:
    """ICPlan (Linear) coefficients: alpha_t = t, sigma_t = 1 - t."""
    transport = create_transport(path_type="Linear", prediction="velocity")
    p = transport.path_sampler
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    alpha_t, _ = p.compute_alpha_t(t)
    sigma_t, _ = p.compute_sigma_t(t)
    torch.testing.assert_close(alpha_t, t)
    torch.testing.assert_close(sigma_t, 1.0 - t)


def test_training_losses_runs_on_dummy_model() -> None:
    """End-to-end: a tiny dummy velocity-prediction model produces a finite loss."""

    class DummyVelocityNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Conv2d(4, 4, kernel_size=1)

        def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
            return self.proj(x)

    transport = create_transport(path_type="Linear", prediction="velocity")
    model = DummyVelocityNet()
    x = torch.randn(2, 4, 8, 8)
    out = transport.training_losses(model, x, dict())
    assert "loss" in out
    assert torch.isfinite(out["loss"]).all()
