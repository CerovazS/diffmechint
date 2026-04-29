"""Activation extraction primitives — hooks, timestep router, buffer."""

from .activation_buffer import ActivationBuffer
from .activation_taps import ResidualStreamTap, default_tap_layers
from .timestep_router import (
    DEFAULT_T_BINS_DDPM,
    DEFAULT_T_BINS_SIT,
    DEFAULT_T_TOL,
    bin_revelio,
    current_t,
    set_t,
    timestep_context,
)

__all__ = [
    "ActivationBuffer",
    "ResidualStreamTap",
    "default_tap_layers",
    "DEFAULT_T_BINS_SIT",
    "DEFAULT_T_BINS_DDPM",
    "DEFAULT_T_TOL",
    "bin_revelio",
    "current_t",
    "set_t",
    "timestep_context",
]
