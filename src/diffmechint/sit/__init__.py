"""Vendored SiT (Scalable Interpolant Transformer) — see models.py for vendor commit."""

from .factory import build_sit_model, list_ema_checkpoints
from .models import SiT, SiT_models, SiTBlock
from .transport import create_transport

__all__ = [
    "SiT",
    "SiT_models",
    "SiTBlock",
    "build_sit_model",
    "create_transport",
    "list_ema_checkpoints",
]
