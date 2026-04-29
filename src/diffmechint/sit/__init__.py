"""Vendored SiT (Scalable Interpolant Transformer) — see models.py for vendor commit."""

from .models import SiT, SiT_models, SiTBlock
from .transport import create_transport

__all__ = ["SiT", "SiT_models", "SiTBlock", "create_transport"]
