"""Adapter registry — name → factory dispatch consumed by Hydra `_target_`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .base import TokenizerAdapter

_REGISTRY: dict[str, Callable[..., "TokenizerAdapter"]] = {}


def register(name: str) -> Callable[[type], type]:
    """Class decorator: register an adapter under `name`."""

    def _wrap(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"Tokenizer adapter '{name}' already registered.")
        _REGISTRY[name] = cls
        return cls

    return _wrap


def build(name: str, **kwargs) -> "TokenizerAdapter":
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown tokenizer '{name}'. Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def list_registered() -> list[str]:
    return sorted(_REGISTRY.keys())
