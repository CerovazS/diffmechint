"""Helpers for stable SiT model variant metadata and output roots."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_FAMILY_META = {
    "s": {"depth": 12, "hidden_size": 384},
    "b": {"depth": 12, "hidden_size": 768},
    "l": {"depth": 24, "hidden_size": 1024},
    "xl": {"depth": 28, "hidden_size": 1152},
}
_FAMILY_CANONICAL = {"s": "S", "b": "B", "l": "L", "xl": "XL"}
_PATCH_SIZES = ("1", "2", "4", "8")
_MODEL_NAMES = {
    f"SiT-{_FAMILY_CANONICAL[family]}/{patch}"
    for family in _FAMILY_META
    for patch in _PATCH_SIZES
}


def _default_tap_layers(depth: int) -> tuple[int, int, int]:
    return (
        max(1, depth // 4),
        depth // 2,
        max(depth // 2 + 1, (3 * depth) // 4),
    )


@dataclass(frozen=True)
class ModelVariantSpec:
    """Stable metadata needed by downstream extraction and analysis jobs."""

    model_name: str
    variant_id: str
    depth: int
    hidden_size: int
    patch_size: int
    tap_layers: tuple[int, ...]


def canonical_model_name(value: str) -> str:
    """Return the SiT registry key for a Hydra name, variant id, or model name."""
    raw = value.strip()
    if raw in _MODEL_NAMES:
        return raw

    key = raw.lower().replace("-", "_").replace("/", "_")
    match = re.fullmatch(r"sit_?(xl|s|b|l)_?([1248])", key)
    if not match:
        choices = ", ".join(sorted(_MODEL_NAMES))
        raise ValueError(f"unknown SiT model variant {value!r}; choose from {choices}")

    family, patch = match.groups()
    model_name = f"SiT-{_FAMILY_CANONICAL[family]}/{patch}"
    if model_name not in _MODEL_NAMES:
        choices = ", ".join(sorted(_MODEL_NAMES))
        raise ValueError(f"unknown SiT model variant {value!r}; choose from {choices}")
    return model_name


def model_variant_id(value: str) -> str:
    """Return a filesystem-safe id such as `sit_l_2`."""
    model_name = canonical_model_name(value)
    family, patch = model_name.removeprefix("SiT-").split("/")
    return f"sit_{family.lower()}_{patch}"


def model_variant_spec(value: str) -> ModelVariantSpec:
    """Return metadata for a SiT variant without instantiating the full model."""
    model_name = canonical_model_name(value)
    family, patch = model_name.removeprefix("SiT-").split("/")
    family_key = family.lower()
    meta = _FAMILY_META[family_key]
    depth = int(meta["depth"])
    return ModelVariantSpec(
        model_name=model_name,
        variant_id=model_variant_id(model_name),
        depth=depth,
        hidden_size=int(meta["hidden_size"]),
        patch_size=int(patch),
        tap_layers=_default_tap_layers(depth),
    )


def parse_layers(values: Sequence[str | int] | None, spec: ModelVariantSpec) -> list[int]:
    """Resolve CLI layer values, accepting `auto` as a model-aware shorthand."""
    if values is None:
        return list(spec.tap_layers)
    if len(values) == 1 and str(values[0]).lower() == "auto":
        return list(spec.tap_layers)
    layers: list[int] = []
    for value in values:
        raw = str(value)
        if raw.lower() == "auto":
            raise ValueError("`auto` cannot be combined with explicit layer indices")
        layers.append(int(raw))
    return layers


def by_model_root(base: Path | str, model: str) -> Path:
    """Return `<base>/by_model/<variant_id>` for new model-namespaced outputs."""
    return Path(base) / "by_model" / model_variant_id(model)


def model_subdir(base: Path | str, model: str, *parts: str) -> Path:
    """Return a model-namespaced subdirectory under `base`."""
    return by_model_root(base, model).joinpath(*parts)


def first_hdf5_d_in(shards: Iterable[Path]) -> int:
    """Infer residual stream width from the first HDF5 activation shard."""
    import h5py

    for shard in shards:
        with h5py.File(shard, "r") as handle:
            shape = handle["activations"].shape
        if len(shape) != 3:
            raise ValueError(f"expected activations shape (N, T, D), got {shape} in {shard}")
        return int(shape[-1])
    raise ValueError("cannot infer d_in from an empty shard list")
