"""Concept-axis registry for the Revelio-grid probes.

Per PLAN §9.1, the proposal asks for 5 concept axes:
  object  — 1000-way ImageNet class
  scene   — 365-way Places365
  color   — 11-way (red, blue, green, ...)
  texture — DTD 47-way
  shape   — ShapeNet 12-way

Only `object` ships fully usable today: its labels are the ImageNet class
indices already saved in HDF5 by `precompute_latents.py` (Phase 1.11) and
by the activation extraction loop (Phase 5 driver). The other axes need
either an external dataset (Places365, DTD, ImageNet-Sketch+ShapeNet) or
a CLIP-zero-shot labelling pass; they ship as TODO stubs that an
implementer can flesh out without changing the probe code itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConceptAxis:
    """A concept whose probe accuracy we measure.

    `label_fn` maps `(image_label_or_index)` → `int` in `[0, num_classes)`.
    For axes whose labels are not in HDF5, the probe pipeline calls
    `label_fn` on a side channel (e.g. CLIP-zero-shot output).
    """

    name: str
    num_classes: int
    description: str
    available: bool = True
    label_fn: Callable[[int], int] = field(default=lambda x: int(x))


def _identity_label(x: int) -> int:
    return int(x)


def _todo_label(x: int) -> int:  # noqa: ARG001
    raise NotImplementedError(
        "This concept axis requires an external label source. See concepts.py "
        "docstring; expose a `label_fn` that maps your sample-id / CLIP-zero-shot "
        "label → int in [0, num_classes)."
    )


CONCEPTS: dict[str, ConceptAxis] = {
    "object": ConceptAxis(
        name="object",
        num_classes=1000,
        description="ImageNet-1K class index (label is already in HDF5).",
        available=True,
        label_fn=_identity_label,
    ),
    "scene": ConceptAxis(
        name="scene",
        num_classes=365,
        description="Places365 — TODO: zero-shot CLIP labelling on each image.",
        available=False,
        label_fn=_todo_label,
    ),
    "color": ConceptAxis(
        name="color",
        num_classes=11,
        description="11-way color (red, blue, green, ...) — TODO: WordNet-based.",
        available=False,
        label_fn=_todo_label,
    ),
    "texture": ConceptAxis(
        name="texture",
        num_classes=47,
        description="DTD 47-way — TODO: requires DTD test images & labels.",
        available=False,
        label_fn=_todo_label,
    ),
    "shape": ConceptAxis(
        name="shape",
        num_classes=12,
        description="ShapeNet 12-way via ImageNet-Sketch — TODO.",
        available=False,
        label_fn=_todo_label,
    ),
}


def get_concept(name: str) -> ConceptAxis:
    if name not in CONCEPTS:
        raise KeyError(f"Unknown concept {name!r}. Known: {sorted(CONCEPTS)}.")
    return CONCEPTS[name]


def available_concepts() -> list[str]:
    return [n for n, c in CONCEPTS.items() if c.available]


def pool_tokens(activations: Tensor, mode: str = "tokens") -> Tensor:
    """Reduce per-image (T, D) tensors to probe-friendly shape.

    Modes:
      tokens  — flatten (N, T, D) → (N*T, D); probe sees each spatial token
                as its own sample (Revelio convention). Caller must repeat
                labels by `T` to match.
      mean    — mean-pool over tokens: (N, T, D) → (N, D).
      cls     — error: SiT has no CLS token; use 'tokens' or 'mean'.
    """
    if activations.ndim != 3:
        raise ValueError(f"pool_tokens expects (N, T, D); got {tuple(activations.shape)}.")
    if mode == "tokens":
        return activations.reshape(-1, activations.shape[-1])
    if mode == "mean":
        return activations.mean(dim=1)
    if mode == "cls":
        raise ValueError("SiT has no CLS token; use 'tokens' or 'mean'.")
    raise ValueError(f"Unknown pool mode {mode!r}.")


def expand_labels_for_tokens(labels: Tensor, n_tokens: int) -> Tensor:
    """When using `pool_tokens(..., mode='tokens')`, repeat each label `T` times."""
    return labels.repeat_interleave(int(n_tokens))
