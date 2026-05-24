"""Concept-axis registry for the Revelio-grid probes.

Per PLAN §9.1, the proposal originally listed 5 concept axes:
  object  — 1000-way ImageNet class
  scene   — 365-way Places365 (needs external labelling)
  color   — 11-way (needs external labelling)
  texture — DTD 47-way (needs external labelling)
  shape   — ShapeNet 12-way (needs external labelling)

Of these, `object` runs out-of-the-box: its labels are the ImageNet class
indices already saved in HDF5 by `precompute_latents.py` (Phase 1.11) and
by the activation-extraction loop. The others would need an external
dataset or a CLIP-zero-shot pass; they ship as TODO stubs.

To get more out of the dataset we already have, this module also exposes
**WordNet-derived axes**: each axis is a deterministic function of the
ImageNet class index. The mapping is precomputed once at
`data/imagenet_concepts.json` (offline, on a login node with internet),
so probe-side runs at compute-time stay offline:

  animal_binary  — 2-way   (animal vs non-animal; ~398 / 1000 are animal)
  broad_8        — 8-way   (dog / cat / bird / fish / reptile / insect-arthropod
                            / other-mammal / non-animal)
  vehicle_binary — 2-way   (vehicle vs non-vehicle; 67 / 1000)
  food_binary    — 2-way   (food vs non-food; 48 / 1000)
  instrument_binary — 2-way (musical-instrument vs not; 26 / 1000)

Together with `object`, that gives 6 axes runnable on the K=3 production
grid without any new data collection. The proposal's scene / color /
texture / shape axes can land later via a CLIP zero-shot pass.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from torch import Tensor

_CONCEPTS_JSON = (
    Path(__file__).resolve().parents[3] / "data" / "imagenet_concepts.json"
)


def _load_concepts_json() -> dict:
    if not _CONCEPTS_JSON.exists():
        return {}
    return json.loads(_CONCEPTS_JSON.read_text())


_CONCEPTS_DATA = _load_concepts_json()
_PER_CLASS: dict[int, dict] = {
    int(k): v for k, v in _CONCEPTS_DATA.get("per_class", {}).items()
}
BROAD_8_CLASS_NAMES: list[str] = list(_CONCEPTS_DATA.get("broad_8_classes", []))


def _wordnet_label(field: str) -> Callable[[int], int]:
    def fn(class_idx: int) -> int:
        row = _PER_CLASS.get(int(class_idx))
        if row is None:
            raise KeyError(
                f"ImageNet class {class_idx} missing from "
                f"{_CONCEPTS_JSON} — regenerate the mapping."
            )
        return int(row[field])
    return fn


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


def _todo_label(x: int) -> int:
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
    "animal_binary": ConceptAxis(
        name="animal_binary",
        num_classes=2,
        description="WordNet-derived: animal vs non-animal (~398 / 1000 animal).",
        available=bool(_PER_CLASS),
        label_fn=_wordnet_label("animal_binary"),
    ),
    "broad_8": ConceptAxis(
        name="broad_8",
        num_classes=8,
        description=(
            "WordNet-derived coarse 8-way taxonomy: dog / cat / bird / fish / "
            "reptile / insect-arthropod / other-mammal / non-animal."
        ),
        available=bool(_PER_CLASS),
        label_fn=_wordnet_label("broad_8"),
    ),
    "vehicle_binary": ConceptAxis(
        name="vehicle_binary",
        num_classes=2,
        description="WordNet-derived: vehicle vs non-vehicle (67 / 1000 vehicle).",
        available=bool(_PER_CLASS),
        label_fn=_wordnet_label("vehicle_binary"),
    ),
    "food_binary": ConceptAxis(
        name="food_binary",
        num_classes=2,
        description="WordNet-derived: food vs non-food (48 / 1000 food).",
        available=bool(_PER_CLASS),
        label_fn=_wordnet_label("food_binary"),
    ),
    "instrument_binary": ConceptAxis(
        name="instrument_binary",
        num_classes=2,
        description="WordNet-derived: musical-instrument vs not (26 / 1000).",
        available=bool(_PER_CLASS),
        label_fn=_wordnet_label("instrument_binary"),
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
