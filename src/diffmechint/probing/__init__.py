"""Linear-probe pipeline (Revelio-grid)."""

from .concepts import (
    CONCEPTS,
    ConceptAxis,
    available_concepts,
    expand_labels_for_tokens,
    get_concept,
    pool_tokens,
)
from .revelio_grid import (
    CellResult,
    GridResult,
    evaluate_grid,
    probe_one_cell,
    train_probe,
    write_grid_result,
)

__all__ = [
    "CONCEPTS",
    "ConceptAxis",
    "available_concepts",
    "expand_labels_for_tokens",
    "get_concept",
    "pool_tokens",
    "CellResult",
    "GridResult",
    "evaluate_grid",
    "probe_one_cell",
    "train_probe",
    "write_grid_result",
]
