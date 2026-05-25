"""Tests for model-variant metadata and path helpers."""

from pathlib import Path

import h5py
import numpy as np

from diffmechint.utils.model_variants import (
    by_model_root,
    canonical_model_name,
    first_hdf5_d_in,
    model_subdir,
    model_variant_id,
    model_variant_spec,
    parse_layers,
)


def test_model_name_normalization() -> None:
    assert canonical_model_name("sit_l_2") == "SiT-L/2"
    assert canonical_model_name("SiT-L/2") == "SiT-L/2"
    assert model_variant_id("SiT-B/2") == "sit_b_2"
    assert model_variant_id("sit_xl_2") == "sit_xl_2"


def test_model_variant_specs() -> None:
    b = model_variant_spec("sit_b_2")
    assert b.model_name == "SiT-B/2"
    assert b.depth == 12
    assert b.hidden_size == 768
    assert b.patch_size == 2
    assert b.tap_layers == (3, 6, 9)

    large = model_variant_spec("SiT-L/2")
    assert large.variant_id == "sit_l_2"
    assert large.depth == 24
    assert large.hidden_size == 1024
    assert large.patch_size == 2
    assert large.tap_layers == (6, 12, 18)


def test_parse_layers_auto_and_explicit() -> None:
    spec = model_variant_spec("sit_l_2")
    assert parse_layers(["auto"], spec) == [6, 12, 18]
    assert parse_layers(["1", "2"], spec) == [1, 2]
    assert parse_layers([3, 6, 9], spec) == [3, 6, 9]


def test_model_output_paths() -> None:
    root = Path("outputs")
    assert by_model_root(root, "SiT-L/2") == Path("outputs/by_model/sit_l_2")
    assert model_subdir(root, "sit_b_2", "probes") == Path("outputs/by_model/sit_b_2/probes")


def test_first_hdf5_d_in(tmp_path: Path) -> None:
    shard = tmp_path / "acts.h5"
    with h5py.File(shard, "w") as handle:
        handle.create_dataset("activations", data=np.zeros((2, 3, 1024), dtype=np.float16))
    assert first_hdf5_d_in([shard]) == 1024
