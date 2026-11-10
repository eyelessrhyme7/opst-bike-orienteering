from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from opst.road_matrix import (
    assert_finite_directed_matrix,
    symmetrize_directed_times_s,
    validate_symmetric_travel_matrix,
    write_matrix_bundle,
)


def test_symmetrize_directed_times_averages_and_ceils() -> None:
    directed = np.array(
        [
            [0.0, 10.2, 20.0],
            [12.0, 0.0, 10.1],
            [22.0, 12.0, 0.0],
        ]
    )

    symmetric = symmetrize_directed_times_s(directed)

    np.testing.assert_array_equal(
        symmetric,
        np.array(
            [
                [0, 12, 21],
                [12, 0, 12],
                [21, 12, 0],
            ],
            dtype=np.int64,
        ),
    )
    validate_symmetric_travel_matrix(symmetric)


def test_directed_matrix_validation_rejects_nonfinite_values() -> None:
    directed = np.array([[0.0, np.inf], [1.0, 0.0]])

    with pytest.raises(ValueError, match="non-finite"):
        assert_finite_directed_matrix(directed)


def test_validate_symmetric_matrix_rejects_triangle_violation() -> None:
    matrix = np.array(
        [
            [0, 1, 10],
            [1, 0, 1],
            [10, 1, 0],
        ],
        dtype=np.int64,
    )

    with pytest.raises(ValueError, match="triangle"):
        validate_symmetric_travel_matrix(matrix, tolerance_s=0)


def test_write_matrix_bundle_outputs_files_and_metadata(tmp_path: Path) -> None:
    directed = np.array(
        [
            [0.0, 10.0, 20.0],
            [12.0, 0.0, 10.0],
            [22.0, 12.0, 0.0],
        ]
    )

    bundle = write_matrix_bundle(
        output_dir=tmp_path,
        matrix_id="test_matrix",
        directed_time_s=directed,
        metadata={"source": "unit_test"},
    )

    assert bundle.directed_matrix_path.exists()
    assert bundle.symmetric_matrix_path.exists()
    assert bundle.metadata_path.exists()
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    assert metadata["matrix_id"] == "test_matrix"
    assert metadata["source"] == "unit_test"
    assert metadata["shape"] == [3, 3]
