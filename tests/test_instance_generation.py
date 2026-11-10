from __future__ import annotations

from pathlib import Path

import numpy as np

from opst.baselines import solve_cheapest_insertion
from opst.exact_dp import solve_exact_subset_dp
from opst.instance_generation import (
    MatrixStation,
    generate_instance_from_matrix,
    instance_to_dict,
    read_instance_json,
    select_spatial_subset,
    write_instance_json,
)
from opst.validate import validate_solution


def fixture_matrix() -> tuple[MatrixStation, tuple[MatrixStation, ...], np.ndarray]:
    depot = MatrixStation(0, "depot", "d", "Depot", 0.0, 0.0, 0, "fixture")
    coords = [(0, 0), (1, 0), (2, 0), (0, 2), (2, 2), (5, 0), (0, 5), (5, 5), (8, 8)]
    candidates = tuple(
        MatrixStation(i, "candidate", f"s{i}", f"Station {i}", float(x), float(y), 10, "fixture")
        for i, (x, y) in enumerate(coords[1:], start=1)
    )
    points = np.array(coords)
    matrix = (np.abs(points[:, None, :] - points[None, :, :]).sum(axis=2) * 60).astype(
        np.int64
    )
    return depot, candidates, matrix


def test_spatial_subset_selection_is_deterministic() -> None:
    _, candidates, _ = fixture_matrix()

    first = select_spatial_subset(candidates, n=5, regime="mixed", seed=10)
    second = select_spatial_subset(candidates, n=5, regime="mixed", seed=10)

    assert first == second
    assert len(first) == 5


def test_generated_instance_is_deterministic_and_solver_valid() -> None:
    depot, candidates, matrix = fixture_matrix()

    first = generate_instance_from_matrix(
        matrix_id="fixture",
        depot=depot,
        candidates=candidates,
        travel_time_s=matrix,
        n=5,
        regime="outlier",
        service_scale=1.0,
        budget_tightness="medium",
        seed=7,
    )
    second = generate_instance_from_matrix(
        matrix_id="fixture",
        depot=depot,
        candidates=candidates,
        travel_time_s=matrix,
        n=5,
        regime="outlier",
        service_scale=1.0,
        budget_tightness="medium",
        seed=7,
    )

    assert instance_to_dict(first) == instance_to_dict(second)
    assert first.budget_s > 0
    exact = solve_exact_subset_dp(first)
    insertion = solve_cheapest_insertion(first)
    assert validate_solution(first, exact).ok
    assert validate_solution(first, insertion).ok


def test_instance_json_round_trip(tmp_path: Path) -> None:
    depot, candidates, matrix = fixture_matrix()
    instance = generate_instance_from_matrix(
        matrix_id="fixture",
        depot=depot,
        candidates=candidates,
        travel_time_s=matrix,
        n=4,
        regime="compact",
        service_scale=0.5,
        budget_tightness="tight",
        seed=3,
    )
    path = tmp_path / "instance.json"

    write_instance_json(instance, path)
    loaded = read_instance_json(path)

    assert instance_to_dict(loaded) == instance_to_dict(instance)
