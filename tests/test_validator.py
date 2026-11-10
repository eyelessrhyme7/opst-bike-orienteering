from __future__ import annotations

from dataclasses import replace

from opst.instances import tiny_metric_instance
from opst.validate import make_solution, validate_solution


def test_validator_accepts_recomputed_solution() -> None:
    instance = tiny_metric_instance()
    solution = make_solution(instance, "test", (0, 1, 2, 0), runtime_ms=0.0)

    result = validate_solution(instance, solution)

    assert result.ok
    assert result.feasible == solution.feasible


def test_validator_rejects_duplicate_station() -> None:
    instance = tiny_metric_instance()
    solution = make_solution(instance, "test", (0, 1, 1, 0), runtime_ms=0.0)

    result = validate_solution(instance, solution)

    assert not result.ok
    assert "route visits a station more than once" in result.errors


def test_validator_rejects_wrong_totals() -> None:
    instance = tiny_metric_instance()
    solution = make_solution(instance, "test", (0, 1, 0), runtime_ms=0.0)
    corrupted = replace(solution, reward=solution.reward + 1)

    result = validate_solution(instance, corrupted)

    assert not result.ok
    assert any("reward field" in error for error in result.errors)


def test_validator_rejects_infeasible_route() -> None:
    instance = tiny_metric_instance()
    solution = make_solution(instance, "test", (0, 1, 2, 3, 4, 5, 0), runtime_ms=0.0)

    result = validate_solution(instance, solution)

    assert not result.ok
    assert "route exceeds budget" in result.errors
