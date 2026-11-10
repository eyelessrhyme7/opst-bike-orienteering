from __future__ import annotations

from dataclasses import replace

from opst.baselines import solve_service_density_then_route
from opst.instances import random_metric_instance, tiny_metric_instance
from opst.models import Instance
from opst.route_ops import augment_route_feasibly, repair_route_by_removal
from opst.split_heuristic import (
    compact_candidate_pools,
    select_service_knapsack,
    solve_split_compact_grid,
    solve_split_only_grid,
)
from opst.validate import route_totals, validate_solution


def tiny_instance_with_cells() -> Instance:
    instance = tiny_metric_instance()
    waypoints = tuple(
        replace(waypoint, h3_cell="cell_a" if index <= 3 else "cell_b")
        for index, waypoint in enumerate(instance.waypoints, start=1)
    )
    return replace(instance, waypoints=waypoints)


def test_repair_route_removes_until_feasible() -> None:
    instance = tiny_metric_instance()
    infeasible_route = (0, 1, 2, 3, 4, 5, 0)

    repaired = repair_route_by_removal(instance, infeasible_route)

    assert route_totals(instance, repaired)[3] <= instance.budget_s
    assert repaired[0] == repaired[-1] == 0


def test_augmentation_preserves_feasibility() -> None:
    instance = tiny_metric_instance()

    augmented = augment_route_feasibly(instance, (0, 0))

    assert route_totals(instance, augmented)[3] <= instance.budget_s
    assert augmented[0] == augmented[-1] == 0


def test_service_knapsack_selects_best_reward_under_service_budget() -> None:
    instance = tiny_metric_instance()

    selected = select_service_knapsack(instance, (1, 2, 3, 4, 5), service_budget_s=420)

    assert selected == (1, 2)


def test_service_density_then_route_returns_valid_solution() -> None:
    instance = tiny_metric_instance()

    solution = solve_service_density_then_route(instance, {"service_fraction": 0.6})

    assert validate_solution(instance, solution).ok


def test_split_only_grid_returns_valid_solution() -> None:
    instance = random_metric_instance(seed=17, n=8)

    solution = solve_split_only_grid(instance, {"alpha_grid": [0.3, 0.5, 0.7]})

    assert validate_solution(instance, solution).ok


def test_compact_candidate_pools_include_cells() -> None:
    instance = tiny_instance_with_cells()

    pools = compact_candidate_pools(instance)

    assert ("all", (1, 2, 3, 4, 5)) in pools
    assert ("cell:cell_a", (1, 2, 3)) in pools
    assert ("cell:cell_b", (4, 5)) in pools


def test_split_compact_grid_returns_valid_solution() -> None:
    instance = tiny_instance_with_cells()

    solution = solve_split_compact_grid(instance, {"alpha_grid": [0.3, 0.5, 0.7]})

    assert validate_solution(instance, solution).ok
    assert "pool" in solution.parameters
