from __future__ import annotations

from opst.baselines import solve_best_single, solve_cheapest_insertion
from opst.instances import tiny_metric_instance
from opst.route_ops import insertion_delta_s, two_opt
from opst.validate import make_solution, route_totals, validate_solution


def test_insertion_delta_matches_recomputed_total_change() -> None:
    instance = tiny_metric_instance()
    route = (0, 1, 0)
    before = route_totals(instance, route)[3]
    delta = insertion_delta_s(instance, route, node=2, position=2)
    after = route_totals(instance, (0, 1, 2, 0))[3]

    assert before + delta == after


def test_two_opt_never_worsens_travel_time() -> None:
    instance = tiny_metric_instance()
    route = (0, 2, 5, 1, 4, 0)

    improved = two_opt(instance, route)

    assert route_totals(instance, improved)[1] <= route_totals(instance, route)[1]
    assert set(improved[1:-1]) == set(route[1:-1])


def test_cheapest_insertion_baseline_returns_valid_solution() -> None:
    instance = tiny_metric_instance()

    solution = solve_cheapest_insertion(instance)

    assert validate_solution(instance, solution).ok


def test_best_single_returns_best_individually_feasible_station() -> None:
    instance = tiny_metric_instance()

    solution = solve_best_single(instance)

    assert validate_solution(instance, solution).ok
    feasible_singles = [
        make_solution(instance, "single", (0, i, 0), runtime_ms=0.0)
        for i in range(1, len(instance.waypoints) + 1)
    ]
    feasible_singles = [candidate for candidate in feasible_singles if candidate.feasible]
    assert solution.reward == max(candidate.reward for candidate in feasible_singles)
