from __future__ import annotations

from dataclasses import replace

from opst.brute_force import solve_brute_force
from opst.exact_dp import solve_exact_subset_dp
from opst.instances import random_metric_instance, tiny_metric_instance
from opst.validate import validate_solution


def assert_matches_brute_force(seed: int, n: int) -> None:
    instance = random_metric_instance(seed=seed, n=n)
    exact = solve_exact_subset_dp(instance)
    brute = solve_brute_force(instance)

    assert validate_solution(instance, exact).ok
    assert exact.reward == brute.reward
    assert exact.total_time_s == brute.total_time_s


def test_exact_dp_matches_brute_force_on_tiny_fixture() -> None:
    instance = tiny_metric_instance()

    exact = solve_exact_subset_dp(instance)
    brute = solve_brute_force(instance)

    assert exact.status == "optimal"
    assert exact.reward == brute.reward
    assert exact.total_time_s == brute.total_time_s


def test_exact_dp_matches_brute_force_random_metrics_up_to_8_nodes() -> None:
    for seed, n in [(1, 1), (2, 3), (3, 5), (4, 6), (5, 8)]:
        assert_matches_brute_force(seed, n)


def test_exact_dp_empty_route_when_no_station_fits() -> None:
    instance = replace(tiny_metric_instance(), budget_s=1)

    exact = solve_exact_subset_dp(instance)

    assert exact.route_indices == (0, 0)
    assert exact.reward == 0
    assert exact.feasible


def test_exact_dp_takes_all_when_budget_is_large() -> None:
    instance = replace(tiny_metric_instance(), budget_s=10_000)

    exact = solve_exact_subset_dp(instance)

    assert set(exact.selected_indices) == set(range(1, len(instance.waypoints) + 1))
    assert exact.reward == sum(w.reward for w in instance.waypoints)
