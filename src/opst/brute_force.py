from __future__ import annotations

import itertools
import time

from opst.models import Instance, Solution
from opst.validate import make_solution, validate_solution


def solve_brute_force(instance: Instance) -> Solution:
    """Exhaustively enumerate routes; intended only for tiny tests."""

    start = time.perf_counter()
    n = len(instance.waypoints)
    best_route = (0, 0)
    best_reward = 0
    best_total = 0

    station_indices = range(1, n + 1)
    for size in range(1, n + 1):
        for subset in itertools.combinations(station_indices, size):
            for order in itertools.permutations(subset):
                route = (0, *order, 0)
                solution = make_solution(instance, "brute_force", route, runtime_ms=0.0)
                if not solution.feasible:
                    continue
                key = (solution.reward, -solution.total_time_s, tuple(-x for x in route))
                best_key = (best_reward, -best_total, tuple(-x for x in best_route))
                if key > best_key:
                    best_route = route
                    best_reward = solution.reward
                    best_total = solution.total_time_s

    runtime_ms = (time.perf_counter() - start) * 1000
    solution = make_solution(
        instance,
        "brute_force",
        best_route,
        runtime_ms=runtime_ms,
        status="optimal_by_enumeration",
    )
    validate_solution(instance, solution, strict=True)
    return solution
