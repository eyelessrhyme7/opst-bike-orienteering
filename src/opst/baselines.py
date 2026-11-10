from __future__ import annotations

import time

from opst.models import Instance, Solution
from opst.route_ops import (
    augment_route_feasibly,
    cheapest_insertion_route,
    repair_route_by_removal,
    route_selected_by_cheapest_insertion,
)
from opst.validate import make_solution, validate_solution


def solve_best_single(instance: Instance) -> Solution:
    """Return the best individually feasible station, or the empty route."""

    start = time.perf_counter()
    best_route = (0, 0)
    best_key = (0, 0, 0)
    for index, waypoint in enumerate(instance.waypoints, start=1):
        route = (0, index, 0)
        candidate = make_solution(instance, "best_single", route, runtime_ms=0.0)
        if candidate.feasible:
            key = (candidate.reward, -candidate.total_time_s, -index)
            if key > best_key:
                best_key = key
                best_route = route

    solution = make_solution(
        instance,
        "best_single",
        best_route,
        runtime_ms=(time.perf_counter() - start) * 1000,
        status="ok",
    )
    validate_solution(instance, solution, strict=True)
    return solution


def solve_cheapest_insertion(instance: Instance) -> Solution:
    """Travel-aware reward per marginal-time insertion baseline."""

    start = time.perf_counter()
    route = cheapest_insertion_route(instance)
    solution = make_solution(
        instance,
        "cheapest_insertion",
        route,
        runtime_ms=(time.perf_counter() - start) * 1000,
        status="ok",
    )
    validate_solution(instance, solution, strict=True)
    return solution


def solve_service_density_then_route(
    instance: Instance, config: dict[str, object] | None = None
) -> Solution:
    """Select by reward/service density, then route, repair, and augment."""

    start = time.perf_counter()
    config = {} if config is None else dict(config)
    service_fraction = float(config.get("service_fraction", 0.7))
    service_budget_s = int(service_fraction * instance.budget_s)

    selected: list[int] = []
    used_service = 0
    candidates = []
    for index, waypoint in enumerate(instance.waypoints, start=1):
        if waypoint.service_time_s == 0:
            density = float("inf") if waypoint.reward > 0 else 0.0
        else:
            density = waypoint.reward / waypoint.service_time_s
        candidates.append((density, waypoint.reward, -waypoint.service_time_s, -index, index))

    for _, _, _, _, index in sorted(candidates, reverse=True):
        service = instance.waypoints[index - 1].service_time_s
        if used_service + service <= service_budget_s:
            selected.append(index)
            used_service += service

    route = route_selected_by_cheapest_insertion(instance, tuple(selected))
    route = repair_route_by_removal(instance, route)
    route = augment_route_feasibly(instance, route)
    solution = make_solution(
        instance,
        "service_density_then_route",
        route,
        runtime_ms=(time.perf_counter() - start) * 1000,
        parameters={"service_fraction": service_fraction},
        status="ok",
    )
    validate_solution(instance, solution, strict=True)
    return solution
