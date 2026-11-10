from __future__ import annotations

from dataclasses import replace
import time

from opst.models import Instance, Solution
from opst.route_ops import (
    augment_route_feasibly,
    repair_route_by_removal,
    route_selected_by_cheapest_insertion,
)
from opst.validate import make_solution, validate_solution

DEFAULT_ALPHA_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def _alpha_grid(config: dict[str, object] | None) -> tuple[float, ...]:
    if config is None or "alpha_grid" not in config:
        return DEFAULT_ALPHA_GRID
    return tuple(float(alpha) for alpha in config["alpha_grid"])  # type: ignore[index]


def select_service_knapsack(
    instance: Instance,
    candidate_indices: tuple[int, ...],
    service_budget_s: int,
) -> tuple[int, ...]:
    """Exact 0/1 service-time knapsack over a candidate pool."""

    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for node in sorted(candidate_indices):
        waypoint = instance.waypoints[node - 1]
        next_states = dict(states)
        for used_service, (reward, selected) in states.items():
            new_service = used_service + waypoint.service_time_s
            if new_service > service_budget_s:
                continue
            new_selected = (*selected, node)
            new_value = (reward + waypoint.reward, new_selected)
            old_value = next_states.get(new_service)
            if old_value is None or (
                new_value[0],
                tuple(-idx for idx in new_value[1]),
            ) > (
                old_value[0],
                tuple(-idx for idx in old_value[1]),
            ):
                next_states[new_service] = new_value
        states = next_states

    best_service, (best_reward, best_selected) = max(
        states.items(),
        key=lambda item: (item[1][0], -item[0], tuple(-idx for idx in item[1][1])),
    )
    _ = best_service, best_reward
    return best_selected


def _solve_for_pool_and_alpha(
    instance: Instance,
    candidate_indices: tuple[int, ...],
    alpha: float,
    algorithm: str,
    pool_name: str,
) -> Solution:
    service_budget_s = int(alpha * instance.budget_s)
    selected = select_service_knapsack(instance, candidate_indices, service_budget_s)
    route = route_selected_by_cheapest_insertion(instance, selected)
    route = repair_route_by_removal(instance, route)
    route = augment_route_feasibly(instance, route, candidate_indices)
    solution = make_solution(
        instance,
        algorithm,
        route,
        runtime_ms=0.0,
        parameters={
            "alpha": alpha,
            "service_budget_s": service_budget_s,
            "pool": pool_name,
            "candidate_count": len(candidate_indices),
        },
        status="ok",
    )
    validate_solution(instance, solution, strict=True)
    return solution


def _solution_key(solution: Solution) -> tuple[int, int, tuple[int, ...]]:
    return (
        solution.reward,
        -solution.total_time_s,
        tuple(-index for index in solution.route_indices),
    )


def solve_split_only_grid(
    instance: Instance, config: dict[str, object] | None = None
) -> Solution:
    """Search a deterministic service-budget fraction grid over all stations."""

    start = time.perf_counter()
    all_indices = tuple(range(1, len(instance.waypoints) + 1))
    best = make_solution(instance, "split_only_grid", (0, 0), runtime_ms=0.0, status="ok")
    for alpha in _alpha_grid(config):
        candidate = _solve_for_pool_and_alpha(
            instance, all_indices, alpha, "split_only_grid", "all"
        )
        if _solution_key(candidate) > _solution_key(best):
            best = candidate
    final = replace(best, runtime_ms=(time.perf_counter() - start) * 1000)
    validate_solution(instance, final, strict=True)
    return final


def _h3_disk(cell: str) -> set[str]:
    try:
        import h3  # type: ignore[import-not-found]

        return set(h3.grid_disk(cell, 1))
    except Exception:
        return {cell}


def compact_candidate_pools(instance: Instance) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Build all-station, occupied-cell, and one-ring compact pools."""

    all_indices = tuple(range(1, len(instance.waypoints) + 1))
    pools: list[tuple[str, tuple[int, ...]]] = [("all", all_indices)]
    by_cell: dict[str, list[int]] = {}
    for index, waypoint in enumerate(instance.waypoints, start=1):
        if waypoint.h3_cell:
            by_cell.setdefault(waypoint.h3_cell, []).append(index)

    for cell in sorted(by_cell):
        pools.append((f"cell:{cell}", tuple(sorted(by_cell[cell]))))
        disk = _h3_disk(cell)
        ring_indices = tuple(
            sorted(
                index
                for index, waypoint in enumerate(instance.waypoints, start=1)
                if waypoint.h3_cell in disk
            )
        )
        pools.append((f"cell_plus_ring:{cell}", ring_indices))

    deduped: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for name, indices in pools:
        if not indices or indices in seen:
            continue
        seen.add(indices)
        deduped.append((name, indices))
    return tuple(deduped)


def solve_split_compact_grid(
    instance: Instance, config: dict[str, object] | None = None
) -> Solution:
    """Search service split and compact candidate pools, retaining the best feasible route."""

    start = time.perf_counter()
    best = make_solution(instance, "split_compact_grid", (0, 0), runtime_ms=0.0, status="ok")
    for pool_name, candidate_indices in compact_candidate_pools(instance):
        for alpha in _alpha_grid(config):
            candidate = _solve_for_pool_and_alpha(
                instance, candidate_indices, alpha, "split_compact_grid", pool_name
            )
            if _solution_key(candidate) > _solution_key(best):
                best = candidate
    final = replace(best, runtime_ms=(time.perf_counter() - start) * 1000)
    validate_solution(instance, final, strict=True)
    return final
