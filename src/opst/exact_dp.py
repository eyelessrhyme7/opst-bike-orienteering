from __future__ import annotations

import time

import numpy as np

from opst.models import Instance, Solution
from opst.transform import doubled_transformed_cost_s
from opst.validate import make_solution, validate_solution


def _reward_for_mask(instance: Instance, mask: int) -> int:
    return sum(w.reward for bit, w in enumerate(instance.waypoints) if mask & (1 << bit))


def _reconstruct_route(parent: np.ndarray, mask: int, end_bit: int) -> tuple[int, ...]:
    bits_reversed: list[int] = []
    current_mask = mask
    current = end_bit
    while current != -1:
        bits_reversed.append(current)
        previous = int(parent[current_mask, current])
        current_mask ^= 1 << current
        current = previous
    station_indices = tuple(bit + 1 for bit in reversed(bits_reversed))
    return (0, *station_indices, 0)


def solve_exact_subset_dp(instance: Instance, config: dict[str, object] | None = None) -> Solution:
    """Solve small OP-ST instances exactly by Held-Karp style subset DP."""

    start = time.perf_counter()
    n = len(instance.waypoints)
    size = 1 << n
    inf = np.iinfo(np.int64).max // 4
    costs = doubled_transformed_cost_s(instance)
    dp = np.full((size, n), inf, dtype=np.int64)
    parent = np.full((size, n), -1, dtype=np.int64)

    for bit in range(n):
        dp[1 << bit, bit] = costs[0, bit + 1]

    for mask in range(size):
        for end in range(n):
            current_cost = int(dp[mask, end])
            if current_cost >= inf:
                continue
            remaining = ((1 << n) - 1) ^ mask
            next_bits = remaining
            while next_bits:
                lowbit = next_bits & -next_bits
                nxt = lowbit.bit_length() - 1
                next_mask = mask | lowbit
                candidate = current_cost + int(costs[end + 1, nxt + 1])
                if candidate < dp[next_mask, nxt]:
                    dp[next_mask, nxt] = candidate
                    parent[next_mask, nxt] = end
                next_bits ^= lowbit

    best_route = (0, 0)
    best_reward = 0
    best_total2 = 0
    budget2 = 2 * instance.budget_s

    for mask in range(1, size):
        reward = _reward_for_mask(instance, mask)
        if reward < best_reward:
            continue
        for end in range(n):
            path_cost = int(dp[mask, end])
            if path_cost >= inf:
                continue
            total2 = path_cost + int(costs[end + 1, 0])
            if total2 > budget2:
                continue
            route = _reconstruct_route(parent, mask, end)
            candidate_key = (reward, -total2, tuple(-x for x in route))
            best_key = (best_reward, -best_total2, tuple(-x for x in best_route))
            if candidate_key > best_key:
                best_route = route
                best_reward = reward
                best_total2 = total2

    runtime_ms = (time.perf_counter() - start) * 1000
    solution = make_solution(
        instance,
        algorithm="exact_subset_dp",
        route_indices=best_route,
        runtime_ms=runtime_ms,
        parameters={} if config is None else dict(config),
        status="optimal",
    )
    validate_solution(instance, solution, strict=True)
    return solution
