from __future__ import annotations

import numpy as np

from opst.models import Instance, Waypoint
from opst.transform import assert_symmetric_metric


def tiny_metric_instance() -> Instance:
    """Hand-built 5-station metric instance for smoke tests and route plotting."""

    coordinates = [
        (59.9139, 10.7522),  # depot
        (59.9149, 10.7527),
        (59.9160, 10.7509),
        (59.9128, 10.7485),
        (59.9118, 10.7547),
        (59.9151, 10.7562),
    ]
    # Manhattan travel in an abstract grid, scaled to seconds; this is metric by construction.
    grid = np.array(
        [
            [0, 1, 3, 4, 3, 4],
            [1, 0, 2, 5, 4, 3],
            [3, 2, 0, 5, 6, 5],
            [4, 5, 5, 0, 3, 6],
            [3, 4, 6, 3, 0, 3],
            [4, 3, 5, 6, 3, 0],
        ],
        dtype=np.int64,
    )
    travel_time_s = grid * 60
    assert_symmetric_metric(travel_time_s)

    waypoints = tuple(
        Waypoint(
            waypoint_id=f"s{i}",
            lat=coordinates[i][0],
            lon=coordinates[i][1],
            service_time_s=service,
            reward=reward,
            h3_cell=None,
            metadata={"fixture": "tiny_metric"},
        )
        for i, (service, reward) in enumerate(
            [
                (180, 8),
                (240, 9),
                (120, 4),
                (300, 11),
                (180, 7),
            ],
            start=1,
        )
    )
    return Instance(
        instance_id="tiny_metric_5",
        depot_id="depot_oslo_demo",
        depot_lat=coordinates[0][0],
        depot_lon=coordinates[0][1],
        waypoints=waypoints,
        travel_time_s=travel_time_s,
        budget_s=900,
        metadata={"source": "hand_built_fixture", "time_unit": "seconds"},
    )


def random_metric_instance(seed: int, n: int, budget_s: int | None = None) -> Instance:
    """Generate a tiny random Manhattan-metric instance for solver cross-checks."""

    rng = np.random.default_rng(seed)
    points = rng.integers(0, 20, size=(n + 1, 2))
    distances = np.abs(points[:, None, :] - points[None, :, :]).sum(axis=2)
    travel_time_s = (distances * 30).astype(np.int64)
    assert_symmetric_metric(travel_time_s)

    services = rng.integers(30, 181, size=n)
    rewards = rng.integers(1, 16, size=n)
    waypoints = tuple(
        Waypoint(
            waypoint_id=f"r{seed}_{i}",
            lat=float(points[i, 0]),
            lon=float(points[i, 1]),
            service_time_s=int(services[i - 1]),
            reward=int(rewards[i - 1]),
            metadata={"seed": seed},
        )
        for i in range(1, n + 1)
    )
    if budget_s is None:
        # Keep some, but usually not all, stations feasible.
        nearest_round_trip = min(
            int(2 * travel_time_s[0, i] + waypoints[i - 1].service_time_s)
            for i in range(1, n + 1)
        )
        budget_s = max(nearest_round_trip, int(np.percentile(travel_time_s, 65)) * n // 2)
    return Instance(
        instance_id=f"random_metric_seed{seed}_n{n}",
        depot_id="depot",
        depot_lat=float(points[0, 0]),
        depot_lon=float(points[0, 1]),
        waypoints=waypoints,
        travel_time_s=travel_time_s,
        budget_s=int(budget_s),
        metadata={"source": "seeded_random_manhattan", "seed": seed},
    )
