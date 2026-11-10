from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from opst.models import Instance


def service_times_by_index(instance: Instance) -> NDArray[np.int64]:
    """Return service time vector indexed like the travel matrix, with depot service 0."""

    return np.array([0, *(w.service_time_s for w in instance.waypoints)], dtype=np.int64)


def doubled_transformed_cost_s(instance: Instance) -> NDArray[np.int64]:
    """Return 2 * (travel + half endpoint services).

    The half-service transformation can be fractional when service times are odd
    seconds. Doubling keeps DP arithmetic exact and integer-valued.
    """

    services = service_times_by_index(instance)
    travel2 = 2 * instance.travel_time_s.astype(np.int64)
    return travel2 + services[:, None] + services[None, :]


def route_doubled_transformed_cost_s(
    instance: Instance, route_indices: tuple[int, ...]
) -> int:
    """Compute doubled transformed cost for a route."""

    costs = doubled_transformed_cost_s(instance)
    return sum(int(costs[a, b]) for a, b in zip(route_indices, route_indices[1:]))


def assert_symmetric_metric(matrix_s: NDArray[np.int64], *, tolerance_s: int = 0) -> None:
    """Raise if an integer travel-time matrix is not a symmetric metric."""

    matrix = np.asarray(matrix_s)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if np.any(matrix < 0):
        raise ValueError("matrix contains negative entries")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("matrix diagonal must be zero")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("matrix must be symmetric")

    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if np.any(matrix[i, j] > matrix[i, :] + matrix[:, j] + tolerance_s):
                raise ValueError("matrix violates triangle inequality")
