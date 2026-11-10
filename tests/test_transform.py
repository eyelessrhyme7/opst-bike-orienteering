from __future__ import annotations

from opst.instances import random_metric_instance, tiny_metric_instance
from opst.transform import assert_symmetric_metric, route_doubled_transformed_cost_s
from opst.validate import route_totals


def test_doubled_service_transform_equals_travel_plus_service() -> None:
    instance = tiny_metric_instance()
    route = (0, 1, 4, 5, 0)
    total = route_totals(instance, route)[3]

    assert route_doubled_transformed_cost_s(instance, route) == 2 * total


def test_random_fixture_travel_matrix_is_metric() -> None:
    instance = random_metric_instance(seed=42, n=8)

    assert_symmetric_metric(instance.travel_time_s)
