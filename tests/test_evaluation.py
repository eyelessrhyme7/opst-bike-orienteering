from __future__ import annotations

from pathlib import Path

import csv

from opst.evaluation import run_algorithms_on_instance, write_results_csv
from opst.instances import tiny_metric_instance


def test_run_algorithms_records_opt_ratios() -> None:
    instance = tiny_metric_instance()

    rows = run_algorithms_on_instance(
        instance,
        run_id="unit",
        algorithm_names=("exact_subset_dp", "best_single"),
    )

    assert len(rows) == 2
    exact, single = rows
    assert exact["algorithm"] == "exact_subset_dp"
    assert exact["reward_ratio"] == 1.0
    assert single["opt_reward"] == exact["reward"]
    assert single["reward_ratio"] <= 1.0
    assert single["feasible"] is True


def test_write_results_csv_round_trips_headers(tmp_path: Path) -> None:
    instance = tiny_metric_instance()
    rows = run_algorithms_on_instance(
        instance,
        run_id="unit",
        algorithm_names=("exact_subset_dp",),
    )
    path = tmp_path / "results.csv"

    write_results_csv(rows, path)

    with path.open(encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[0]["run_id"] == "unit"
    assert loaded[0]["algorithm"] == "exact_subset_dp"
