from __future__ import annotations

from opst.reporting import paired_ablations, summarize_by_algorithm


def result_row(instance_id: str, algorithm: str, ratio: str) -> dict[str, str]:
    return {
        "run_id": "unit",
        "instance_id": instance_id,
        "seed": "1",
        "n": "8",
        "regime": "compact",
        "service_scale": "1.0",
        "budget_tightness": "tight",
        "algorithm": algorithm,
        "reward": "1",
        "opt_reward": "1",
        "reward_ratio": ratio,
        "approximation_factor": "1",
        "travel_time_s": "1",
        "service_time_s": "1",
        "total_time_s": "2",
        "budget_s": "3",
        "feasible": "True",
        "selected_count": "1",
        "runtime_ms": "2.5",
        "status": "ok",
        "error": "",
    }


def test_summarize_by_algorithm_computes_reward_ratio_stats() -> None:
    rows = [
        result_row("a", "best_single", "0.5"),
        result_row("b", "best_single", "1.0"),
        result_row("a", "exact_subset_dp", "1.0"),
    ]

    summaries = summarize_by_algorithm(rows)
    best_single = next(row for row in summaries if row["algorithm"] == "best_single")

    assert best_single["row_count"] == 2
    assert best_single["median_reward_ratio"] == 0.75
    assert best_single["min_reward_ratio"] == 0.5


def test_paired_ablations_compute_per_instance_deltas() -> None:
    rows = [
        result_row("a", "service_density_then_route", "0.4"),
        result_row("a", "split_only_grid", "0.7"),
        result_row("a", "split_compact_grid", "0.9"),
        result_row("a", "cheapest_insertion", "0.8"),
    ]

    ablations = paired_ablations(rows)

    compact = next(
        row for row in ablations if row["comparison"] == "split_compact_vs_split_only"
    )
    assert abs(compact["delta_reward_ratio"] - 0.2) < 1e-9
