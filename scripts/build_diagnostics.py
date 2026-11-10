from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opst.instance_generation import read_instance_json
from opst.models import Instance
from opst.transform import doubled_transformed_cost_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build diagnostic tables from exact-suite artifacts.")
    parser.add_argument("--manifest", type=Path, default=Path("data/instances/exact_suite/manifest.csv"))
    parser.add_argument("--results", type=Path, default=Path("results/raw/exact_suite_results.csv"))
    parser.add_argument("--solutions", type=Path, default=Path("results/raw/exact_suite_solutions.jsonl"))
    parser.add_argument(
        "--split-trace", type=Path, default=Path("results/raw/exact_suite_split_trace.parquet")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def float_value(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def exact_match_rates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(row)
    output = []
    for algorithm, alg_rows in sorted(grouped.items()):
        ratios = [float(row["reward_ratio"]) for row in alg_rows if row["reward_ratio"]]
        exact_matches = sum(abs(ratio - 1.0) <= 1e-12 for ratio in ratios)
        output.append(
            {
                "algorithm": algorithm,
                "row_count": len(alg_rows),
                "exact_reward_match_count": exact_matches,
                "exact_reward_match_rate": exact_matches / len(ratios) if ratios else "",
            }
        )
    return output


def breakdown(rows: list[dict[str, str]], field: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        ratio = float_value(row["reward_ratio"])
        if ratio is not None:
            grouped[(row["algorithm"], row[field])].append(ratio)
    output = []
    for (algorithm, value), ratios in sorted(grouped.items()):
        ordered = sorted(ratios)
        p5 = ordered[max(0, int(0.05 * (len(ordered) - 1)))]
        output.append(
            {
                "breakdown_field": field,
                "breakdown_value": value,
                "algorithm": algorithm,
                "count": len(ratios),
                "median_reward_ratio": float(np.median(ratios)),
                "mean_reward_ratio": float(np.mean(ratios)),
                "p5_reward_ratio": p5,
                "min_reward_ratio": min(ratios),
            }
        )
    return output


def runtime_by_n(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["runtime_ms"]:
            grouped[(row["algorithm"], row["n"])].append(float(row["runtime_ms"]))
    return [
        {
            "algorithm": algorithm,
            "n": n,
            "count": len(values),
            "median_runtime_ms": float(np.median(values)),
            "mean_runtime_ms": float(np.mean(values)),
            "max_runtime_ms": max(values),
        }
        for (algorithm, n), values in sorted(grouped.items())
    ]


def paired_improvements(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_instance: dict[str, dict[str, float]] = defaultdict(dict)
    meta: dict[str, dict[str, str]] = {}
    for row in rows:
        ratio = float_value(row["reward_ratio"])
        if ratio is None:
            continue
        by_instance[row["instance_id"]][row["algorithm"]] = ratio
        meta.setdefault(
            row["instance_id"],
            {
                "n": row["n"],
                "regime": row["regime"],
                "service_scale": row["service_scale"],
                "budget_tightness": row["budget_tightness"],
                "seed": row["seed"],
            },
        )
    comparisons = [
        ("density_to_split_only", "split_only_grid", "service_density_then_route"),
        ("split_only_to_split_compact", "split_compact_grid", "split_only_grid"),
    ]
    detailed = []
    summary = []
    for label, algorithm, baseline in comparisons:
        deltas = []
        for instance_id, ratios in by_instance.items():
            if algorithm not in ratios or baseline not in ratios:
                continue
            delta = ratios[algorithm] - ratios[baseline]
            deltas.append(delta)
            detailed.append(
                {
                    "instance_id": instance_id,
                    **meta[instance_id],
                    "comparison": label,
                    "algorithm": algorithm,
                    "baseline": baseline,
                    "algorithm_reward_ratio": ratios[algorithm],
                    "baseline_reward_ratio": ratios[baseline],
                    "delta_reward_ratio": delta,
                }
            )
        summary.append(
            {
                "comparison": label,
                "count": len(deltas),
                "positive_count": sum(delta > 1e-12 for delta in deltas),
                "zero_count": sum(abs(delta) <= 1e-12 for delta in deltas),
                "negative_count": sum(delta < -1e-12 for delta in deltas),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "max_delta": max(deltas),
                "min_delta": min(deltas),
            }
        )
    return detailed, summary


def worst_cases(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    heuristics = [
        "best_single",
        "service_density_then_route",
        "cheapest_insertion",
        "split_only_grid",
        "split_compact_grid",
    ]
    output = []
    for algorithm in heuristics:
        alg_rows = [row for row in rows if row["algorithm"] == algorithm and row["reward_ratio"]]
        for row in sorted(alg_rows, key=lambda item: float(item["reward_ratio"]))[:10]:
            output.append(dict(row))
    return output


def selected_alpha_distribution(split_trace: pd.DataFrame) -> list[dict[str, Any]]:
    chosen = split_trace[split_trace["is_final_chosen"] == True]  # noqa: E712
    grouped = chosen.groupby(["algorithm", "alpha"]).size().reset_index(name="count")
    return grouped.to_dict("records")


def alpha_reward_summary(split_trace: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        split_trace.groupby(["algorithm", "alpha"])
        .agg(
            count=("reward", "size"),
            mean_reward_ratio=("reward_ratio", "mean"),
            median_reward_ratio=("reward_ratio", "median"),
            feasible_rate=("feasible", "mean"),
            final_chosen_count=("is_final_chosen", "sum"),
            reward_tied_with_final_count=("is_reward_tied_with_final", "sum"),
        )
        .reset_index()
    )
    return grouped.to_dict("records")


def canonical_route(route: list[str]) -> tuple[str, ...]:
    inner = route[1:-1]
    reversed_route = [route[0], *reversed(inner), route[-1]]
    return min(tuple(route), tuple(reversed_route))


def solution_similarity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_instance: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_instance[record["instance_id"]][record["algorithm"]] = record
    output = []
    for instance_id, algorithms in sorted(by_instance.items()):
        exact = algorithms["exact_subset_dp"]
        exact_subset = set(exact["selected_station_ids"])
        exact_route = canonical_route(exact["route_station_ids"])
        opt_reward = exact["reward"]
        for algorithm, record in sorted(algorithms.items()):
            subset = set(record["selected_station_ids"])
            route = canonical_route(record["route_station_ids"])
            reward_optimal = record["reward"] == opt_reward
            subset_match = subset == exact_subset
            route_match = route == exact_route
            output.append(
                {
                    "instance_id": instance_id,
                    "algorithm": algorithm,
                    "reward_optimal": reward_optimal,
                    "canonical_subset_match": subset_match,
                    "canonical_route_match_up_to_reversal": route_match,
                    "same_subset_different_order": subset_match and not route_match,
                    "optimal_reward_different_subset": reward_optimal and not subset_match,
                }
            )
    return output


def mask_stats(instance: Instance) -> dict[str, Any]:
    n = len(instance.waypoints)
    size = 1 << n
    inf = np.iinfo(np.int64).max // 4
    costs = doubled_transformed_cost_s(instance)
    dp = np.full((size, n), inf, dtype=np.int64)
    for bit in range(n):
        dp[1 << bit, bit] = costs[0, bit + 1]
    for mask in range(size):
        for end in range(n):
            current = int(dp[mask, end])
            if current >= inf:
                continue
            remaining = ((1 << n) - 1) ^ mask
            while remaining:
                lowbit = remaining & -remaining
                nxt = lowbit.bit_length() - 1
                candidate = current + int(costs[end + 1, nxt + 1])
                next_mask = mask | lowbit
                if candidate < dp[next_mask, nxt]:
                    dp[next_mask, nxt] = candidate
                remaining ^= lowbit

    reward_by_mask = np.zeros(size, dtype=np.int64)
    station_count_by_mask = np.zeros(size, dtype=np.int64)
    for mask in range(1, size):
        lowbit = mask & -mask
        bit = lowbit.bit_length() - 1
        previous = mask ^ lowbit
        reward_by_mask[mask] = reward_by_mask[previous] + instance.waypoints[bit].reward
        station_count_by_mask[mask] = station_count_by_mask[previous] + 1

    feasible_rewards = [0]
    optimal_masks = [0]
    opt_reward = 0
    budget2 = 2 * instance.budget_s
    for mask in range(1, size):
        best_closed = min(
            int(dp[mask, end]) + int(costs[end + 1, 0])
            for end in range(n)
            if int(dp[mask, end]) < inf
        )
        if best_closed <= budget2:
            reward = int(reward_by_mask[mask])
            feasible_rewards.append(reward)
            if reward > opt_reward:
                opt_reward = reward
                optimal_masks = [mask]
            elif reward == opt_reward:
                optimal_masks.append(mask)

    station_rewards = [waypoint.reward for waypoint in instance.waypoints]
    optimal_counts = [int(station_count_by_mask[mask]) for mask in optimal_masks]
    return {
        "instance_id": instance.instance_id,
        "n": n,
        "regime": instance.metadata.get("spatial_regime", ""),
        "service_scale": instance.metadata.get("service_scale", ""),
        "budget_tightness": instance.metadata.get("budget_tightness", ""),
        "seed": instance.metadata.get("seed", ""),
        "opt_reward": opt_reward,
        "feasible_subset_count": len(feasible_rewards),
        "optimal_subset_count": len(optimal_masks),
        "distinct_feasible_subset_reward_values": len(set(feasible_rewards)),
        "optimal_subset_fraction": len(optimal_masks) / len(feasible_rewards),
        "min_optimal_station_count": min(optimal_counts),
        "max_optimal_station_count": max(optimal_counts),
        "station_rewards": json.dumps(station_rewards),
        "distinct_station_rewards": len(set(station_rewards)),
    }


def optimal_tie_analysis(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_csv_rows(manifest_path)
    output = []
    for row in manifest:
        output.append(mask_stats(read_instance_json(Path(row["path"]))))
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = read_csv_rows(args.results)
    solution_records = read_jsonl(args.solutions)
    split_trace = pd.read_parquet(args.split_trace)

    write_csv(args.output_dir / "exact_suite_exact_match_rates.csv", exact_match_rates(result_rows))
    for field in ["n", "regime", "service_scale", "budget_tightness"]:
        write_csv(args.output_dir / f"exact_suite_breakdown_by_{field}.csv", breakdown(result_rows, field))
    write_csv(args.output_dir / "exact_suite_runtime_by_n.csv", runtime_by_n(result_rows))
    detailed, summary = paired_improvements(result_rows)
    write_csv(args.output_dir / "exact_suite_paired_improvement_rows.csv", detailed)
    write_csv(args.output_dir / "exact_suite_paired_improvement_summary.csv", summary)
    write_csv(args.output_dir / "exact_suite_worst_cases.csv", worst_cases(result_rows))
    write_csv(args.output_dir / "exact_suite_selected_alpha_distribution.csv", selected_alpha_distribution(split_trace))
    write_csv(args.output_dir / "exact_suite_alpha_reward_summary.csv", alpha_reward_summary(split_trace))
    write_csv(args.output_dir / "solution_similarity_summary.csv", solution_similarity(solution_records))
    write_csv(args.output_dir / "optimal_tie_analysis.csv", optimal_tie_analysis(args.manifest))
    print("Wrote diagnostic tables to", args.output_dir)


if __name__ == "__main__":
    main()
