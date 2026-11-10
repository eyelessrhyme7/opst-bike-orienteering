from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Callable

from opst.baselines import (
    solve_best_single,
    solve_cheapest_insertion,
    solve_service_density_then_route,
)
from opst.exact_dp import solve_exact_subset_dp
from opst.instance_generation import read_instance_json
from opst.models import Instance, Solution
from opst.route_ops import (
    augment_route_feasibly,
    repair_route_by_removal,
    route_selected_by_cheapest_insertion,
)
from opst.split_heuristic import (
    DEFAULT_ALPHA_GRID,
    compact_candidate_pools,
    select_service_knapsack,
    solve_split_compact_grid,
    solve_split_only_grid,
)
from opst.validate import make_solution, validate_solution

Solver = Callable[[Instance], Solution]

SOLVERS: dict[str, Solver] = {
    "exact_subset_dp": solve_exact_subset_dp,
    "best_single": solve_best_single,
    "service_density_then_route": solve_service_density_then_route,
    "cheapest_insertion": solve_cheapest_insertion,
    "split_only_grid": solve_split_only_grid,
    "split_compact_grid": solve_split_compact_grid,
}

DEFAULT_ALGORITHMS = tuple(SOLVERS.keys())

RESULT_FIELDNAMES = [
    "run_id",
    "instance_id",
    "seed",
    "n",
    "regime",
    "service_scale",
    "budget_tightness",
    "algorithm",
    "reward",
    "opt_reward",
    "reward_ratio",
    "approximation_factor",
    "travel_time_s",
    "service_time_s",
    "total_time_s",
    "budget_s",
    "feasible",
    "selected_count",
    "chosen_alpha",
    "chosen_pool_type",
    "chosen_pool_id",
    "solution_record_id",
    "runtime_ms",
    "status",
    "error",
]


def _base_row(instance: Instance, *, run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "instance_id": instance.instance_id,
        "seed": instance.metadata.get("seed", ""),
        "n": len(instance.waypoints),
        "regime": instance.metadata.get("spatial_regime", ""),
        "service_scale": instance.metadata.get("service_scale", ""),
        "budget_tightness": instance.metadata.get("budget_tightness", ""),
        "budget_s": instance.budget_s,
    }


def _raw_station_id(station_id: object) -> str:
    value = str(station_id)
    return value.removeprefix("station_")


def _route_station_ids(instance: Instance, route_indices: tuple[int, ...]) -> list[str]:
    output = []
    for index in route_indices:
        if index == 0:
            output.append(_raw_station_id(instance.depot_id))
        else:
            output.append(str(instance.waypoints[index - 1].metadata.get("station_id", index)))
    return output


def _route_global_matrix_indices(instance: Instance, route_indices: tuple[int, ...]) -> list[int]:
    source_indices = instance.metadata.get("source_matrix_indices")
    if isinstance(source_indices, list) and len(source_indices) >= len(instance.waypoints) + 1:
        return [int(source_indices[index]) for index in route_indices]
    return list(route_indices)


def _selected_station_ids(instance: Instance, selected_indices: tuple[int, ...]) -> list[str]:
    return [
        str(instance.waypoints[index - 1].metadata.get("station_id", index))
        for index in selected_indices
    ]


def _pool_type_and_id(pool: object) -> tuple[str, str]:
    value = "" if pool is None else str(pool)
    if value == "":
        return "", ""
    if value == "all":
        return "all", "all"
    if value.startswith("cell_plus_ring:"):
        return "cell_plus_ring", value.split(":", 1)[1]
    if value.startswith("cell:"):
        return "cell", value.split(":", 1)[1]
    return "custom", value


def solution_record(
    instance: Instance,
    solution: Solution,
    *,
    run_id: str,
    opt_reward: int | None,
    solution_record_id: str,
) -> dict[str, object]:
    validation = validate_solution(instance, solution)
    pool_type, pool_id = _pool_type_and_id(solution.parameters.get("pool"))
    reward_ratio = None
    if opt_reward is not None and opt_reward > 0:
        reward_ratio = solution.reward / opt_reward
    elif opt_reward == 0 and solution.reward == 0:
        reward_ratio = 1.0
    return {
        "solution_record_id": solution_record_id,
        "run_id": run_id,
        "instance_id": instance.instance_id,
        "algorithm": solution.algorithm,
        "route_station_ids": _route_station_ids(instance, solution.route_indices),
        "route_matrix_indices": _route_global_matrix_indices(instance, solution.route_indices),
        "selected_station_ids": _selected_station_ids(instance, solution.selected_indices),
        "reward": solution.reward,
        "opt_reward": opt_reward,
        "reward_ratio": reward_ratio,
        "travel_time_s": solution.travel_time_s,
        "service_time_s": solution.service_time_s,
        "total_time_s": solution.total_time_s,
        "budget_s": instance.budget_s,
        "unused_budget_s": instance.budget_s - solution.total_time_s,
        "feasible": solution.feasible,
        "runtime_ms": solution.runtime_ms,
        "selected_count": len(solution.selected_indices),
        "n": len(instance.waypoints),
        "regime": instance.metadata.get("spatial_regime", ""),
        "service_scale": instance.metadata.get("service_scale", ""),
        "budget_tightness": instance.metadata.get("budget_tightness", ""),
        "seed": instance.metadata.get("seed", ""),
        "parameters": solution.parameters,
        "chosen_alpha": solution.parameters.get("alpha"),
        "chosen_pool_type": pool_type,
        "chosen_pool_id": pool_id,
        "repair_removals": solution.parameters.get("repair_removals"),
        "augmentation_additions": solution.parameters.get("augmentation_additions"),
        "validator_status": "ok" if validation.ok else "; ".join(validation.errors),
    }


def solution_to_result_row(
    instance: Instance,
    solution: Solution,
    *,
    run_id: str,
    opt_reward: int | None,
    solution_record_id: str = "",
) -> dict[str, object]:
    validation = validate_solution(instance, solution)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    reward_ratio = ""
    approximation_factor = ""
    if opt_reward is not None and opt_reward > 0 and solution.reward > 0:
        reward_ratio = solution.reward / opt_reward
        approximation_factor = opt_reward / solution.reward
    elif opt_reward == 0 and solution.reward == 0:
        reward_ratio = 1.0
        approximation_factor = 1.0

    pool_type, pool_id = _pool_type_and_id(solution.parameters.get("pool"))
    return {
        **_base_row(instance, run_id=run_id),
        "algorithm": solution.algorithm,
        "reward": solution.reward,
        "opt_reward": "" if opt_reward is None else opt_reward,
        "reward_ratio": reward_ratio,
        "approximation_factor": approximation_factor,
        "travel_time_s": solution.travel_time_s,
        "service_time_s": solution.service_time_s,
        "total_time_s": solution.total_time_s,
        "feasible": solution.feasible,
        "selected_count": len(solution.selected_indices),
        "chosen_alpha": solution.parameters.get("alpha", ""),
        "chosen_pool_type": pool_type,
        "chosen_pool_id": pool_id,
        "solution_record_id": solution_record_id,
        "runtime_ms": round(solution.runtime_ms, 3),
        "status": solution.status,
        "error": "",
    }


def error_result_row(
    instance: Instance,
    *,
    run_id: str,
    algorithm: str,
    error: Exception,
    opt_reward: int | None,
) -> dict[str, object]:
    return {
        **_base_row(instance, run_id=run_id),
        "algorithm": algorithm,
        "reward": "",
        "opt_reward": "" if opt_reward is None else opt_reward,
        "reward_ratio": "",
        "approximation_factor": "",
        "travel_time_s": "",
        "service_time_s": "",
        "total_time_s": "",
        "feasible": False,
        "selected_count": "",
        "chosen_alpha": "",
        "chosen_pool_type": "",
        "chosen_pool_id": "",
        "solution_record_id": "",
        "runtime_ms": "",
        "status": "error",
        "error": str(error),
    }


def run_algorithms_on_instance(
    instance: Instance,
    *,
    run_id: str,
    algorithm_names: tuple[str, ...] = DEFAULT_ALGORITHMS,
) -> list[dict[str, object]]:
    """Run algorithms and return validator-backed result rows."""

    rows: list[dict[str, object]] = []
    opt_reward: int | None = None
    for algorithm in algorithm_names:
        solver = SOLVERS[algorithm]
        try:
            solution = solver(instance)
            if algorithm == "exact_subset_dp":
                opt_reward = solution.reward
            rows.append(
                solution_to_result_row(
                    instance,
                    solution,
                    run_id=run_id,
                    opt_reward=opt_reward,
                )
            )
        except Exception as exc:
            rows.append(
                error_result_row(
                    instance,
                    run_id=run_id,
                    algorithm=algorithm,
                    error=exc,
                    opt_reward=opt_reward,
                )
            )
    return rows


def _candidate_key(
    solution: Solution,
    *,
    alpha: float,
    pool_name: str,
) -> tuple[int, int, int, float, str, tuple[int, ...]]:
    return (
        -solution.reward,
        solution.total_time_s,
        solution.travel_time_s,
        alpha,
        pool_name,
        solution.route_indices,
    )


def _solve_split_candidate(
    instance: Instance,
    *,
    candidate_indices: tuple[int, ...],
    alpha: float,
    algorithm: str,
    pool_name: str,
    pool_size: int,
) -> tuple[Solution, int, int]:
    service_budget_s = int(alpha * instance.budget_s)
    selected = select_service_knapsack(instance, candidate_indices, service_budget_s)
    selected_set = set(selected)
    route = route_selected_by_cheapest_insertion(instance, selected)
    repaired_route = repair_route_by_removal(instance, route)
    repaired_set = set(repaired_route[1:-1])
    final_route = augment_route_feasibly(instance, repaired_route, candidate_indices)
    final_set = set(final_route[1:-1])
    repair_removals = len(selected_set - repaired_set)
    augmentation_additions = len(final_set - repaired_set)
    solution = make_solution(
        instance,
        algorithm,
        final_route,
        runtime_ms=0.0,
        parameters={
            "alpha": alpha,
            "service_budget_s": service_budget_s,
            "pool": pool_name,
            "pool_size": pool_size,
            "candidate_count": len(candidate_indices),
            "repair_removals": repair_removals,
            "augmentation_additions": augmentation_additions,
        },
        status="ok",
    )
    validate_solution(instance, solution, strict=True)
    return solution, repair_removals, augmentation_additions


def trace_split_algorithm(
    instance: Instance,
    *,
    algorithm: str,
    opt_reward: int | None,
) -> tuple[Solution, list[dict[str, object]], list[dict[str, object]]]:
    if algorithm == "split_only_grid":
        pools = (("all", tuple(range(1, len(instance.waypoints) + 1))),)
    elif algorithm == "split_compact_grid":
        pools = compact_candidate_pools(instance)
    else:
        raise ValueError(f"not a split algorithm: {algorithm}")

    best_solution: Solution | None = None
    best_alpha = 0.0
    best_pool = ""
    alpha_rows: list[dict[str, object]] = []
    pool_rows: list[dict[str, object]] = []
    final_candidates_by_alpha: dict[float, Solution] = {}

    total_start = time.perf_counter()
    for alpha in DEFAULT_ALPHA_GRID:
        alpha_start = time.perf_counter()
        best_for_alpha: tuple[Solution, str, int, int] | None = None
        for pool_name, candidate_indices in pools:
            pool_start = time.perf_counter()
            solution, repair_removals, augmentation_additions = _solve_split_candidate(
                instance,
                candidate_indices=candidate_indices,
                alpha=alpha,
                algorithm=algorithm,
                pool_name=pool_name,
                pool_size=len(candidate_indices),
            )
            pool_runtime_ms = (time.perf_counter() - pool_start) * 1000
            pool_type, pool_id = _pool_type_and_id(pool_name)
            ratio = "" if not opt_reward else solution.reward / opt_reward
            pool_rows.append(
                {
                    "instance_id": instance.instance_id,
                    "algorithm": algorithm,
                    "alpha": alpha,
                    "pool_runtime_ms": pool_runtime_ms,
                    "reward": solution.reward,
                    "reward_ratio": ratio,
                    "selected_station_ids": json.dumps(
                        _selected_station_ids(instance, solution.selected_indices)
                    ),
                    "route_station_ids": json.dumps(
                        _route_station_ids(instance, solution.route_indices)
                    ),
                    "travel_time_s": solution.travel_time_s,
                    "service_time_s": solution.service_time_s,
                    "total_time_s": solution.total_time_s,
                    "feasible": solution.feasible,
                    "pool_type": pool_type,
                    "pool_id": pool_id,
                    "pool_size": len(candidate_indices),
                    "repair_removals": repair_removals,
                    "augmentation_additions": augmentation_additions,
                }
            )
            candidate_tuple = (solution, pool_name, repair_removals, augmentation_additions)
            if best_for_alpha is None or _candidate_key(
                solution, alpha=alpha, pool_name=pool_name
            ) < _candidate_key(best_for_alpha[0], alpha=alpha, pool_name=best_for_alpha[1]):
                best_for_alpha = candidate_tuple

        assert best_for_alpha is not None
        alpha_solution, alpha_pool, alpha_repairs, alpha_additions = best_for_alpha
        final_candidates_by_alpha[alpha] = alpha_solution
        alpha_runtime_ms = (time.perf_counter() - alpha_start) * 1000
        pool_type, pool_id = _pool_type_and_id(alpha_pool)
        alpha_rows.append(
            {
                "instance_id": instance.instance_id,
                "algorithm": algorithm,
                "alpha": alpha,
                "alpha_runtime_ms": alpha_runtime_ms,
                "reward": alpha_solution.reward,
                "reward_ratio": "" if not opt_reward else alpha_solution.reward / opt_reward,
                "selected_station_ids": json.dumps(
                    _selected_station_ids(instance, alpha_solution.selected_indices)
                ),
                "route_station_ids": json.dumps(
                    _route_station_ids(instance, alpha_solution.route_indices)
                ),
                "travel_time_s": alpha_solution.travel_time_s,
                "service_time_s": alpha_solution.service_time_s,
                "total_time_s": alpha_solution.total_time_s,
                "feasible": alpha_solution.feasible,
                "pool_type": pool_type,
                "pool_id": pool_id,
                "pool_size": int(alpha_solution.parameters.get("pool_size", 0)),
                "repair_removals": alpha_repairs,
                "augmentation_additions": alpha_additions,
                "is_final_chosen": False,
                "is_reward_tied_with_final": False,
            }
        )
        if best_solution is None or _candidate_key(
            alpha_solution, alpha=alpha, pool_name=alpha_pool
        ) < _candidate_key(best_solution, alpha=best_alpha, pool_name=best_pool):
            best_solution = alpha_solution
            best_alpha = alpha
            best_pool = alpha_pool

    assert best_solution is not None
    total_runtime_ms = (time.perf_counter() - total_start) * 1000
    best_solution = Solution(
        **{
            **best_solution.__dict__,
            "runtime_ms": total_runtime_ms,
        }
    )
    final_reward = best_solution.reward
    for row in alpha_rows:
        row["is_final_chosen"] = (
            row["alpha"] == best_alpha
            and row["pool_id"] == _pool_type_and_id(best_pool)[1]
            and row["pool_type"] == _pool_type_and_id(best_pool)[0]
        )
        row["is_reward_tied_with_final"] = row["reward"] == final_reward
    for row in pool_rows:
        row["is_final_chosen"] = (
            row["alpha"] == best_alpha
            and row["pool_id"] == _pool_type_and_id(best_pool)[1]
            and row["pool_type"] == _pool_type_and_id(best_pool)[0]
            and row["reward"] == final_reward
        )
        row["is_reward_tied_with_final"] = row["reward"] == final_reward
    return best_solution, alpha_rows, pool_rows


def run_manifest_instrumented(
    manifest_path: Path,
    *,
    run_id: str,
    algorithm_names: tuple[str, ...] = DEFAULT_ALGORITHMS,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    result_rows: list[dict[str, object]] = []
    solution_records: list[dict[str, object]] = []
    alpha_trace_rows: list[dict[str, object]] = []
    compact_trace_rows: list[dict[str, object]] = []

    for manifest_row in manifest_rows:
        instance = read_instance_json(Path(manifest_row["path"]))
        opt_reward: int | None = None
        for algorithm in algorithm_names:
            try:
                if algorithm in {"split_only_grid", "split_compact_grid"}:
                    solution, alpha_rows, pool_rows = trace_split_algorithm(
                        instance, algorithm=algorithm, opt_reward=opt_reward
                    )
                    alpha_trace_rows.extend(alpha_rows)
                    if algorithm == "split_compact_grid":
                        compact_trace_rows.extend(pool_rows)
                else:
                    solution = SOLVERS[algorithm](instance)
                if algorithm == "exact_subset_dp":
                    opt_reward = solution.reward
                record_id = f"{instance.instance_id}::{solution.algorithm}"
                solution_records.append(
                    solution_record(
                        instance,
                        solution,
                        run_id=run_id,
                        opt_reward=opt_reward,
                        solution_record_id=record_id,
                    )
                )
                result_rows.append(
                    solution_to_result_row(
                        instance,
                        solution,
                        run_id=run_id,
                        opt_reward=opt_reward,
                        solution_record_id=record_id,
                    )
                )
            except Exception as exc:
                result_rows.append(
                    error_result_row(
                        instance,
                        run_id=run_id,
                        algorithm=algorithm,
                        error=exc,
                        opt_reward=opt_reward,
                    )
                )
    return result_rows, solution_records, alpha_trace_rows, compact_trace_rows


def run_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    algorithm_names: tuple[str, ...] = DEFAULT_ALGORITHMS,
) -> list[dict[str, object]]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for manifest_row in manifest_rows:
        instance = read_instance_json(Path(manifest_row["path"]))
        rows.extend(
            run_algorithms_on_instance(
                instance,
                run_id=run_id,
                algorithm_names=algorithm_names,
            )
        )
    return rows


def write_results_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_parquet(rows: list[dict[str, object]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
