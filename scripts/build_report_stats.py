"""Derive report-specific summary tables from frozen Phase 3 artifacts.

Reads only saved results/traces; never reruns solvers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
TABLES = ROOT / "results/tables"
SPLIT_ALGOS = ("split_only_grid", "split_compact_grid")


def alpha_trace_stats(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in SPLIT_ALGOS:
        sub = trace[trace["algorithm"] == algorithm].sort_values(["instance_id", "alpha"])
        tied_counts = []
        distinct_rewards = []
        non_monotone = 0
        drops = 0
        for _, group in sub.groupby("instance_id"):
            rewards = group["reward"].tolist()
            tied_counts.append(int(group["is_reward_tied_with_final"].sum()))
            distinct_rewards.append(len(set(rewards)))
            diffs = [b - a for a, b in zip(rewards, rewards[1:], strict=False)]
            if any(d > 0 for d in diffs) and any(d < 0 for d in diffs):
                non_monotone += 1
            if any(d < 0 for d in diffs):
                drops += 1
        count = len(tied_counts)
        rows.append(
            {
                "algorithm": algorithm,
                "instances": count,
                "mean_alphas_tied_with_final": sum(tied_counts) / count,
                "instances_all_15_alphas_tied": sum(c == 15 for c in tied_counts),
                "instances_single_best_alpha": sum(c == 1 for c in tied_counts),
                "mean_distinct_rewards_across_alpha": sum(distinct_rewards) / count,
                "instances_reward_constant_in_alpha": sum(d == 1 for d in distinct_rewards),
                "instances_with_reward_drop_as_alpha_increases": drops,
                "instances_non_monotone_in_alpha": non_monotone,
                "mean_alpha_runtime_ms": float(sub["alpha_runtime_ms"].mean()),
            }
        )
    return pd.DataFrame(rows)


def fixed_alpha_counterfactual(trace: pd.DataFrame) -> pd.DataFrame:
    sub = trace[(trace["algorithm"] == "split_only_grid")]
    rows = []
    for alpha, group in sub.groupby("alpha"):
        rows.append(
            {
                "alpha": float(alpha),
                "instances": len(group),
                "mean_reward_ratio": float(group["reward_ratio"].mean()),
                "min_reward_ratio": float(group["reward_ratio"].min()),
                "exact_reward_match_rate": float((group["reward_ratio"] >= 1 - 1e-9).mean()),
            }
        )
    return pd.DataFrame(rows)


def compact_gain_cases(results: pd.DataFrame, solutions: dict[str, dict]) -> pd.DataFrame:
    pivot = results.pivot_table(
        index="instance_id", columns="algorithm", values="reward_ratio", aggfunc="first"
    )
    gains = (pivot["split_compact_grid"] - pivot["split_only_grid"]).sort_values(ascending=False)
    rows = []
    for instance_id, delta in gains[gains > 1e-12].items():
        meta = results[results["instance_id"] == instance_id].iloc[0]
        compact = solutions[f"{instance_id}::split_compact_grid"]
        rows.append(
            {
                "instance_id": instance_id,
                "n": int(meta["n"]),
                "regime": meta["regime"],
                "service_scale": float(meta["service_scale"]),
                "budget_tightness": meta["budget_tightness"],
                "delta_reward_ratio": float(delta),
                "split_only_ratio": float(pivot.loc[instance_id, "split_only_grid"]),
                "split_compact_ratio": float(pivot.loc[instance_id, "split_compact_grid"]),
                "compact_pool_type": compact.get("chosen_pool_type"),
            }
        )
    return pd.DataFrame(rows)


def case_study(results: pd.DataFrame, instance_id: str) -> pd.DataFrame:
    sub = results[results["instance_id"] == instance_id]
    cols = [
        "algorithm",
        "reward",
        "opt_reward",
        "reward_ratio",
        "travel_time_s",
        "service_time_s",
        "total_time_s",
        "budget_s",
        "selected_count",
        "chosen_alpha",
        "chosen_pool_type",
    ]
    return sub[cols]


def main() -> None:
    trace = pd.read_parquet(RAW / "exact_suite_split_trace.parquet")
    results = pd.read_csv(RAW / "exact_suite_results.csv")
    solutions: dict[str, dict] = {}
    with (RAW / "exact_suite_solutions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            solutions[f"{record['instance_id']}::{record['algorithm']}"] = record

    alpha_trace_stats(trace).to_csv(TABLES / "report_alpha_trace_stats.csv", index=False)
    fixed_alpha_counterfactual(trace).to_csv(
        TABLES / "report_fixed_alpha_counterfactual.csv", index=False
    )
    gains = compact_gain_cases(results, solutions)
    gains.to_csv(TABLES / "report_compact_gain_cases.csv", index=False)
    top = str(gains.iloc[0]["instance_id"])
    case_study(results, top).to_csv(TABLES / "report_case_study.csv", index=False)
    print(f"Wrote report stats; compact-gain cases: {len(gains)}; case study: {top}")


if __name__ == "__main__":
    main()
