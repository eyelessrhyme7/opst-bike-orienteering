from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt

ALGORITHM_ORDER = [
    "best_single",
    "service_density_then_route",
    "cheapest_insertion",
    "split_only_grid",
    "split_compact_grid",
    "exact_subset_dp",
]


def read_result_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def summarize_by_algorithm(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(row)

    summaries = []
    for algorithm in sorted(grouped, key=lambda name: ALGORITHM_ORDER.index(name) if name in ALGORITHM_ORDER else 99):
        algorithm_rows = grouped[algorithm]
        ratios = [
            value
            for row in algorithm_rows
            if (value := _float_or_none(row["reward_ratio"])) is not None
        ]
        runtimes = [float(row["runtime_ms"]) for row in algorithm_rows if row["runtime_ms"]]
        summaries.append(
            {
                "algorithm": algorithm,
                "row_count": len(algorithm_rows),
                "feasible_count": sum(row["feasible"] == "True" for row in algorithm_rows),
                "error_count": sum(row["status"] == "error" for row in algorithm_rows),
                "median_reward_ratio": "" if not ratios else median(ratios),
                "mean_reward_ratio": "" if not ratios else mean(ratios),
                "p10_reward_ratio": "" if not ratios else _percentile(ratios, 0.10),
                "min_reward_ratio": "" if not ratios else min(ratios),
                "median_runtime_ms": "" if not runtimes else median(runtimes),
            }
        )
    return summaries


def paired_ablations(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_instance: dict[str, dict[str, float]] = defaultdict(dict)
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        ratio = _float_or_none(row["reward_ratio"])
        if ratio is None:
            continue
        instance_id = row["instance_id"]
        by_instance[instance_id][row["algorithm"]] = ratio
        metadata.setdefault(
            instance_id,
            {
                "regime": row["regime"],
                "budget_tightness": row["budget_tightness"],
                "seed": row["seed"],
            },
        )

    comparisons = [
        ("split_only_vs_density", "split_only_grid", "service_density_then_route"),
        ("split_compact_vs_split_only", "split_compact_grid", "split_only_grid"),
        ("insertion_vs_split_compact", "cheapest_insertion", "split_compact_grid"),
    ]
    output = []
    for instance_id, ratios in sorted(by_instance.items()):
        for label, better, baseline in comparisons:
            if better not in ratios or baseline not in ratios:
                continue
            output.append(
                {
                    "instance_id": instance_id,
                    **metadata[instance_id],
                    "comparison": label,
                    "algorithm": better,
                    "baseline": baseline,
                    "algorithm_reward_ratio": ratios[better],
                    "baseline_reward_ratio": ratios[baseline],
                    "delta_reward_ratio": ratios[better] - ratios[baseline],
                }
            )
    return output


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_reward_ratio_boxplot(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    labels = []
    for algorithm in ALGORITHM_ORDER:
        values = [
            value
            for row in rows
            if row["algorithm"] == algorithm
            if (value := _float_or_none(row["reward_ratio"])) is not None
        ]
        if values:
            labels.append(algorithm)
            data.append(values)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("reward ratio ALG/OPT")
    ax.set_title("Smoke Suite Reward Ratios by Algorithm")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_markdown_summary(path: Path, summaries: list[dict[str, object]], row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Results Summary",
        "",
        f"Rows analyzed: {row_count}",
        "",
        "This file is generated from saved result CSV rows.",
        "",
        "## Algorithm Summary",
        "",
        "| Algorithm | Rows | Median ALG/OPT | Min ALG/OPT | Median runtime ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {algorithm} | {row_count} | {median_reward_ratio} | {min_reward_ratio} | {median_runtime_ms} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
