from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "exact_subset_dp": "black",
    "split_only_grid": "tab:blue",
    "split_compact_grid": "tab:green",
    "cheapest_insertion": "tab:orange",
    "service_density_then_route": "tab:purple",
    "best_single": "tab:gray",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact-suite diagnostic plots.")
    parser.add_argument("--results", type=Path, default=Path("results/raw/exact_suite_results.csv"))
    parser.add_argument(
        "--split-trace", type=Path, default=Path("results/raw/exact_suite_split_trace.parquet")
    )
    parser.add_argument(
        "--paired", type=Path, default=Path("results/tables/exact_suite_paired_improvement_rows.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["reward_ratio"] = pd.to_numeric(df["reward_ratio"], errors="coerce")
    df["runtime_ms"] = pd.to_numeric(df["runtime_ms"], errors="coerce")
    return df


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_ecdf(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algorithm, group in df.groupby("algorithm"):
        values = np.sort(group["reward_ratio"].dropna().to_numpy())
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, label=algorithm, color=COLORS.get(algorithm))
    ax.set_xlabel("ALG/OPT")
    ax.set_ylabel("ECDF")
    ax.set_title("Full Exact Suite: Reward Ratio ECDF")
    ax.legend(fontsize=7)
    save(fig, path)


def plot_box_by_field(df: pd.DataFrame, field: str, path: Path) -> None:
    heuristics = [
        "service_density_then_route",
        "cheapest_insertion",
        "split_only_grid",
        "split_compact_grid",
    ]
    labels = []
    data = []
    for value in sorted(df[field].unique()):
        for algorithm in heuristics:
            values = df[(df[field] == value) & (df["algorithm"] == algorithm)]["reward_ratio"].dropna()
            if not values.empty:
                labels.append(f"{value}\n{algorithm.replace('_grid', '').replace('_then_route', '')}")
                data.append(values.to_numpy())
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ALG/OPT")
    ax.set_title(f"Full Exact Suite: Reward Ratio by {field}")
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    save(fig, path)


def plot_paired_delta(paired_path: Path, path: Path) -> None:
    paired = pd.read_csv(paired_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for comparison, group in paired.groupby("comparison"):
        ax.hist(group["delta_reward_ratio"], bins=30, alpha=0.6, label=comparison)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Delta reward ratio")
    ax.set_ylabel("Instance count")
    ax.set_title("Full Exact Suite: Paired Improvement Distribution")
    ax.legend(fontsize=8)
    save(fig, path)


def plot_alpha_hist(trace: pd.DataFrame, path: Path) -> None:
    chosen = trace[trace["is_final_chosen"] == True]  # noqa: E712
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algorithm, group in chosen.groupby("algorithm"):
        ax.hist(group["alpha"], bins=np.arange(0.175, 0.951, 0.05), alpha=0.6, label=algorithm)
    ax.set_xlabel("Chosen alpha")
    ax.set_ylabel("Instance count")
    ax.set_title("Full Exact Suite: Chosen Alpha Distribution")
    ax.legend()
    save(fig, path)


def plot_alpha_curve(trace: pd.DataFrame, results: pd.DataFrame, path: Path) -> None:
    split_only = results[results["algorithm"] == "split_only_grid"].sort_values("reward_ratio")
    chosen_instances = list(split_only["instance_id"].head(2))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for instance_id in chosen_instances:
        subset = trace[
            (trace["instance_id"] == instance_id)
            & (trace["algorithm"].isin(["split_only_grid", "split_compact_grid"]))
        ]
        for algorithm, group in subset.groupby("algorithm"):
            ax.plot(group["alpha"], group["reward_ratio"], marker="o", label=f"{instance_id} {algorithm}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("ALG/OPT at alpha")
    ax.set_title("Full Exact Suite: Alpha vs Reward Ratio for Worst Split-Only Cases")
    ax.legend(fontsize=6)
    save(fig, path)


def plot_runtime_by_n(df: pd.DataFrame, path: Path) -> None:
    grouped = df.groupby(["algorithm", "n"])["runtime_ms"].median().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algorithm, group in grouped.groupby("algorithm"):
        ax.plot(group["n"], group["runtime_ms"], marker="o", label=algorithm, color=COLORS.get(algorithm))
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Median runtime ms (log scale)")
    ax.set_title("Full Exact Suite: Runtime vs n")
    ax.legend(fontsize=7)
    save(fig, path)


def plot_opt_reward_selected(df: pd.DataFrame, path: Path) -> None:
    exact = df[df["algorithm"] == "exact_subset_dp"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(exact["reward"], bins=20, color="tab:red", alpha=0.8)
    axes[0].set_title("OPT reward distribution")
    axes[0].set_xlabel("OPT reward")
    axes[0].set_ylabel("count")
    axes[1].hist(exact["selected_count"], bins=range(0, int(exact["selected_count"].max()) + 2), color="tab:blue")
    axes[1].set_title("OPT selected count distribution")
    axes[1].set_xlabel("selected count")
    save(fig, path)


def main() -> None:
    args = parse_args()
    results = read_results(args.results)
    trace = pd.read_parquet(args.split_trace)
    plot_ecdf(results, args.output_dir / "exact_suite_reward_ratio_ecdf.png")
    plot_box_by_field(results, "regime", args.output_dir / "exact_suite_reward_ratio_by_regime.png")
    plot_box_by_field(
        results, "budget_tightness", args.output_dir / "exact_suite_reward_ratio_by_budget.png"
    )
    plot_paired_delta(args.paired, args.output_dir / "exact_suite_paired_delta_hist.png")
    plot_alpha_hist(trace, args.output_dir / "exact_suite_chosen_alpha_hist.png")
    plot_alpha_curve(trace, results, args.output_dir / "exact_suite_alpha_reward_curves.png")
    plot_runtime_by_n(results, args.output_dir / "exact_suite_runtime_vs_n.png")
    plot_opt_reward_selected(results, args.output_dir / "exact_suite_opt_reward_selected_dist.png")
    print("Wrote diagnostic plots to", args.output_dir)


if __name__ == "__main__":
    main()
