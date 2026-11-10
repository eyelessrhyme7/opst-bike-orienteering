"""Build the main-body report figures from frozen Phase 3 artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opst.instance_generation import read_instance_json

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
TABLES = ROOT / "results/tables"
OUT = ROOT / "report/assets/figures"

DISPLAY = {
    "exact_subset_dp": "Exact DP",
    "best_single": "Best single",
    "service_density_then_route": "Density then route",
    "cheapest_insertion": "Cheapest insertion",
    "split_only_grid": "Split-only",
    "split_compact_grid": "Split + compact",
}
COLORS = {
    "exact_subset_dp": "black",
    "best_single": "0.6",
    "service_density_then_route": "tab:purple",
    "cheapest_insertion": "tab:orange",
    "split_only_grid": "tab:blue",
    "split_compact_grid": "tab:green",
}
ORDER = [
    "best_single",
    "service_density_then_route",
    "cheapest_insertion",
    "split_only_grid",
    "split_compact_grid",
    "exact_subset_dp",
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def problem_figure() -> None:
    depot = (0.0, 0.0)
    cluster = {
        "a": (1.0, 0.9, 4, 3),
        "b": (1.6, 1.4, 3, 2),
        "c": (2.1, 0.8, 5, 4),
        "d": (1.4, 0.2, 3, 2),
    }
    far = {"e": (4.8, 2.6, 9, 3), "f": (-1.2, 2.6, 8, 4)}
    route = [depot, cluster["a"][:2], cluster["b"][:2], cluster["c"][:2], cluster["d"][:2], depot]
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    xs, ys = zip(*route, strict=False)
    ax.plot(xs, ys, color="tab:blue", lw=1.8, zorder=2)
    for (x0, y0), (x1, y1) in zip(route, route[1:], strict=False):
        ax.annotate(
            "",
            xy=(x0 + 0.6 * (x1 - x0), y0 + 0.6 * (y1 - y0)),
            xytext=(x0 + 0.4 * (x1 - x0), y0 + 0.4 * (y1 - y0)),
            arrowprops={"arrowstyle": "->", "color": "tab:blue", "lw": 1.4},
        )
    ax.scatter(*depot, marker="s", s=110, c="white", edgecolors="black", zorder=4)
    ax.text(0.05, -0.38, "depot 0", fontsize=8)
    offsets = {"a": (-1.05, 0.12), "b": (-0.35, 0.2), "c": (0.15, 0.05), "d": (0.15, -0.28)}
    for key, (x, y, v, s) in cluster.items():
        ax.scatter(x, y, s=70, c="tab:green", edgecolors="black", zorder=3)
        dx, dy = offsets[key]
        ax.text(x + dx, y + dy, f"v={v}, s={s}", fontsize=7.5)
    for x, y, v, s in far.values():
        ax.scatter(x, y, s=70, c="white", edgecolors="0.4", zorder=3)
        ax.text(x + 0.1, y + 0.12, f"v={v}, s={s}", fontsize=7.5, color="0.35")
    ax.plot([0, far["e"][0]], [0, far["e"][1]], ls=":", color="0.55", lw=1)
    ax.plot([0, far["f"][0]], [0, far["f"][1]], ls=":", color="0.55", lw=1)
    ax.text(
        2.6,
        -0.35,
        "selected cluster: reward 15, service 11, short travel",
        fontsize=7.5,
        color="tab:green",
    )
    ax.text(
        3.2,
        1.2,
        "high reward but distant:\nround trip alone uses most of T",
        fontsize=7.5,
        color="0.35",
    )
    ax.set_xlim(-1.8, 6.4)
    ax.set_ylim(-0.6, 3.1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    save(fig, "fig1_problem")


def quality_runtime_figure(results: pd.DataFrame) -> None:
    runtime = pd.read_csv(TABLES / "exact_suite_runtime_by_n.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.45))
    for algorithm in ORDER:
        if algorithm in ("exact_subset_dp", "best_single"):
            continue
        values = np.sort(results.loc[results["algorithm"] == algorithm, "reward_ratio"].to_numpy())
        ecdf = np.arange(1, len(values) + 1) / len(values)
        ax1.step(
            values, ecdf, where="post", color=COLORS[algorithm], label=DISPLAY[algorithm], lw=1.4
        )
    ax1.set_xlim(0.6, 1.005)
    ax1.set_xlabel("ALG/OPT")
    ax1.set_ylabel("fraction of instances with ratio $\\leq x$")
    ax1.set_title("(a) Reward-ratio ECDF, 540 instances")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(alpha=0.3)
    label_nudge = {"service_density_then_route": 0.75, "cheapest_insertion": 1.3}
    for algorithm in ORDER:
        sub = runtime[runtime["algorithm"] == algorithm].sort_values("n")
        ax2.plot(
            sub["n"],
            sub["median_runtime_ms"],
            marker="o",
            ms=3.5,
            color=COLORS[algorithm],
            lw=1.3,
        )
        last = float(sub["median_runtime_ms"].iloc[-1])
        ax2.text(
            16.3,
            last * label_nudge.get(algorithm, 1.0),
            DISPLAY[algorithm],
            fontsize=7,
            color=COLORS[algorithm],
            va="center",
        )
    ax2.set_yscale("log")
    ax2.set_xlim(7.5, 20.5)
    ax2.set_xticks([8, 12, 16])
    ax2.set_xlabel("number of candidate stations n")
    ax2.set_ylabel("median runtime (ms, log scale)")
    ax2.set_title("(b) Median runtime by n")
    ax2.grid(alpha=0.3, which="major")
    fig.tight_layout()
    save(fig, "fig2_quality_runtime")


def alpha_figure(results: pd.DataFrame, trace: pd.DataFrame, case_id: str) -> None:
    work = (
        trace[trace["algorithm"] == "split_only_grid"]
        .groupby("alpha")[["augmentation_additions", "repair_removals"]]
        .mean()
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax1.plot(
        work.index,
        work["augmentation_additions"],
        marker="s",
        ms=3,
        color="tab:green",
        label="stations added by augmentation",
    )
    ax1.plot(
        work.index,
        work["repair_removals"],
        marker="o",
        ms=3,
        color="tab:red",
        label="stations removed by repair",
    )
    ax1.set_xlabel("service fraction $\\alpha$")
    ax1.set_ylabel("mean stations per instance")
    ax1.set_title("(a) Post-processing work across $\\alpha$ (split-only)")
    ax1.legend(
        loc="upper center",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
        fontsize=6.4,
    )
    ax1.grid(alpha=0.3)

    opt = int(
        results[(results["instance_id"] == case_id) & (results["algorithm"] == "exact_subset_dp")][
            "reward"
        ].iloc[0]
    )
    case = trace[trace["instance_id"] == case_id]
    so = case[case["algorithm"] == "split_only_grid"].sort_values("alpha")
    sc = (
        case[case["algorithm"] == "split_compact_grid"]
        .groupby("alpha")["reward"]
        .max()
        .reset_index()
    )
    ax2.step(
        so["alpha"],
        so["reward"],
        where="mid",
        color="tab:blue",
        marker="o",
        ms=3,
        label="Split-only",
    )
    ax2.step(
        sc["alpha"],
        sc["reward"],
        where="mid",
        color="tab:green",
        marker="^",
        ms=3,
        label="Split + compact (best pool)",
    )
    ax2.axhline(opt, color="black", ls=":", lw=1, label=f"OPT = {opt}")
    ax2.set_xlabel("service fraction $\\alpha$")
    ax2.set_ylabel("reward at this $\\alpha$")
    ax2.set_title("(b) Case-study instance: reward across $\\alpha$")
    ax2.legend(loc="lower right", frameon=False, fontsize=6.8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig3_alpha")


def route_figure(case_id: str) -> None:
    with (ROOT / "data/instances/exact_suite/manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        paths = {row["instance_id"]: row["path"] for row in csv.DictReader(handle)}
    instance = read_instance_json(ROOT / paths[case_id])
    solutions = {}
    with (RAW / "exact_suite_solutions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["instance_id"] == case_id:
                solutions[record["algorithm"]] = record
    coords = [(instance.depot_lon, instance.depot_lat)] + [
        (w.lon, w.lat) for w in instance.waypoints
    ]
    kx = 111.32 * math.cos(math.radians(instance.depot_lat))
    ky = 110.54
    xy = [((lon - coords[0][0]) * kx, (lat - coords[0][1]) * ky) for lon, lat in coords]
    rewards = [0] + [w.reward for w in instance.waypoints]
    algorithms = ["exact_subset_dp", "split_only_grid", "split_compact_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.25), sharex=True, sharey=True)
    local = {instance.depot_id.removeprefix("station_"): 0}
    for idx, waypoint in enumerate(instance.waypoints, start=1):
        local[str(waypoint.metadata["station_id"])] = idx
    for ax, algorithm in zip(axes, algorithms, strict=False):
        record = solutions[algorithm]
        route = [local[str(sid)] for sid in record["route_station_ids"]]
        ax.scatter([p[0] for p in xy[1:]], [p[1] for p in xy[1:]], s=12, c="0.75", zorder=1)
        for idx in range(1, len(xy)):
            if xy[idx][1] > -0.28:
                ax.text(
                    xy[idx][0] + 0.03,
                    xy[idx][1] - 0.06,
                    str(rewards[idx]),
                    fontsize=5.8,
                    color="0.45",
                    clip_on=True,
                )
        rx = [xy[i][0] for i in route]
        ry = [xy[i][1] for i in route]
        ax.plot(rx, ry, color=COLORS[algorithm], lw=1.6, zorder=2)
        ax.scatter(
            rx[1:-1], ry[1:-1], s=26, c=COLORS[algorithm], edgecolors="black", lw=0.5, zorder=3
        )
        for order, i in enumerate(route[1:-1], start=1):
            ax.text(
                xy[i][0] - 0.02,
                xy[i][1] + 0.05,
                str(order),
                fontsize=7,
                weight="bold",
                color=COLORS[algorithm],
            )
        ax.scatter(0, 0, marker="s", s=45, c="white", edgecolors="black", zorder=4)
        alpha = record.get("chosen_alpha")
        alpha_text = f", $\\alpha$={alpha:g}" if alpha is not None else ""
        time_text = f"time {record['total_time_s']}/{record['budget_s']} s"
        ax.set_title(
            f"{DISPLAY[algorithm]}\nreward {record['reward']}, {time_text}{alpha_text}",
            fontsize=7.6,
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("east (km)")
        ax.set_ylim(-0.3, 0.5)
        ax.set_xlim(-0.15, 0.8)
        hidden = sum(1 for p in xy[1:] if p[1] < -0.3)
        ax.text(0.02, -0.27, f"+{hidden} candidates further south", fontsize=6, color="0.4")
    axes[0].set_ylabel("north (km)")
    fig.tight_layout()
    save(fig, "fig4_route_case")


def main() -> None:
    results = pd.read_csv(RAW / "exact_suite_results.csv")
    trace = pd.read_parquet(RAW / "exact_suite_split_trace.parquet")
    case_id = str(pd.read_csv(TABLES / "report_compact_gain_cases.csv").iloc[0]["instance_id"])
    problem_figure()
    quality_runtime_figure(results)
    alpha_figure(results, trace, case_id)
    route_figure(case_id)
    print(f"Wrote report figures to {OUT}")


if __name__ == "__main__":
    main()
