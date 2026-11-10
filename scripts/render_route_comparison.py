from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from opst.instance_generation import read_instance_json

COLORS = {
    "exact_subset_dp": "red",
    "split_only_grid": "dodgerblue",
    "split_compact_grid": "limegreen",
    "cheapest_insertion": "gold",
    "service_density_then_route": "magenta",
    "best_single": "white",
}


def offset_line(
    xs: list[float], ys: list[float], offset_m: float
) -> tuple[list[float], list[float]]:
    offset_xs: list[float] = []
    offset_ys: list[float] = []
    for index, (x, y) in enumerate(zip(xs, ys, strict=False)):
        if index == 0:
            next_x, next_y = xs[1], ys[1]
            prev_x, prev_y = x, y
        elif index == len(xs) - 1:
            next_x, next_y = x, y
            prev_x, prev_y = xs[index - 1], ys[index - 1]
        else:
            next_x, next_y = xs[index + 1], ys[index + 1]
            prev_x, prev_y = xs[index - 1], ys[index - 1]
        lat_scale = 110_540.0
        lon_scale = 111_320.0 * max(0.1, abs(math.cos(math.radians(y))))
        dx = (next_x - prev_x) * lon_scale
        dy = (next_y - prev_y) * lat_scale
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            offset_xs.append(x)
            offset_ys.append(y)
            continue
        nx = -dy / length
        ny = dx / length
        offset_xs.append(x + nx * offset_m / lon_scale)
        offset_ys.append(y + ny * offset_m / lat_scale)
    return offset_xs, offset_ys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render saved route comparisons.")
    parser.add_argument("--instance-id", default=None)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["exact_subset_dp", "split_only_grid", "split_compact_grid"],
    )
    parser.add_argument("--auto-cases", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/routes"))
    return parser.parse_args()


def load_manifest() -> dict[str, str]:
    with Path("data/instances/exact_suite/manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return {row["instance_id"]: row["path"] for row in csv.DictReader(handle)}


def load_solutions() -> dict[str, dict[str, object]]:
    output = {}
    with Path("results/raw/exact_suite_solutions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            output[f"{record['instance_id']}::{record['algorithm']}"] = record
    return output


def station_lookup(instance_id: str, manifest: dict[str, str]) -> dict[str, dict[str, object]]:
    instance = read_instance_json(Path(manifest[instance_id]))
    rows = {
        instance.depot_id.removeprefix("station_"): {
            "lat": instance.depot_lat,
            "lon": instance.depot_lon,
            "name": "DEPOT",
            "reward": 0,
            "service_time_s": 0,
        }
    }
    for waypoint in instance.waypoints:
        station_id = str(waypoint.metadata.get("station_id"))
        rows[station_id] = {
            "lat": waypoint.lat,
            "lon": waypoint.lon,
            "name": waypoint.metadata.get("name", waypoint.waypoint_id),
            "reward": waypoint.reward,
            "service_time_s": waypoint.service_time_s,
            "h3_cell": waypoint.h3_cell,
        }
    return rows


def render(instance_id: str, algorithms: list[str], output_dir: Path) -> None:
    manifest = load_manifest()
    solutions = load_solutions()
    stations = station_lookup(instance_id, manifest)
    fig, ax = plt.subplots(figsize=(8, 6))
    candidate_ids = [
        station_id for station_id in stations if stations[station_id]["name"] != "DEPOT"
    ]
    ax.scatter(
        [stations[station_id]["lon"] for station_id in candidate_ids],
        [stations[station_id]["lat"] for station_id in candidate_ids],
        c="lightgray",
        s=16,
        label="candidate stations",
        zorder=1,
    )
    depot_id = next(
        station_id for station_id, station in stations.items() if station["name"] == "DEPOT"
    )
    ax.scatter(
        [stations[depot_id]["lon"]],
        [stations[depot_id]["lat"]],
        marker="s",
        c="cyan",
        s=120,
        label="depot",
        zorder=4,
    )
    caption_parts = []
    offset_step_m = 14.0
    algorithm_offsets = {
        algorithm: (index - (len(algorithms) - 1) / 2.0) * offset_step_m
        for index, algorithm in enumerate(algorithms)
    }
    for algorithm in algorithms:
        record = solutions[f"{instance_id}::{algorithm}"]
        route = [str(station_id) for station_id in record["route_station_ids"]]
        xs = [stations[station_id]["lon"] for station_id in route]
        ys = [stations[station_id]["lat"] for station_id in route]
        plot_xs, plot_ys = offset_line(xs, ys, algorithm_offsets[algorithm])
        ax.plot(
            plot_xs,
            plot_ys,
            color=COLORS[algorithm],
            linewidth=3,
            marker="o",
            markersize=4,
            label=algorithm,
            zorder=3,
        )
        for order, station_id in enumerate(route[1:-1], start=1):
            ax.text(
                stations[station_id]["lon"],
                stations[station_id]["lat"],
                str(order),
                fontsize=7,
                color=COLORS[algorithm],
            )
        caption_parts.append(
            f"{algorithm}: reward {record['reward']}, ratio {record['reward_ratio']:.3f}, "
            f"time {record['total_time_s']}/{record['budget_s']}s"
        )
    ax.set_title(f"Route comparison: {instance_id}\n" + " | ".join(caption_parts), fontsize=9)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(fontsize=8)
    ax.text(
        0.01,
        0.01,
        "Displayed lines connect station coordinates; optimization used "
        "symmetrized OSM travel-time matrix.",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = instance_id.replace("/", "_")
    png = output_dir / f"{safe_name}.png"
    svg = output_dir / f"{safe_name}.svg"
    fig.tight_layout()
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {svg}")


def auto_cases() -> list[str]:
    results = pd.read_csv("results/raw/exact_suite_results.csv")
    paired = pd.read_csv("results/tables/exact_suite_paired_improvement_rows.csv")
    worst_split = str(
        results[results["algorithm"] == "split_only_grid"]
        .sort_values("reward_ratio")
        .iloc[0]["instance_id"]
    )
    compact_gain = str(
        paired[paired["comparison"] == "split_only_to_split_compact"]
        .sort_values("delta_reward_ratio", ascending=False)
        .iloc[0]["instance_id"]
    )
    compact = str(
        results[
            (results["regime"] == "compact") & (results["algorithm"] == "exact_subset_dp")
        ].iloc[0]["instance_id"]
    )
    outlier = str(
        results[(results["regime"] == "outlier") & (results["algorithm"] == "split_compact_grid")]
        .sort_values("reward_ratio")
        .iloc[0]["instance_id"]
    )
    return list(dict.fromkeys([worst_split, compact_gain, compact, outlier]))


def main() -> None:
    args = parse_args()
    if args.auto_cases:
        for instance_id in auto_cases():
            render(instance_id, args.algorithms, args.output_dir)
    elif args.instance_id:
        render(args.instance_id, args.algorithms, args.output_dir)
    else:
        raise SystemExit("Provide --instance-id or --auto-cases")


if __name__ == "__main__":
    main()
