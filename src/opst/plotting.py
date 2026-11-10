from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from opst.models import Instance, Solution


def plot_route(instance: Instance, solution: Solution, output_path: str | Path) -> Path:
    """Write a simple static station/route plot for a validated solution."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    depot_xy = (instance.depot_lon, instance.depot_lat)
    waypoint_xy = [(w.lon, w.lat) for w in instance.waypoints]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [xy[0] for xy in waypoint_xy],
        [xy[1] for xy in waypoint_xy],
        c="lightgray",
        s=[20 + 8 * w.reward for w in instance.waypoints],
        edgecolors="black",
        linewidths=0.4,
        label="candidate stations",
    )
    ax.scatter([depot_xy[0]], [depot_xy[1]], marker="s", c="tab:red", s=100, label="depot")

    coords = [depot_xy, *waypoint_xy]
    for order, (a, b) in enumerate(zip(solution.route_indices, solution.route_indices[1:]), start=1):
        xa, ya = coords[a]
        xb, yb = coords[b]
        ax.annotate(
            "",
            xy=(xb, yb),
            xytext=(xa, ya),
            arrowprops={"arrowstyle": "->", "color": "tab:blue", "lw": 1.5},
        )
        ax.text((xa + xb) / 2, (ya + yb) / 2, str(order), color="tab:blue", fontsize=8)

    for idx in solution.selected_indices:
        x, y = coords[idx]
        ax.scatter([x], [y], c="tab:blue", s=70, edgecolors="black", linewidths=0.6)
        ax.text(x, y, f" {idx}", va="bottom", fontsize=8)

    ax.set_title(
        f"{solution.algorithm}: reward {solution.reward}, "
        f"time {solution.total_time_s / 60:.1f}/{instance.budget_s / 60:.1f} min"
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
