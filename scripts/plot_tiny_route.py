from __future__ import annotations

from pathlib import Path

from opst.exact_dp import solve_exact_subset_dp
from opst.instances import tiny_metric_instance
from opst.plotting import plot_route


def main() -> None:
    instance = tiny_metric_instance()
    solution = solve_exact_subset_dp(instance)
    output = plot_route(instance, solution, Path("results/figures/tiny_route.png"))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
