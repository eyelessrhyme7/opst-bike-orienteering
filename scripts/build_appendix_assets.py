"""Build appendix figures and tables from frozen Phase 3 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
TABLES = ROOT / "results/tables"
INSTANCES = ROOT / "data/instances/exact_suite"
MATRIX = ROOT / "data/processed/oslo_matrix/oslo_sandbox_n39_drive"
OUT = ROOT / "report/assets/figures"
PREFIX = "oslo_sandbox_n39_drive_"

ALGS = [
    "service_density_then_route",
    "cheapest_insertion",
    "split_only_grid",
    "split_compact_grid",
]
DISPLAY = {
    "best_single": "Best single",
    "service_density_then_route": "Density then route",
    "cheapest_insertion": "Cheapest insertion",
    "split_only_grid": "Split-only",
    "split_compact_grid": "Split + compact",
    "exact_subset_dp": "Exact DP",
}
COLORS = {
    "service_density_then_route": "tab:purple",
    "cheapest_insertion": "tab:orange",
    "split_only_grid": "tab:blue",
    "split_compact_grid": "tab:green",
    "exact_subset_dp": "black",
}
REGIMES = ["compact", "mixed", "dispersed", "outlier"]
BUDGETS = ["tight", "medium", "loose"]
SCALES = [0.5, 1.0, 2.0]

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


def load_instances() -> pd.DataFrame:
    rows = []
    for path in sorted(INSTANCES.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload["metadata"]
        for wp in payload["waypoints"]:
            rows.append(
                {
                    "instance_id": payload["instance_id"],
                    "n": len(payload["waypoints"]),
                    "regime": meta["spatial_regime"],
                    "service_scale": float(meta["service_scale"]),
                    "budget_tightness": meta["budget_tightness"],
                    "budget_s": payload["budget_s"],
                    "all_route_total_s": meta["all_route_total_s"],
                    "station_id": str(wp["metadata"]["station_id"]),
                    "reward": wp["reward"],
                    "service_time_s": wp["service_time_s"],
                    "task_count": wp["metadata"]["task_count"],
                    "urgency": wp["metadata"]["urgency"],
                    "h3_cell": wp["h3_cell"],
                }
            )
    return pd.DataFrame(rows)


def load_solutions() -> pd.DataFrame:
    with (RAW / "exact_suite_solutions.jsonl").open(encoding="utf-8") as handle:
        return pd.DataFrame(json.loads(line) for line in handle)


def dataset_figure(stations: pd.DataFrame, directed: np.ndarray, sym: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    cells = sorted(
        {
            h3.latlng_to_cell(lat, lon, 8)
            for lat, lon in zip(stations.lat, stations.lon, strict=True)
        }
    )
    cmap = plt.get_cmap("tab10")
    cell_colors = {cell: cmap(i % 10) for i, cell in enumerate(cells)}
    for cell in cells:
        boundary = np.array(h3.cell_to_boundary(cell))
        ax.fill(boundary[:, 1], boundary[:, 0], color=cell_colors[cell], alpha=0.14, lw=0)
        ax.plot(
            np.r_[boundary[:, 1], boundary[0, 1]],
            np.r_[boundary[:, 0], boundary[0, 0]],
            color=cell_colors[cell],
            lw=0.7,
        )
    depot = stations.iloc[0]
    rest = stations.iloc[1:]
    ax.scatter(
        rest.lon,
        rest.lat,
        s=12,
        c=[
            cell_colors[h3.latlng_to_cell(a, b, 8)] for a, b in zip(rest.lat, rest.lon, strict=True)
        ],
        edgecolor="black",
        lw=0.3,
        zorder=3,
    )
    ax.scatter(depot.lon, depot.lat, marker="s", s=34, c="cyan", edgecolor="black", zorder=4)
    ax.set_title("(a) Sandbox stations and H3 res-8 cells")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(stations.lat.mean())))
    ax.tick_params(labelsize=6.5)

    iu = np.triu_indices_from(sym, k=1)
    axes[1].hist(sym[iu], bins=30, color="tab:blue", alpha=0.85)
    axes[1].axvline(np.median(sym[iu]), color="black", lw=0.8, ls="--")
    axes[1].set_title("(b) Symmetric travel times")
    axes[1].set_xlabel("Seconds")
    axes[1].set_ylabel("Station pairs")

    asym = np.abs(directed - directed.T)[iu]
    axes[2].hist(asym, bins=np.arange(0, asym.max() + 6, 5), color="tab:orange", alpha=0.85)
    axes[2].axvline(np.median(asym), color="black", lw=0.8, ls="--")
    axes[2].set_title("(c) Directional asymmetry")
    axes[2].set_xlabel(r"$|d_{ij}-d_{ji}|$ (s)")
    fig.tight_layout()
    save(fig, "figB1_dataset")


def regimes_figure(stations: pd.DataFrame, inst: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05), sharex=True, sharey=True)
    depot = stations.iloc[0]
    for ax, regime in zip(axes, REGIMES, strict=True):
        iid = f"{PREFIX}{regime}_n12_scale1_medium_seed1"
        rows = inst[inst.instance_id == iid].merge(
            stations[["station_id", "lat", "lon"]], on="station_id"
        )
        ax.scatter(stations.lon[1:], stations.lat[1:], s=5, c="0.82", zorder=1)
        sc = ax.scatter(
            rows.lon,
            rows.lat,
            s=8 + 5 * rows.reward,
            c=rows.service_time_s,
            cmap="viridis",
            vmin=200,
            vmax=900,
            edgecolor="black",
            lw=0.3,
            zorder=3,
        )
        ax.scatter(depot.lon, depot.lat, marker="s", s=26, c="cyan", edgecolor="black", zorder=4)
        ax.set_title(regime.capitalize())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1 / np.cos(np.radians(stations.lat.mean())))
    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("Service time (s)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    save(fig, "figB2_regimes")


def heatmap_figure(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.2), sharey=True)
    for ax, alg in zip(
        axes, ["cheapest_insertion", "split_only_grid", "split_compact_grid"], strict=True
    ):
        sub = results[results.algorithm == alg]
        grid = (
            sub.assign(match=sub.reward_ratio >= 1 - 1e-12)
            .pivot_table(index="regime", columns="budget_tightness", values="match", aggfunc="mean")
            .loc[REGIMES, BUDGETS]
            * 100
        )
        ax.imshow(grid.values, cmap="Blues", vmin=40, vmax=100)
        for i in range(len(REGIMES)):
            for j in range(len(BUDGETS)):
                value = grid.values[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value > 82 else "black",
                )
        ax.set_xticks(range(3), BUDGETS)
        ax.set_yticks(range(4), REGIMES)
        ax.set_title(DISPLAY[alg])
    fig.suptitle("Exact-match rate (%) by spatial regime and budget tightness", fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, "figC1_match_heatmaps")


def slack_figure(solutions: pd.DataFrame, opt_count: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25))
    algs = [*ALGS, "exact_subset_dp"]
    data = [
        100
        * solutions.loc[solutions.algorithm == a, "unused_budget_s"].to_numpy()
        / solutions.loc[solutions.algorithm == a, "budget_s"].to_numpy()
        for a in algs
    ]
    box = axes[0].boxplot(
        data, orientation="horizontal", patch_artist=True, widths=0.6, showfliers=False
    )
    for patch, alg in zip(box["boxes"], algs, strict=True):
        patch.set_facecolor(COLORS[alg])
        patch.set_alpha(0.55)
    for median in box["medians"]:
        median.set_color("black")
    axes[0].set_yticks(range(1, len(algs) + 1), [DISPLAY[a] for a in algs])
    axes[0].set_xlabel("Unused budget (% of T)")
    axes[0].set_title("(a) Budget slack of returned routes")

    diffs = range(-3, 2)
    width = 0.2
    for k, alg in enumerate(ALGS):
        sub = solutions[solutions.algorithm == alg].set_index("instance_id")
        delta = (sub.selected_count - opt_count.loc[sub.index]).clip(-3, 1)
        counts = delta.value_counts().reindex(list(diffs), fill_value=0)
        axes[1].bar(
            np.array(list(diffs)) + (k - 1.5) * width,
            counts.values,
            width=width,
            color=COLORS[alg],
            label=DISPLAY[alg],
        )
    axes[1].set_yscale("log")
    axes[1].set_xticks(list(diffs), ["≤−3", "−2", "−1", "0", "≥+1"])
    axes[1].set_xlabel("Stations visited minus OPT stations")
    axes[1].set_ylabel("Instances")
    axes[1].set_title("(b) Visit-count difference to OPT")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "figC2_slack_counts")


def pick_trace_cases(trace: pd.DataFrame) -> list[tuple[str, str]]:
    so = trace[trace.algorithm == "split_only_grid"]
    stats = so.groupby("instance_id").agg(
        levels=("reward", "nunique"),
        ties=("is_reward_tied_with_final", "sum"),
        lo=("reward_ratio", "min"),
    )
    by_alpha = so.pivot_table(index="instance_id", columns="alpha", values="reward_ratio")
    alphas = by_alpha.columns.to_numpy()
    best_alpha = by_alpha.apply(lambda r: alphas[r.to_numpy() >= r.max() - 1e-12], axis=1)
    only_low = [i for i, a in best_alpha.items() if a.max() <= 0.45]
    only_high = [i for i, a in best_alpha.items() if a.min() >= 0.70]
    cases = [
        (f"{PREFIX}compact_n12_scale0.5_tight_seed4", "Compactness recovers OPT"),
        (
            stats.loc[only_low].sort_values(["lo", "levels"]).index[0],
            "Split-only: only small α reach OPT",
        ),
        (
            stats.loc[only_high].sort_values(["lo", "levels"]).index[0],
            "Split-only: only large α reach OPT",
        ),
        (
            stats.sort_values(["levels", "lo"], ascending=[False, True]).index[0],
            "Many reward levels",
        ),
        (stats[stats.ties == 1].sort_values("lo").index[0], "Split-only: unique best α"),
        (f"{PREFIX}dispersed_n16_scale1_medium_seed4", "Neither variant reaches OPT"),
    ]
    return cases


def short_label(instance_id: str) -> str:
    parts = instance_id.removeprefix(PREFIX).split("_")
    regime, n, scale, budget, seed = parts
    return f"{regime}, n={n[1:]}, σ={scale[5:]}, {budget}, s{seed[4:]}"


def trace_figure(trace: pd.DataFrame) -> list[tuple[str, str]]:
    cases = pick_trace_cases(trace)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharex=True)
    for ax, (iid, title) in zip(axes.flat, cases, strict=True):
        sub = trace[trace.instance_id == iid]
        so = sub[sub.algorithm == "split_only_grid"].sort_values("alpha")
        sc = sub[sub.algorithm == "split_compact_grid"].groupby("alpha").reward_ratio.max()
        ax.plot(
            so.alpha,
            so.reward_ratio,
            "-o",
            ms=2.8,
            color=COLORS["split_only_grid"],
            label="Split-only",
        )
        ax.plot(
            sc.index,
            sc.values,
            "--s",
            ms=2.4,
            color=COLORS["split_compact_grid"],
            label="Split + compact (best pool)",
        )
        ax.axhline(1.0, color="black", lw=0.6, ls=":")
        ax.set_title(f"{title}\n{short_label(iid)}", fontsize=7.6)
        ax.tick_params(labelsize=6.8)
    for ax in axes[1]:
        ax.set_xlabel(r"Service share $\alpha$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reward / OPT")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "figD1_alpha_traces")
    return cases


def mechanism_figure(trace: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    so = trace[trace.algorithm == "split_only_grid"].merge(
        results[results.algorithm == "split_only_grid"][
            ["instance_id", "service_scale", "budget_tightness"]
        ],
        on="instance_id",
    )
    so["is_opt"] = so.reward_ratio >= 1 - 1e-12
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25))
    for scale, color in zip(SCALES, ["tab:cyan", "tab:blue", "navy"], strict=True):
        line = so[so.service_scale == scale].groupby("alpha").is_opt.mean() * 100
        axes[0].plot(line.index, line.values, "-o", ms=2.5, color=color, label=f"σ = {scale:g}")
    axes[0].set_title("(a) OPT reached at α, by service scale")
    axes[0].set_ylabel("Instances reaching OPT (%)")
    axes[0].legend(frameon=False, fontsize=6.5)
    for budget, color in zip(BUDGETS, ["tab:red", "tab:orange", "goldenrod"], strict=True):
        line = so[so.budget_tightness == budget].groupby("alpha").is_opt.mean() * 100
        axes[1].plot(line.index, line.values, "-o", ms=2.5, color=color, label=budget)
    axes[1].set_title("(b) OPT reached at α, by budget")
    axes[1].legend(frameon=False, fontsize=6.5)
    mech = so.groupby("alpha")[["repair_removals", "augmentation_additions"]].mean()
    so = so.merge(results[["instance_id", "regime"]].drop_duplicates(), on="instance_id")
    for regime, color in zip(
        REGIMES, ["tab:green", "tab:olive", "tab:brown", "tab:pink"], strict=True
    ):
        line = so[so.regime == regime].groupby("alpha").reward_ratio.mean()
        axes[2].plot(line.index, line.values, "-o", ms=2.5, color=color, label=regime)
    axes[2].set_title("(c) Mean ALG/OPT at α, by regime")
    axes[2].set_ylabel("Mean reward / OPT")
    axes[2].legend(frameon=False, fontsize=6.5)
    for ax in axes:
        ax.set_xlabel(r"Service share $\alpha$")
    fig.tight_layout()
    save(fig, "figD2_alpha_mechanism")
    return mech


def ties_figure(trace: pd.DataFrame) -> pd.Series:
    so = trace[trace.algorithm == "split_only_grid"]
    ties = so.groupby("instance_id").is_reward_tied_with_final.sum()
    levels = so.groupby("instance_id").reward.nunique()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0))
    axes[0].hist(ties, bins=np.arange(0.5, 16.5, 1), color="tab:blue", alpha=0.85)
    axes[0].set_xlabel("α values tied with the final reward (of 15)")
    axes[0].set_ylabel("Instances")
    axes[0].set_title("(a) Width of the best-α set")
    axes[1].hist(levels, bins=np.arange(0.5, levels.max() + 1.5, 1), color="tab:purple", alpha=0.85)
    axes[1].set_xlabel("Distinct rewards across the α grid")
    axes[1].set_title("(b) Reward levels visited by the grid")
    fig.tight_layout()
    save(fig, "figD3_alpha_ties")
    return ties


def main() -> None:
    stations = pd.read_csv(f"{MATRIX}.stations.csv", dtype={"station_id": str})
    directed = np.load(f"{MATRIX}.directed_time_s.npy")
    sym = np.load(f"{MATRIX}.symmetric_time_s.npy")
    inst = load_instances()
    results = pd.read_csv(RAW / "exact_suite_results.csv")
    solutions = load_solutions()
    trace = pd.read_parquet(RAW / "exact_suite_split_trace.parquet")

    dataset_figure(stations, directed, sym)
    regimes_figure(stations, inst)
    heatmap_figure(results)
    exact = solutions[solutions.algorithm == "exact_subset_dp"].set_index("instance_id")
    slack_figure(solutions, exact.selected_count)
    cases = trace_figure(trace)
    mech = mechanism_figure(trace, results)
    ties = ties_figure(trace)

    per_instance = inst.groupby("instance_id").agg(
        n=("n", "first"),
        service_scale=("service_scale", "first"),
        budget_tightness=("budget_tightness", "first"),
        budget_s=("budget_s", "first"),
        all_route_total_s=("all_route_total_s", "first"),
        mean_service_s=("service_time_s", "mean"),
        mean_reward=("reward", "mean"),
    )
    per_instance = per_instance.join(
        exact[["selected_count", "service_time_s", "total_time_s", "unused_budget_s"]]
    )
    per_instance["opt_service_share"] = per_instance.service_time_s / per_instance.total_time_s
    per_instance["opt_visit_share"] = per_instance.selected_count / per_instance.n
    rows = []
    for field, values in [
        ("service_scale", SCALES),
        ("budget_tightness", BUDGETS),
        ("n", [8, 12, 16]),
    ]:
        for value in values:
            sub = per_instance[per_instance[field] == value]
            rows.append(
                {
                    "group": f"{field}={value:g}"
                    if field != "budget_tightness"
                    else f"budget={value}",
                    "instances": len(sub),
                    "mean_service_min": sub.mean_service_s.mean() / 60,
                    "mean_reward": sub.mean_reward.mean(),
                    "mean_budget_min": sub.budget_s.mean() / 60,
                    "opt_visit_share": sub.opt_visit_share.mean(),
                    "opt_service_share": sub.opt_service_share.mean(),
                    "opt_unused_s": sub.unused_budget_s.mean(),
                }
            )
    instance_table = pd.DataFrame(rows)
    instance_table.to_csv(TABLES / "appendix_instance_profile.csv", index=False)

    results["match"] = results.reward_ratio >= 1 - 1e-12
    breakdown = []
    for field, values in [("service_scale", SCALES), ("budget_tightness", BUDGETS)]:
        for value in values:
            sub = results[results[field] == value]
            row = {"group": f"{field}={value}"}
            for alg in ALGS:
                a = sub[sub.algorithm == alg]
                row[f"{alg}_mean"] = a.reward_ratio.mean()
                row[f"{alg}_min"] = a.reward_ratio.min()
                row[f"{alg}_match"] = a.match.mean()
            breakdown.append(row)
    pd.DataFrame(breakdown).to_csv(TABLES / "appendix_breakdown.csv", index=False)

    print(instance_table.round(3).to_string())
    print(pd.DataFrame(breakdown).round(3).T.to_string())
    print("cases", cases)
    print(mech.round(3).to_string())
    print(
        "ties",
        ties.describe().round(2).to_dict(),
        "unique",
        int((ties == 1).sum()),
        "all15",
        int((ties == 15).sum()),
    )
    iu = np.triu_indices_from(sym, k=1)
    asym = np.abs(directed - directed.T)[iu]
    print(
        "sym",
        sym[iu].min(),
        np.median(sym[iu]),
        sym[iu].max(),
        "asym median",
        np.median(asym),
        "p90",
        np.percentile(asym, 90),
        "max",
        asym.max(),
    )
    print("cells", inst.h3_cell.nunique(), "stations", len(stations))
    print(
        "task_count dist",
        inst.drop_duplicates(["instance_id", "station_id"])
        .task_count.value_counts(normalize=True)
        .sort_index()
        .round(3)
        .to_dict(),
    )
    print(
        "reward range",
        inst.reward.min(),
        inst.reward.max(),
        "service range",
        inst.service_time_s.min(),
        inst.service_time_s.max(),
    )
    print(
        "budget floor binding",
        int(
            (
                per_instance.budget_s
                > np.ceil(
                    per_instance.all_route_total_s
                    * per_instance.budget_tightness.map(
                        {"tight": 0.35, "medium": 0.5, "loose": 0.65}
                    )
                )
            ).sum()
        ),
    )
    sim = pd.read_csv(TABLES / "solution_similarity_summary.csv")
    print(sim.round(3).to_string())
    for name in ["mean", "median"]:
        print(name, solutions.groupby("algorithm").unused_budget_s.agg(name).round(1).to_dict())


if __name__ == "__main__":
    main()
