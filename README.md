# Budget Split Meets Geography

**Heuristics for Orienteering with Service Times in Bike-Share Maintenance**
CS4234 Optimisation Algorithms · AY2026/27 Semester 1

Gupta Raman · Koh Jun Hong Nicolas · Terence Wong Wei Wen · Amy Ziqi Kang

- Report: [`report/report.pdf`](report/report.pdf)
- Interactive solution explorer: [opst-explorer-cs4234.streamlit.app](https://opst-explorer-cs4234.streamlit.app/)

## Problem

One maintenance truck leaves a depot, visits a subset of bike stations and returns within a hard time budget. Each station has a reward and a fixed on-site service time, so the budget is shared between driving and servicing. This is the Orienteering Problem with service times (OP-ST): a 0/1 knapsack (which stations) coupled with a travelling salesperson problem (in what order).

The study uses real public Oslo City Bike station geography, an OpenStreetMap road travel-time matrix and seeded synthetic task states, and compares:

| Algorithm | Code name | Idea |
| --- | --- | --- |
| Exact subset DP | `exact_subset_dp` | Held–Karp style DP, \(O(n^2 2^n)\); ground truth |
| Best single station | `best_single` | Highest-reward single out-and-back visit |
| Service density | `service_density_then_route` | Rank by reward per service second, then route |
| Cheapest insertion | `cheapest_insertion` | Travel-aware greedy insertion |
| **Split-only** | `split_only_grid` | Reserve a fraction α of the budget for service, solve the service knapsack exactly, route, repair, augment; search α ∈ {0.20, 0.25, …, 0.90} |
| Split + compactness | `split_compact_grid` | Split-only restricted to H3 spatial pools (global, cell, cell + one ring) |

## Headline results (540 instances, n ∈ {8, 12, 16})

All 3,240 solutions are feasible and passed the shared validator. Ratios are empirical ALG/OPT against the exact DP, not worst-case guarantees.

| Algorithm | Mean ALG/OPT | Min ALG/OPT | Exact matches | Median runtime |
| --- | --- | --- | --- | --- |
| Service density | 0.966 | 0.667 | 53.1% | 0.31 ms |
| Cheapest insertion | 0.986 | 0.600 | 71.7% | 0.35 ms |
| **Split-only** | **0.998** | **0.917** | **92.8%** | **8.0 ms** |
| Split + compactness | 0.999 | 0.964 | 95.2% | 31.2 ms |
| Exact subset DP | 1.000 | 1.000 | 100% | 110.7 ms |

Source: [`results/tables/exact_suite_core_summary.csv`](results/tables/exact_suite_core_summary.csv).

## Repository layout

```
src/opst/          core library: Instance/Solution models, validator, exact DP, heuristics, data and matrix builders
scripts/           command-line entry points for data, instances, experiments, tables and figures
tests/             pytest suite on small hand-checkable instances
app/               Streamlit solution explorer (view-only; reads saved results)
data/raw/          cached public GBFS station snapshot with retrieval metadata and licence
data/processed/    normalised stations and the directed + symmetrised road travel-time matrices
data/instances/    generated, seeded instance suites with manifests (smoke, exact_small, exact_suite, reward_sensitivity)
results/raw/       per-solution result rows, final routes (JSONL) and split/compact traces (Parquet)
results/tables/    summary and diagnostic tables
results/figures/   diagnostic plots
results/manifests/ run provenance (seeds, versions, timestamps)
report/            final report PDF, report figures and explorer screenshots
```

Conventions: all times are integer seconds internally; matrix index 0 is the depot and indices 1..n follow `Instance.waypoints`. The exact solver uses the half-service edge transform \(d'_{ij} = d_{ij} + (s_i + s_j)/2\), stored as doubled integers so odd service times do not introduce rounding.

## Setup

Python 3.11+ with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
```

Without uv, create a virtual environment and `pip install -e .` plus `pytest`.

## Reproduction

The generated instances and all results are committed, so every table and figure can be rebuilt from saved data without re-running solvers. To regenerate everything from scratch:

```bash
# 1. Public data: Oslo City Bike GBFS snapshot and OSMnx road matrix (network access)
uv run python scripts/fetch_data.py
uv run python scripts/build_matrix.py --limit 40

# 2. Seeded 540-instance exact suite
uv run python scripts/generate_instances.py --output-dir data/instances/exact_suite --n-values 8 12 16 --service-scales 0.5 1.0 2.0 --seeds 1 2 3 4 5

# 3. Run all six algorithms with validation and instrumentation
uv run python scripts/run_experiments.py --manifest data/instances/exact_suite/manifest.csv --output results/raw/exact_suite_results.csv --run-id exact_suite --instrument

# 4. Tables, diagnostics and figures from saved results
uv run python scripts/build_report_assets.py --results results/raw/exact_suite_results.csv --prefix exact_suite --summary-md results/EXACT_SUITE_SUMMARY.md
uv run python scripts/build_diagnostics.py
uv run python scripts/build_diagnostic_plots.py
uv run python scripts/render_route_comparison.py --auto-cases
uv run python scripts/build_report_stats.py
PYTHONPATH=src uv run python scripts/build_report_figures.py
PYTHONPATH=src uv run python scripts/build_appendix_assets.py
```

A road matrix rebuilt today can differ slightly from the committed one because OpenStreetMap changes; use the committed `data/processed/` files to reproduce the reported numbers exactly.

Smaller suites for quick checks:

```bash
uv run python scripts/generate_instances.py --n-values 8 --service-scales 1.0 --seeds 1 2 3
uv run python scripts/run_experiments.py --run-id smoke
uv run python scripts/build_report_assets.py
```

## Solution explorer

Hosted: [opst-explorer-cs4234.streamlit.app](https://opst-explorer-cs4234.streamlit.app/)

Locally:

```bash
uv run streamlit run app/solution_explorer.py
```

The explorer filters the 540 instances by size, spatial regime, service scale, budget and seed, overlays the routes of any algorithms on the map and shows their metrics. It accepts URL parameters, for example `?instance_id=<id>&algorithms=exact_subset_dp,split_only_grid`. Route segments connect station coordinates for display; optimisation used road travel times. `app/requirements.txt` holds the lightweight dependencies used by the hosted app.

## Data and licences

- Station geography: Oslo City Bike / Oslo Bysykkel public GBFS feed, Norwegian Licence for Open Government Data (NLOD) 2.0.
- Road network and travel times: © OpenStreetMap contributors, ODbL, via OSMnx.
- Rewards and service times are synthetic, generated with fixed seeds; they are not observed operator data.

No private operator exports or proprietary distance matrices are used. No API keys are required; see `.env.example` for optional variable names.
