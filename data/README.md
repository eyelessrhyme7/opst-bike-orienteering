# Data Notes

Phase 1 used only hand-built and seeded synthetic metric instances in code.

## Oslo City Bike Station Snapshot

The public GBFS station-information fetcher is implemented in `src/opst/data_oslo.py` and exposed by:

```bash
uv run python scripts/fetch_data.py
```

The first successful local fetch wrote:

- raw snapshot: `data/raw/oslo_gbfs/oslo_station_information_20261109T163259Z.json`
- metadata: `data/raw/oslo_gbfs/oslo_station_information_20261109T163259Z.metadata.json`
- normalized CSV: `data/processed/oslo_stations.csv`
- station count: 267
- SHA-256: `6d19326359a54a39ea470c359302709c694573fe84e1b994d7c361ced4f7705f`

Source: Oslo City Bike / Oslo Bysykkel official GBFS open data, `station_information.json`.

License/attribution: Norwegian Licence for Open Government Data (NLOD) 2.0; attribute Oslo City Bike / Oslo Bysykkel official open data.

Later phases will use OSMnx/OpenStreetMap road-network travel times, with graph/matrix metadata, checksums, retrieval dates, and validation results recorded here.

## Road Matrix Status

The OSMnx matrix builder is implemented in `src/opst/road_matrix.py` and exposed by:

```bash
uv run python scripts/build_matrix.py --limit 40
```

It selects a deterministic central sandbox from `data/processed/oslo_stations.csv`, chooses a station nearest the sandbox centroid as the experimental depot, builds directed shortest-path travel times on an OSM drive graph, symmetrizes with `ceil((d_ij + d_ji) / 2)`, validates the symmetric metric, and writes matrix metadata.

Local status: a Python 3.11 `.venv` with OSMnx was created and tests pass in that environment. After installing `scikit-learn` for OSMnx nearest-node lookup, the following matrices were built:

- smoke matrix: `data/processed/oslo_matrix/oslo_sandbox_n4_drive.symmetric_time_s.npy`, 4 candidates plus depot, buffer `0.005` degrees;
- central sandbox matrix: `data/processed/oslo_matrix/oslo_sandbox_n39_drive.symmetric_time_s.npy`, 39 candidates plus depot, buffer `0.01` degrees.

The central sandbox metadata is `data/processed/oslo_matrix/oslo_sandbox_n39_drive.metadata.json`, and the station/depot manifest is `data/processed/oslo_matrix/oslo_sandbox_n39_drive.stations.csv`. A few nearby stations snap to the same OSM node; this is recorded in metadata through `snapped_osm_node_ids`.

## Generated Smoke Instances

The first smoke suite was generated with:

```bash
.venv\Scripts\python scripts\generate_instances.py --n-values 8 --service-scales 1.0 --seeds 1 2 3
```

It wrote 36 JSON instances under `data/instances/smoke/`:

- matrix: `oslo_sandbox_n39_drive`;
- `n = 8`;
- regimes: `compact`, `dispersed`, `mixed`, `outlier`;
- service scale: `1.0`;
- budget tightness: `tight`, `medium`, `loose`;
- seeds: `1`, `2`, `3`.

These are synthetic task states on real Oslo station geography and OSM-derived travel times. They must not be described as observed maintenance demand.

## Exact-Small Development Suite

The current exact-small suite was generated with:

```bash
.venv\Scripts\python scripts\generate_instances.py --output-dir data/instances/exact_small --n-values 8 12 --service-scales 0.5 1.0 2.0 --seeds 1 2
```

It wrote 144 JSON instances under `data/instances/exact_small/`, using the same real Oslo sandbox matrix and synthetic task-state model.

## Full MVP Exact Suite

The full MVP exact-comparison suite was generated with:

```bash
.venv\Scripts\python scripts\generate_instances.py --output-dir data/instances/exact_suite --n-values 8 12 16 --service-scales 0.5 1.0 2.0 --seeds 1 2 3 4 5
```

It wrote 540 JSON instances under `data/instances/exact_suite/`, covering the documented MVP grid of 3 sizes, 4 regimes, 3 service scales, 3 budget levels, and 5 seeds.

Do not store secrets in this directory. Optional API tokens belong in an untracked `.env` file.
