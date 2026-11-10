from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from opst.road_matrix import build_osmnx_directed_matrix_s, write_matrix_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an OSMnx road travel-time matrix for a small Oslo station sandbox."
    )
    parser.add_argument(
        "--stations-csv",
        type=Path,
        default=Path("data/processed/oslo_stations.csv"),
        help="Normalized station CSV created by scripts/fetch_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/oslo_matrix"),
        help="Directory for matrix files and metadata.",
    )
    parser.add_argument("--limit", type=int, default=40, help="Number of station rows to keep.")
    parser.add_argument("--network-type", default="drive", help="OSMnx network type.")
    parser.add_argument(
        "--buffer-degrees",
        type=float,
        default=0.02,
        help="Bounding-box buffer around selected stations, in degrees.",
    )
    parser.add_argument(
        "--overpass-url",
        default=None,
        help="Optional alternate Overpass interpreter URL for OSMnx.",
    )
    return parser.parse_args()


def read_station_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_central_sandbox(
    rows: list[dict[str, Any]], *, limit: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose a deterministic central station depot and nearby candidate stations."""

    if limit < 2:
        raise ValueError("limit must be at least 2 so one depot and one candidate exist")
    if len(rows) < limit:
        raise ValueError(f"requested {limit} stations but only {len(rows)} rows are available")

    parsed = [
        {
            **row,
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "capacity": int(row.get("capacity", 0) or 0),
        }
        for row in rows
    ]
    center_lat = sum(row["lat"] for row in parsed) / len(parsed)
    center_lon = sum(row["lon"] for row in parsed) / len(parsed)

    def distance_key(row: dict[str, Any]) -> tuple[float, str]:
        squared = (row["lat"] - center_lat) ** 2 + (row["lon"] - center_lon) ** 2
        return squared, str(row["station_id"])

    selected = sorted(parsed, key=distance_key)[:limit]
    depot = min(selected, key=distance_key)
    candidates = [row for row in selected if row["station_id"] != depot["station_id"]]
    return depot, candidates


def write_manifest(path: Path, depot: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "matrix_index": 0,
            "role": "depot",
            **depot,
        }
    ]
    rows.extend(
        {
            "matrix_index": index,
            "role": "candidate",
            **row,
        }
        for index, row in enumerate(candidates, start=1)
    )
    fieldnames = [
        "matrix_index",
        "role",
        "station_id",
        "name",
        "lat",
        "lon",
        "capacity",
        "source_snapshot",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_station_rows(args.stations_csv)
    depot, candidates = select_central_sandbox(rows, limit=args.limit)
    coordinates = [(depot["lat"], depot["lon"])] + [
        (row["lat"], row["lon"]) for row in candidates
    ]
    matrix_id = f"oslo_sandbox_n{len(candidates)}_{args.network_type}"

    try:
        directed, snapped_nodes = build_osmnx_directed_matrix_s(
            coordinates,
            network_type=args.network_type,
            buffer_degrees=args.buffer_degrees,
            overpass_url=args.overpass_url,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_path = args.output_dir / f"{matrix_id}.stations.csv"
    write_manifest(manifest_path, depot, candidates)
    bundle = write_matrix_bundle(
        output_dir=args.output_dir,
        matrix_id=matrix_id,
        directed_time_s=directed,
        metadata={
            "station_manifest": str(manifest_path),
            "station_count_including_depot": len(coordinates),
            "candidate_count": len(candidates),
            "depot_station_id": depot["station_id"],
            "network_type": args.network_type,
            "buffer_degrees": args.buffer_degrees,
            "overpass_url": args.overpass_url,
            "snapped_osm_node_ids": snapped_nodes,
            "source_station_csv": str(args.stations_csv),
        },
    )
    print(json.dumps(
        {
            "matrix_id": matrix_id,
            "candidate_count": len(candidates),
            "depot_station_id": depot["station_id"],
            "station_manifest": str(manifest_path),
            "directed_matrix": str(bundle.directed_matrix_path),
            "symmetric_matrix": str(bundle.symmetric_matrix_path),
            "metadata": str(bundle.metadata_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
