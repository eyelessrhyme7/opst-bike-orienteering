from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from opst.transform import assert_symmetric_metric


@dataclass(frozen=True)
class RoadMatrixBundle:
    directed_matrix_path: Path
    symmetric_matrix_path: Path
    metadata_path: Path
    symmetric_travel_time_s: NDArray[np.int64]


def assert_finite_directed_matrix(matrix_s: NDArray[np.floating[Any]]) -> None:
    matrix = np.asarray(matrix_s, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("directed matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("directed matrix contains non-finite values")
    if np.any(matrix < 0):
        raise ValueError("directed matrix contains negative values")
    if not np.allclose(np.diag(matrix), 0):
        raise ValueError("directed matrix diagonal must be zero")


def symmetrize_directed_times_s(
    directed_time_s: NDArray[np.floating[Any]] | NDArray[np.integer[Any]],
) -> NDArray[np.int64]:
    """Average directed travel times with their reverse and conservatively ceil."""

    directed = np.asarray(directed_time_s, dtype=float)
    assert_finite_directed_matrix(directed)
    symmetric = np.ceil((directed + directed.T) / 2.0).astype(np.int64)
    np.fill_diagonal(symmetric, 0)
    return symmetric


def validate_symmetric_travel_matrix(matrix_s: NDArray[np.int64], *, tolerance_s: int = 1) -> None:
    """Validate the integer symmetric travel-time matrix used by the OP-ST model."""

    assert_symmetric_metric(np.asarray(matrix_s, dtype=np.int64), tolerance_s=tolerance_s)


def write_matrix_bundle(
    *,
    output_dir: Path,
    matrix_id: str,
    directed_time_s: NDArray[np.floating[Any]] | NDArray[np.integer[Any]],
    metadata: dict[str, Any],
) -> RoadMatrixBundle:
    """Write directed/symmetric matrix files and metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    symmetric = symmetrize_directed_times_s(directed_time_s)
    validate_symmetric_travel_matrix(symmetric)

    directed_path = output_dir / f"{matrix_id}.directed_time_s.npy"
    symmetric_path = output_dir / f"{matrix_id}.symmetric_time_s.npy"
    metadata_path = output_dir / f"{matrix_id}.metadata.json"

    np.save(directed_path, np.asarray(directed_time_s, dtype=float))
    np.save(symmetric_path, symmetric)
    metadata = {
        **metadata,
        "matrix_id": matrix_id,
        "symmetrization": "ceil((directed_ij + directed_ji) / 2)",
        "time_unit": "seconds",
        "shape": list(symmetric.shape),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return RoadMatrixBundle(
        directed_matrix_path=directed_path,
        symmetric_matrix_path=symmetric_path,
        metadata_path=metadata_path,
        symmetric_travel_time_s=symmetric,
    )


def build_osmnx_directed_matrix_s(
    coordinates_lat_lon: Sequence[tuple[float, float]],
    *,
    network_type: str = "drive",
    buffer_degrees: float = 0.02,
    overpass_url: str | None = None,
) -> tuple[NDArray[np.float64], list[int]]:
    """Build directed shortest-path travel times with OSMnx.

    Coordinates are ordered exactly as the returned matrix. Index 0 should be
    the depot when used for OP-ST instances.
    """

    try:
        import networkx as nx
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError(
            "OSMnx/networkx are required to build road matrices. "
            "Install project dependencies in a Python 3.11+ environment, e.g. `uv sync`."
        ) from exc
    if overpass_url:
        ox.settings.overpass_url = overpass_url

    if not coordinates_lat_lon:
        raise ValueError("at least one coordinate is required")

    lats = [lat for lat, _ in coordinates_lat_lon]
    lons = [lon for _, lon in coordinates_lat_lon]
    north = max(lats) + buffer_degrees
    south = min(lats) - buffer_degrees
    east = max(lons) + buffer_degrees
    west = min(lons) - buffer_degrees

    try:
        try:
            graph = ox.graph_from_bbox(north, south, east, west, network_type=network_type)
        except TypeError:
            graph = ox.graph_from_bbox((west, south, east, north), network_type=network_type)
    except Exception as exc:
        raise RuntimeError(
            "OSMnx graph download failed. The public Overpass endpoint may be busy; "
            "retry later or pass an alternate --overpass-url."
        ) from exc

    if not nx.is_strongly_connected(graph):
        largest = max(nx.strongly_connected_components(graph), key=len)
        graph = graph.subgraph(largest).copy()

    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    snapped_nodes = list(ox.nearest_nodes(graph, X=lons, Y=lats))

    n = len(snapped_nodes)
    matrix = np.full((n, n), math.inf, dtype=float)
    for source_index, source_node in enumerate(snapped_nodes):
        lengths = nx.single_source_dijkstra_path_length(
            graph, source_node, weight="travel_time"
        )
        for target_index, target_node in enumerate(snapped_nodes):
            matrix[source_index, target_index] = float(lengths.get(target_node, math.inf))
    np.fill_diagonal(matrix, 0.0)
    assert_finite_directed_matrix(matrix)
    return matrix, [int(node) for node in snapped_nodes]
