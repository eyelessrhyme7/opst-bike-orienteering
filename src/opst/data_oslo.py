from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

OSLO_GBFS_AUTO_DISCOVERY_URL = "https://gbfs.urbansharing.com/oslobysykkel.no/gbfs.json"
DEFAULT_CLIENT_IDENTIFIER = "nus-cs4234-opst"
NLOD_2_URL = "https://data.norge.no/nlod/en/2.0"


@dataclass(frozen=True)
class StationSnapshot:
    raw_path: Path
    metadata_path: Path
    normalized_csv_path: Path
    station_count: int
    sha256: str
    source_url: str
    retrieved_at_utc: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_json(url: str, *, client_identifier: str = DEFAULT_CLIENT_IDENTIFIER) -> dict[str, Any]:
    """Fetch a public GBFS JSON resource with the documented client header."""

    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Client-Identifier": client_identifier,
            "User-Agent": client_identifier,
        },
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - public HTTPS data source.
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _feed_lists(auto_discovery: dict[str, Any]) -> list[list[dict[str, Any]]]:
    data = auto_discovery.get("data", {})
    if isinstance(data, dict) and "feeds" in data and isinstance(data["feeds"], list):
        return [data["feeds"]]
    feeds: list[list[dict[str, Any]]] = []
    if isinstance(data, dict):
        for language_payload in data.values():
            if isinstance(language_payload, dict) and isinstance(language_payload.get("feeds"), list):
                feeds.append(language_payload["feeds"])
    return feeds


def find_feed_url(auto_discovery: dict[str, Any], feed_name: str) -> str:
    """Find a named feed URL in GBFS autodiscovery JSON."""

    for feeds in _feed_lists(auto_discovery):
        for feed in feeds:
            if feed.get("name") == feed_name and isinstance(feed.get("url"), str):
                return str(feed["url"])
    raise KeyError(f"GBFS feed not found: {feed_name}")


def station_records(station_information: dict[str, Any], *, source_snapshot: str) -> list[dict[str, Any]]:
    """Normalize GBFS station_information to the project's station table columns."""

    stations = station_information.get("data", {}).get("stations", [])
    records: list[dict[str, Any]] = []
    for station in stations:
        station_id = station.get("station_id")
        lat = station.get("lat")
        lon = station.get("lon")
        if station_id is None or lat is None or lon is None:
            continue
        records.append(
            {
                "station_id": str(station_id),
                "name": str(station.get("name", "")),
                "lat": float(lat),
                "lon": float(lon),
                "capacity": int(station.get("capacity", 0) or 0),
                "source_snapshot": source_snapshot,
            }
        )
    records.sort(key=lambda row: row["station_id"])
    return records


def write_station_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["station_id", "name", "lat", "lon", "capacity", "source_snapshot"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def cache_station_snapshot(
    *,
    output_root: Path,
    client_identifier: str = DEFAULT_CLIENT_IDENTIFIER,
    timestamp: str | None = None,
) -> StationSnapshot:
    """Fetch, cache, checksum, and normalize Oslo station_information."""

    retrieved_at = timestamp or utc_timestamp()
    auto_discovery = fetch_json(
        OSLO_GBFS_AUTO_DISCOVERY_URL, client_identifier=client_identifier
    )
    station_information_url = find_feed_url(auto_discovery, "station_information")
    station_information = fetch_json(
        station_information_url, client_identifier=client_identifier
    )

    raw_text = json.dumps(station_information, indent=2, sort_keys=True)
    checksum = sha256_text(raw_text)
    snapshot_id = f"oslo_station_information_{retrieved_at}"

    raw_path = output_root / "raw" / "oslo_gbfs" / f"{snapshot_id}.json"
    metadata_path = output_root / "raw" / "oslo_gbfs" / f"{snapshot_id}.metadata.json"
    normalized_path = output_root / "processed" / "oslo_stations.csv"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw_text, encoding="utf-8")

    records = station_records(station_information, source_snapshot=snapshot_id)
    write_station_csv(records, normalized_path)

    metadata = {
        "snapshot_id": snapshot_id,
        "retrieved_at_utc": retrieved_at,
        "source_url": station_information_url,
        "autodiscovery_url": OSLO_GBFS_AUTO_DISCOVERY_URL,
        "gbfs_last_updated": station_information.get("last_updated"),
        "gbfs_ttl": station_information.get("ttl"),
        "gbfs_version": station_information.get("version"),
        "license": "Norwegian Licence for Open Government Data (NLOD) 2.0",
        "license_url": NLOD_2_URL,
        "attribution": "Oslo City Bike / Oslo Bysykkel official open data",
        "sha256": checksum,
        "station_count": len(records),
        "client_identifier": client_identifier,
    }
    write_json(metadata_path, metadata)

    return StationSnapshot(
        raw_path=raw_path,
        metadata_path=metadata_path,
        normalized_csv_path=normalized_path,
        station_count=len(records),
        sha256=checksum,
        source_url=station_information_url,
        retrieved_at_utc=retrieved_at,
    )
