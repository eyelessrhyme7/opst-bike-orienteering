from __future__ import annotations

import csv
import json
from pathlib import Path

from opst import data_oslo


def sample_autodiscovery() -> dict[str, object]:
    return {
        "data": {
            "en": {
                "feeds": [
                    {"name": "station_status", "url": "https://example.test/status.json"},
                    {
                        "name": "station_information",
                        "url": "https://example.test/station_information.json",
                    },
                ]
            }
        }
    }


def sample_station_information() -> dict[str, object]:
    return {
        "last_updated": 123,
        "ttl": 10,
        "version": "2.3",
        "data": {
            "stations": [
                {
                    "station_id": "b",
                    "name": "Beta",
                    "lat": 59.91,
                    "lon": 10.75,
                    "capacity": 12,
                },
                {
                    "station_id": "a",
                    "name": "Alpha",
                    "lat": 59.92,
                    "lon": 10.76,
                    "capacity": 9,
                },
            ]
        },
    }


def test_find_feed_url_handles_language_wrapped_gbfs() -> None:
    url = data_oslo.find_feed_url(sample_autodiscovery(), "station_information")

    assert url == "https://example.test/station_information.json"


def test_station_records_normalize_and_sort() -> None:
    records = data_oslo.station_records(
        sample_station_information(), source_snapshot="snapshot_1"
    )

    assert [record["station_id"] for record in records] == ["a", "b"]
    assert records[0] == {
        "station_id": "a",
        "name": "Alpha",
        "lat": 59.92,
        "lon": 10.76,
        "capacity": 9,
        "source_snapshot": "snapshot_1",
    }


def test_cache_station_snapshot_writes_raw_metadata_and_csv(
    tmp_path: Path, monkeypatch
) -> None:
    responses = {
        data_oslo.OSLO_GBFS_AUTO_DISCOVERY_URL: sample_autodiscovery(),
        "https://example.test/station_information.json": sample_station_information(),
    }

    def fake_fetch_json(url: str, *, client_identifier: str) -> dict[str, object]:
        assert client_identifier == "test-client"
        return responses[url]

    monkeypatch.setattr(data_oslo, "fetch_json", fake_fetch_json)

    snapshot = data_oslo.cache_station_snapshot(
        output_root=tmp_path,
        client_identifier="test-client",
        timestamp="20260102T030405Z",
    )

    assert snapshot.station_count == 2
    assert snapshot.raw_path.exists()
    assert snapshot.metadata_path.exists()
    assert snapshot.normalized_csv_path.exists()

    metadata = json.loads(snapshot.metadata_path.read_text(encoding="utf-8"))
    assert metadata["snapshot_id"] == "oslo_station_information_20260102T030405Z"
    assert metadata["station_count"] == 2
    assert metadata["sha256"] == snapshot.sha256

    with snapshot.normalized_csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["station_id"] for row in rows] == ["a", "b"]
