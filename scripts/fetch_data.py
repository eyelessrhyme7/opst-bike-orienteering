from __future__ import annotations

import argparse
import os
from pathlib import Path

from opst.data_oslo import DEFAULT_CLIENT_IDENTIFIER, cache_station_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public Oslo City Bike station data.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root directory for raw and processed data outputs.",
    )
    parser.add_argument(
        "--client-identifier",
        default=os.environ.get("OPST_CLIENT_IDENTIFIER", DEFAULT_CLIENT_IDENTIFIER),
        help="Value for the GBFS Client-Identifier header.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = cache_station_snapshot(
        output_root=args.output_root,
        client_identifier=args.client_identifier,
    )
    print(f"Fetched {snapshot.station_count} stations")
    print(f"Raw snapshot: {snapshot.raw_path}")
    print(f"Metadata: {snapshot.metadata_path}")
    print(f"Normalized CSV: {snapshot.normalized_csv_path}")
    print(f"SHA-256: {snapshot.sha256}")


if __name__ == "__main__":
    main()
