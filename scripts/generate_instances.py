from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from opst.instance_generation import (
    generate_instance_from_matrix,
    load_matrix_stations,
    write_instance_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate seeded OP-ST instances.")
    parser.add_argument(
        "--matrix-id",
        default="oslo_sandbox_n39_drive",
        help="Matrix id prefix under data/processed/oslo_matrix.",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=Path("data/processed/oslo_matrix"),
        help="Directory containing matrix and station manifest files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/instances/smoke"),
        help="Output directory for generated instance JSON files and manifest.",
    )
    parser.add_argument("--n-values", nargs="+", type=int, default=[8])
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["compact", "dispersed", "mixed", "outlier"],
    )
    parser.add_argument("--service-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--budget-tightness",
        nargs="+",
        default=["tight", "medium", "loose"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--reward-model",
        choices=["coarse", "fine"],
        default="coarse",
        help="Synthetic reward model. Use fine only for sensitivity analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    station_manifest = args.matrix_dir / f"{args.matrix_id}.stations.csv"
    matrix_path = args.matrix_dir / f"{args.matrix_id}.symmetric_time_s.npy"
    depot, candidates = load_matrix_stations(station_manifest)
    travel_time_s = np.load(matrix_path).astype(np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.n_values:
        for regime in args.regimes:
            for service_scale in args.service_scales:
                for tightness in args.budget_tightness:
                    for seed in args.seeds:
                        instance = generate_instance_from_matrix(
                            matrix_id=args.matrix_id,
                            depot=depot,
                            candidates=candidates,
                            travel_time_s=travel_time_s,
                            n=n,
                            regime=regime,
                            service_scale=service_scale,
                            budget_tightness=tightness,
                            seed=seed,
                            reward_model=args.reward_model,
                        )
                        instance_path = args.output_dir / f"{instance.instance_id}.json"
                        write_instance_json(instance, instance_path)
                        rows.append(
                            {
                                "instance_id": instance.instance_id,
                                "path": str(instance_path),
                                "n": n,
                                "regime": regime,
                                "service_scale": service_scale,
                                "budget_tightness": tightness,
                                "seed": seed,
                                "budget_s": instance.budget_s,
                                "matrix_id": args.matrix_id,
                                "reward_model": args.reward_model,
                            }
                        )

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} instances")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
