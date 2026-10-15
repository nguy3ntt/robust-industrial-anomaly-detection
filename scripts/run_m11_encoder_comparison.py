from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_industrial_anomaly_detection.experiments.m11 import run_m11


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare matched normal-only frozen encoders")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m11-encoder-comparison.yaml")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dataset", choices=("visa", "ksdd2"), action="append", dest="datasets")
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-new-conditions", type=int, help="Stop safely after N new condition checkpoints"
    )
    args = parser.parse_args()
    if args.max_new_conditions is not None and args.max_new_conditions < 1:
        parser.error("--max-new-conditions must be positive")
    project = Path(__file__).resolve().parents[1]
    result = run_m11(
        project_root=project,
        config_path=(project / args.config).resolve(),
        device=args.device,
        quick=args.quick,
        resume=args.resume,
        datasets=args.datasets,
        categories=args.categories,
        max_new_conditions=args.max_new_conditions,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
