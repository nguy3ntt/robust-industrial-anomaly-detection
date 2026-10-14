from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_industrial_anomaly_detection.experiments.m10 import run_m10


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate normal-only thresholded localization")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m10-localization.yaml")
    )
    parser.add_argument("--dataset", choices=("visa", "ksdd2"), action="append", dest="datasets")
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    result = run_m10(
        project_root=project,
        config_path=(project / args.config).resolve(),
        datasets=args.datasets,
        categories=args.categories,
        device=args.device,
        quick=args.quick,
        resume=args.resume,
        reference_run=args.reference_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
