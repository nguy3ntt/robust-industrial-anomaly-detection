"""Run or safely resume the controlled small-defect localization study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_industrial_anomaly_detection.experiments.m13 import run_m13


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m13-small-defects.yaml")
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dataset", action="append", choices=("visa", "ksdd2"))
    parser.add_argument("--category", action="append")
    parser.add_argument("--max-new-conditions", type=int)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            run_m13(
                project=project,
                config_path=project / args.config,
                quick=args.quick,
                resume=args.resume,
                datasets=args.dataset,
                categories=args.category,
                max_new_conditions=args.max_new_conditions,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
