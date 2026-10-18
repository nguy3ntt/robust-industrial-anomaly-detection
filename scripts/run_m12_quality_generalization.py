"""Run or safely resume the fixed capture-gate comparison on both datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_industrial_anomaly_detection.experiments.m12 import run_m12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m12-quality-generalization.yaml")
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-conditions", type=int)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            run_m12(
                project=project,
                config_path=project / args.config,
                quick=args.quick,
                resume=args.resume,
                stop_after_conditions=args.stop_after_conditions,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
