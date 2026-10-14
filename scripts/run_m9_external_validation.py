from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_industrial_anomaly_detection.experiments.m9 import run_m9


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen recipe on KolektorSDD2")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m9-external-validation.yaml")
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/ksdd2.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    result = run_m9(
        project_root=project,
        config_path=(project / args.config).resolve(),
        data_config=(project / args.data_config).resolve(),
        device=args.device,
        quick=args.quick,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
