"""Run the frozen M3 multi-scale and coreset ablation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_industrial_anomaly_detection.experiments.m3 import run_m3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", help="Repeat to select categories.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--quick", action="store_true", help="Run the bounded smoke protocol.")
    parser.add_argument("--resume", type=Path, help="Resume a hash-validated incomplete run.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m3-multiscale.yaml")
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/visa.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_m3(
        project_root=Path.cwd(),
        experiment_config=args.config,
        data_config=args.data_config,
        categories=args.category,
        device=args.device,
        quick=args.quick,
        resume_run=args.resume,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"Artifacts: {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
