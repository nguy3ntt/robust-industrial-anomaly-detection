from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_industrial_anomaly_detection.experiments.m8 import run_m8


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the M8 pre-inference quality-gate study")
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--source-m5-run", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m8-quality-gate.yaml")
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/visa.yaml"))
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    result = run_m8(
        project_root=project_root,
        experiment_config=(project_root / arguments.config).resolve(),
        data_config=(project_root / arguments.data_config).resolve(),
        categories=arguments.categories,
        quick=arguments.quick,
        source_m5_run=arguments.source_m5_run,
        resume_run=arguments.resume,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"Artifacts: {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
