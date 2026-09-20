"""Run the reproducible M2 baseline suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from robust_industrial_anomaly_detection.experiments.m2 import (
    BASELINES,
    BaselineName,
    run_m2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=(*BASELINES, "all"),
        action="append",
        help="Baseline to run; repeat to select several (default: all).",
    )
    parser.add_argument(
        "--category",
        action="append",
        help="VisA category to run; repeat to select several (default: all).",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a clearly labelled, bounded smoke protocol instead of reportable full data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/m2-baselines.yaml"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted run after validating its config and input hashes.",
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/visa.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = args.baseline or ["all"]
    baselines = list(BASELINES) if "all" in requested else requested
    result = run_m2(
        project_root=Path.cwd(),
        experiment_config=args.config,
        data_config=args.data_config,
        baselines=cast(list[BaselineName], baselines),
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
