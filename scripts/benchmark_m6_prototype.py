from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_industrial_anomaly_detection.prototype.benchmark import run_benchmark
from robust_industrial_anomaly_detection.prototype.engine import load_prototype_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the frozen M6 local prototype")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--m5-run", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/prototype/m6.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/visa.yaml"))
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    settings = load_prototype_settings((project_root / arguments.config).resolve())
    run_dir, summary = run_benchmark(
        project_root=project_root,
        settings=settings,
        data_config=(project_root / arguments.data_config).resolve(),
        device=arguments.device,
        m5_run=arguments.m5_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
