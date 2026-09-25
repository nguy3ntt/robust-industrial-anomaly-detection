from __future__ import annotations

import argparse
from pathlib import Path

from robust_industrial_anomaly_detection.reporting.m7 import build_m7_package


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen M7 publication package")
    parser.add_argument("--config", type=Path, default=Path("configs/reporting/m7.yaml"))
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest = build_m7_package(project_root, project_root / arguments.config)
    print(f"M7 package: {manifest['output_directory']}")
    print(f"Outputs: {len(manifest['outputs'])}")
    print(f"Package SHA-256: {manifest['package_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
