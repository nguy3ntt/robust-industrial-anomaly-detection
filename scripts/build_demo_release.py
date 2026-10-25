"""Build a local-only portable release candidate; never publish it automatically."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_industrial_anomaly_detection.prototype.launch import load_inspector
from robust_industrial_anomaly_detection.prototype.release import export_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/releases/industrial-anomaly-demo-m16")
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    try:
        engine = load_inspector(project)
        result = export_release(project, args.output.resolve(), engine)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Release export refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
