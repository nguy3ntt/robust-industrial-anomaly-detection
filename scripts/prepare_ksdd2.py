from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from robust_industrial_anomaly_detection.data.ksdd2 import download_ksdd2, prepare_ksdd2


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire and audit official KolektorSDD2")
    parser.add_argument("--config", type=Path, default=Path("configs/data/ksdd2.yaml"))
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    load_dotenv(project / ".env")
    raw_root = os.environ.get("ANOMALY_DATA_ROOT")
    if not raw_root:
        raise RuntimeError("ANOMALY_DATA_ROOT is required")
    config = (project / args.config).resolve()
    if args.download_only:
        print(download_ksdd2(config, Path(raw_root)))
    else:
        print(prepare_ksdd2(config, Path(raw_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
