from __future__ import annotations

import argparse
from pathlib import Path

from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionEngine,
    load_prototype_settings,
)
from robust_industrial_anomaly_detection.prototype.server import serve


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the M6 local inspection interface")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--m5-run", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/prototype/m6.yaml"))
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    settings = load_prototype_settings((project_root / arguments.config).resolve())
    host = arguments.host or settings.host
    port = arguments.port or settings.port
    engine = InspectionEngine(
        project_root=project_root,
        settings=settings,
        device=arguments.device,
        m5_run=arguments.m5_run,
    )
    serve(engine, host=host, port=port, open_browser=arguments.open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
