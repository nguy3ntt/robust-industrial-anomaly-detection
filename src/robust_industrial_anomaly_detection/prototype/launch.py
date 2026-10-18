"""Offline-only, actionable startup checks for the recommended local inspector."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

from dotenv import load_dotenv

from robust_industrial_anomaly_detection.baselines.encoders import ENCODERS
from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionEngine,
    PrototypeError,
    load_prototype_settings,
)
from robust_industrial_anomaly_detection.prototype.server import serve


class StartupError(RuntimeError):
    """A failed prerequisite with a concrete recovery action."""


def check_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise StartupError("Port must be between 1 and 65535; try --port 8770.")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # Windows otherwise allows an exclusive-looking bind over another listener.
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", port))
    except OSError as exc:
        suggestion = 8770 if port != 8770 else 8771
        raise StartupError(
            f"Port {port} is occupied or unavailable. Nothing was stopped. "
            f"Close your own earlier inspector or choose --port {suggestion} "
            f"(PowerShell launcher: -Port {suggestion})."
        ) from exc


def load_inspector(
    project: Path,
    *,
    device: str = "cpu",
    m5_run: Path | None = None,
    quality_run: Path | None = None,
    localization_run: Path | None = None,
) -> InspectionEngine:
    """Check local prerequisites before a constructor could fetch missing weights."""

    load_dotenv(project / ".env")
    raw = os.environ.get("ANOMALY_ARTIFACT_ROOT", "")
    if not raw:
        raise StartupError(
            "Experiment storage is not configured. Copy .env.example to .env, "
            "then set ANOMALY_ARTIFACT_ROOT to your existing artifact folder. "
            "See docs/STARTUP.md; a source checkout alone has no trained references."
        )
    root = Path(raw)
    if not root.is_absolute() or not (root / "runs").is_dir():
        raise StartupError(
            f"Experiment storage is missing or not absolute: {root}. "
            "Correct ANOMALY_ARTIFACT_ROOT in .env or restore your local runs. "
            "Do not create empty experiment folders to bypass this check."
        )
    checkpoint = (
        Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
        / "hub/checkpoints/resnet18-f37072fd.pth"
    )
    if not checkpoint.is_file():
        raise StartupError(
            f"ResNet weights are missing: {checkpoint}. Restore the verified local "
            "checkpoint or follow docs/STARTUP.md for explicit acquisition. "
            "Startup does not download weights."
        )
    if sha256_file(checkpoint) != ENCODERS["resnet18"].sha256:
        raise StartupError(
            f"ResNet weights failed their SHA-256 check: {checkpoint}. "
            "Restore the pinned original; do not disable verification."
        )
    settings = load_prototype_settings(project / "configs/prototype/m10.yaml")
    try:
        return InspectionEngine(
            project_root=project,
            settings=settings,
            device=device,
            m5_run=m5_run,
            quality_run=quality_run,
            localization_run=localization_run,
        )
    except (OSError, ValueError, KeyError, PrototypeError) as exc:
        raise StartupError(
            f"Model references are missing, incompatible, or failed verification: {exc}\n"
            "Check the M5 calibration, M4 indexes, M8 gate and M10 localization runs "
            "in your artifact storage. Restore the matching files, or follow the "
            "ordered recovery steps in docs/STARTUP.md. No fallback model was used."
        ) from exc
    except RuntimeError as exc:
        raise StartupError(
            f"The model could not initialize on {device}: {exc}. "
            "Use --device cpu unless your locked CUDA environment is configured."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the verified M10 local inspector, offline")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check", action="store_true", help="verify readiness without serving")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--debug", action="store_true", help="show a traceback on failure")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[3]
    try:
        check_port(args.port)
        print(
            "Checking local weights, calibration, gate and localization references...", flush=True
        )
        engine = load_inspector(project, device=args.device)
        print(json.dumps(engine.runtime_info(), indent=2), flush=True)
        if args.check:
            print("Ready. All model checks passed; no server was started.", flush=True)
            return 0
        print("Keep this terminal open. Stop your inspector with Ctrl+C.", flush=True)
        # Rebinding can still lose a race after preflight; the boundary below handles it.
        serve(engine, host="127.0.0.1", port=args.port, open_browser=args.open_browser)
        return 0
    except KeyboardInterrupt:
        print("Startup cancelled. No other service was stopped.", file=sys.stderr)
        return 130
    except (StartupError, PrototypeError, OSError) as exc:
        if args.debug:
            raise
        print(f"Inspector could not start: {exc}", file=sys.stderr)
        if isinstance(exc, OSError):
            print(
                "Check local file access or try --port 8770. See docs/STARTUP.md.", file=sys.stderr
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
