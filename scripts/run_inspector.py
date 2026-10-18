"""Recommended inspector entry point; dependency failures do not need a traceback."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from robust_industrial_anomaly_detection.prototype.launch import main as launch
    except (ImportError, OSError) as exc:
        reason = f"missing {exc.name}" if isinstance(exc, ModuleNotFoundError) else str(exc)
        print(
            f"The project environment could not load ({reason}).\n"
            "From the repository root, run: uv sync --locked --extra cpu --group dev\n"
            "Then use .\\start-inspector.ps1. See docs/STARTUP.md.",
            file=sys.stderr,
        )
        return 2
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
