"""Prepare, freeze, audit and eventually evaluate genuine local capture sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_industrial_anomaly_detection.data.capture_study import (
    CaptureStudyError,
    audit,
    evaluate,
    freeze,
    initialize,
    verify_freeze,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "audit", "freeze", "evaluate"))
    parser.add_argument(
        "--study", type=Path, default=Path("artifacts/capture-studies/fresh-capture-pilot")
    )
    parser.add_argument(
        "--category", default="candle", help="init only: actual matching supported product"
    )
    parser.add_argument(
        "--evaluation", action="store_true", help="audit the frozen final inventory"
    )
    parser.add_argument(
        "--unlock-holdout", action="store_true", help="confirm first final inference / exact resume"
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.study.resolve()
    try:
        if args.command == "init":
            result = initialize(root, category=args.category)
        elif args.command == "audit":
            audited = audit(root, evaluation=args.evaluation)
            result = {key: value for key, value in audited.items() if key != "captures"}
        elif args.command == "freeze":
            # Validate collection first, before any expensive runtime loading.
            audit(root)
            from robust_industrial_anomaly_detection.prototype.launch import load_inspector

            engine = load_inspector(project)
            result = freeze(root, project, engine)
        else:
            if not args.unlock_holdout:
                raise CaptureStudyError(
                    "Final inference requires --unlock-holdout. This opens the held-out study."
                )
            lock = verify_freeze(root)
            audit(root, evaluation=True)
            from robust_industrial_anomaly_detection.prototype.launch import load_inspector

            engine = load_inspector(
                project,
                device=lock["runtime"]["device"],
                m5_run=Path(lock["m5_run"]),
                quality_run=Path(lock["quality_run"]),
                localization_run=Path(lock["localization_run"]),
            )
            result = evaluate(root, project, engine)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Capture study cannot proceed: {exc}", file=sys.stderr)
        print(
            "See docs/FRESH_CAPTURE_STUDY.md. No substitute photographs or results were created.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
