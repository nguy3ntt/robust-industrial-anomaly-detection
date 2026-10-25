"""Copied as demo.py in the portable release. No .env or experiment tree needed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check files without ML dependencies")
    parser.add_argument(
        "--check", action="store_true", help="load verified runtime without serving"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="explicitly fetch official ResNet weights"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="use an existing local checkpoint without copying it"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    try:
        from robust_industrial_anomaly_detection.bundle_integrity import (
            CHECKPOINT_NAME,
            prepare_checkpoint,
            verify_bundle,
            verify_checkpoint,
        )

        manifest = verify_bundle(root)
        if args.verify:
            print(
                json.dumps(
                    {
                        "status": "verified",
                        "files": len(manifest["files"]),
                        "weights_included": False,
                    }
                )
            )
            return 0
        checkpoint = args.checkpoint or root / ".demo-cache" / CHECKPOINT_NAME
        if args.prepare:
            print(
                "Explicit preparation: checking/fetching official ResNet weights (about 45 MiB).",
                flush=True,
            )
            prepare_checkpoint(checkpoint)
        verify_checkpoint(checkpoint)
        from robust_industrial_anomaly_detection.prototype.launch import check_port
        from robust_industrial_anomaly_detection.prototype.release import load_portable_engine
        from robust_industrial_anomaly_detection.prototype.server import serve

        check_port(args.port)
        print("Loading verified portable runtime; no experiment folders are needed...", flush=True)
        engine = load_portable_engine(root, checkpoint)
        print(json.dumps(engine.runtime_info(), indent=2), flush=True)
        if args.check:
            print("Ready. No server was started.")
            return 0
        serve(engine, host="127.0.0.1", port=args.port, open_browser=args.open_browser)
        return 0
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency {exc.name}. Use Python 3.12 and: "
            "uv run --locked --extra cpu --no-dev python demo.py --check",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Portable demo cannot start: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Portable demo cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
