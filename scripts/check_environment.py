"""Check that the active runtime matches the project's recorded environment."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from robust_deepfake_detection.environment import format_report, inspect_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accelerator",
        choices=("any", "cpu", "cuda"),
        default="any",
        help="accelerator expected from this environment (default: any)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    report = inspect_environment(args.accelerator)
    print(report.to_json() if args.json else format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
