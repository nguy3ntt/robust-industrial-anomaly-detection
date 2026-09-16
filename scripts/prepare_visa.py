"""Download, safely extract, audit, and manifest the official VisA dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from robust_industrial_anomaly_detection.data.visa import (
    VisaDataError,
    build_manifest,
    download_archive,
    extract_archive,
    load_config,
    locate_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/visa.yaml"),
        help="dataset configuration (default: configs/data/visa.yaml)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="external data root; defaults to ANOMALY_DATA_ROOT",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="manifest directory; defaults to DATA_ROOT/manifests/DATASET_ID",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--source-root",
        type=Path,
        help="use an already-extracted source root and skip download/extraction",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    return parser.parse_args()


def _data_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    configured = os.environ.get("ANOMALY_DATA_ROOT")
    if not configured:
        raise VisaDataError("set ANOMALY_DATA_ROOT in .env or pass --data-root")
    return Path(configured).expanduser().resolve()


def main() -> int:
    load_dotenv()
    args = parse_args()
    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        data_root = _data_root(args.data_root)
        archive_path: Path | None = None
        if args.source_root is None:
            print(f"Downloading/verifying {config.archive_url}", file=sys.stderr)
            archive_path = download_archive(config, data_root)
            print(f"Safely extracting/verifying {archive_path}", file=sys.stderr)
            source_root = extract_archive(archive_path, data_root, config.dataset_id)
        else:
            source_root = args.source_root.expanduser().resolve()
        layout = locate_layout(source_root, config)
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else data_root / "manifests" / config.dataset_id
        )
        print(f"Auditing {layout.image_root}", file=sys.stderr)
        result = build_manifest(
            layout,
            config,
            config_path,
            output_dir,
            archive_path=archive_path,
            workers=args.workers,
        )
    except VisaDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = {
        "passed": True,
        "records": result.record_count,
        "manifest": str(result.manifest_path),
        "audit": str(result.audit_path),
        "manifest_sha256": result.manifest_sha256,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"PASS: {result.record_count} records")
        print(f"Manifest: {result.manifest_path}")
        print(f"Audit: {result.audit_path}")
        print(f"SHA-256: {result.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
