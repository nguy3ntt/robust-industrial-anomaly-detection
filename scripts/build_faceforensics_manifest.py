"""Build and audit a leakage-safe FaceForensics++ video manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from robust_deepfake_detection.data import (
    FaceForensicsConfig,
    build_manifest,
    load_config,
    write_build_outputs,
)
from robust_deepfake_detection.data.faceforensics import ManifestConfigurationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/faceforensicspp.yaml"),
        help="YAML build configuration",
    )
    parser.add_argument("--dataset-root", type=Path, help="override the configured dataset root")
    parser.add_argument("--output-dir", type=Path, help="override the configured output directory")
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="development only: skip SHA-256 hashing and duplicate-content checks",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="development only: skip ffprobe media-integrity checks",
    )
    return parser.parse_args()


def apply_overrides(config: FaceForensicsConfig, args: argparse.Namespace) -> FaceForensicsConfig:
    updates: dict[str, object] = {}
    if args.dataset_root:
        updates["dataset_root"] = args.dataset_root.resolve()
    if args.output_dir:
        updates["output_dir"] = args.output_dir.resolve()
    if args.skip_checksums:
        updates["checksum_mode"] = "none"
    if args.skip_probes:
        updates["probe_mode"] = "none"
    return replace(config, **updates)


def main() -> int:
    args = parse_args()
    load_dotenv()
    try:
        config = apply_overrides(load_config(args.config), args)
        result = build_manifest(config)
        manifest_path, audit_path, quarantined_path = write_build_outputs(result, config)
    except (ManifestConfigurationError, OSError, ValueError) as exc:
        print(f"CONFIGURATION ERROR: {exc}")
        return 2

    print(json.dumps(result.audit.summary, indent=2, sort_keys=True))
    print(f"Audit: {audit_path}")
    if manifest_path is None:
        if quarantined_path is not None:
            print(f"Previous manifest quarantined: {quarantined_path}")
        print("RESULT: REJECTED — manifest was not published; inspect audit errors")
        return 1
    print(f"Manifest: {manifest_path}")
    print(f"Manifest SHA-256: {result.audit.manifest_sha256}")
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
