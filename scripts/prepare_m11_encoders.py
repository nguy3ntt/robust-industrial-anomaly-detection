"""Acquire only official numeric weights, checking full pinned hashes before use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from dotenv import load_dotenv

from robust_industrial_anomaly_detection.baselines.encoders import (
    ENCODERS,
    checkpoint_path,
    verify_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare verified local M11 encoder checkpoints")
    parser.add_argument("--encoder", choices=tuple(ENCODERS), action="append")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    for name in args.encoder or ENCODERS:
        spec = ENCODERS[name]
        path = checkpoint_path(name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(
                spec.url, str(path.with_suffix(".download")), hash_prefix=spec.sha256
            )
            # Never overwrite an existing checkpoint, including a failed hash check.
            if path.exists():
                raise FileExistsError(path)
            path.with_suffix(".download").rename(path)
        verify_checkpoint(name)
        print(
            json.dumps(
                {
                    "encoder": name,
                    "weights": spec.weights_id,
                    "sha256": spec.sha256,
                    "bytes": path.stat().st_size,
                    "path": str(path),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
