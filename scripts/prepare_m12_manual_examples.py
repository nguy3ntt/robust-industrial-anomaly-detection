"""Create ignored local inspection examples from fixed official-test IDs."""

from __future__ import annotations

import argparse
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image

from robust_industrial_anomaly_detection.baselines.multiscale import load_fitted_rgb
from robust_industrial_anomaly_detection.data.visa import load_trusted_manifest, sha256_file
from robust_industrial_anomaly_detection.experiments.m2 import _atomic_json
from robust_industrial_anomaly_detection.experiments.m8 import load_m8_settings
from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionEngine,
    load_prototype_settings,
)
from robust_industrial_anomaly_detection.prototype.quality import (
    evaluate_capture_quality,
    load_quality_model,
)
from robust_industrial_anomaly_detection.robustness.transforms import apply_shift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="candle")
    parser.add_argument("--quality-run", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    load_dotenv(project / ".env")
    engine = InspectionEngine(
        project_root=project,
        settings=load_prototype_settings(project / "configs/prototype/m12.yaml"),
        device="cpu",
        quality_run=args.quality_run,
    )
    if args.category not in engine.categories:
        raise ValueError("choose one of the configured VisA categories")
    config = yaml.safe_load(
        (project / "configs/experiments/m12-quality-generalization.yaml").read_text("utf-8")
    )
    original_run = (
        Path(os.environ["ANOMALY_ARTIFACT_ROOT"]) / "runs" / config["sources"]["m8"]["run"]
    )
    if sha256_file(original_run / "metadata.json") != config["sources"]["m8"]["metadata_sha256"]:
        raise ValueError("historical gate metadata does not match its pin")
    original_metadata = json.loads((original_run / "metadata.json").read_text("utf-8"))
    if (
        sha256_file(original_run / "quality_model.json")
        != original_metadata["quality_model_sha256"]
    ):
        raise ValueError("historical gate artifact failed integrity verification")
    original = load_quality_model(original_run / "quality_model.json")
    data = Path(os.environ["ANOMALY_DATA_ROOT"])
    root = data / "sources/visa-20220922"
    manifest = data / "manifests/visa-20220922"
    records = load_trusted_manifest(
        manifest / "manifest.jsonl",
        manifest / "audit.json",
        project / "configs/data/visa.yaml",
        root,
    )
    selected = [
        next(
            row
            for row in records
            if row["category"] == args.category and row["split"] == "test" and row["label"] == label
        )
        for label in ("normal", "anomaly")
    ]
    images = []
    for row in selected:
        path = root / row["image_path"]
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("manual example source image changed")
        images.append(load_fitted_rgb(path, 224))
    protocol = load_m8_settings(project / config["source_protocol"])
    cases = [("clean-normal", images[0]), ("clean-anomaly", images[1])]
    for condition in protocol.conditions:
        if condition.identifier in {"blur_severe", "noise_severe"}:
            cases.append(
                (
                    condition.identifier + "-normal",
                    apply_shift(
                        images[0],
                        np.zeros((224, 224), bool),
                        condition,
                        seed=protocol.seed,
                        sample_key=selected[0]["sample_id"],
                    ).image,
                )
            )
    cases.append(("blank", np.zeros((224, 224, 3), np.uint8)))
    output = (
        project / "artifacts/manual" / f"m12-{args.category}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    output.mkdir(parents=True, exist_ok=False)
    expected: list[dict[str, Any]] = []
    for name, image in cases:
        filename = name + ".png"
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format="PNG")
        payload = stream.getvalue()
        (output / filename).write_bytes(payload)
        result = engine.inspect_bytes(payload, filename=filename, category=args.category)
        old = evaluate_capture_quality(image, category=args.category, model=original)
        expected.append(
            {
                "file": filename,
                "v1_quality_status": old.status,
                **result.public_dict(include_images=False),
            }
        )
    _atomic_json(
        output / "expected.json",
        {
            "category": args.category,
            "selection": "first manifest test normal/anomaly; no gate-outcome search",
            "runtime": engine.runtime_info(),
            "source_test_ids": [row["sample_id"] for row in selected],
            "results": expected,
            "publish_images": False,
        },
    )
    print(
        json.dumps(
            {
                "folder": str(output),
                "expected": str(output / "expected.json"),
                "results": [
                    {
                        key: row[key]
                        for key in (
                            "file",
                            "v1_quality_status",
                            "quality_status",
                            "decision",
                            "model_evaluated",
                        )
                    }
                    for row in expected
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
