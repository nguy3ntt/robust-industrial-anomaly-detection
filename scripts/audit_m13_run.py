"""Independently audit exact normal/test membership in the pinned localization study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml
from build_m13_report import verified_source
from dotenv import load_dotenv

from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.experiments.m2 import _read_csv
from robust_industrial_anomaly_detection.experiments.m12 import verify_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reporting/m13.yaml"))
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Also rehash every consumed source image/mask; no inference",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    load_dotenv(project / ".env")
    config = yaml.safe_load((project / args.config).read_text("utf-8"))
    run = verified_source(config, project)
    metadata = json.loads((run / "metadata.json").read_text("utf-8"))
    settings = json.loads((run / "resolved_config.json").read_text("utf-8"))
    data = Path(os.environ["ANOMALY_DATA_ROOT"])
    seed = settings["experiment"]["seed"]
    total = units = fit_count = calibration_count = 0
    for dataset, dataset_id in (("visa", "visa-20220922"), ("ksdd2", "ksdd2-20230928")):
        manifest = data / "manifests" / dataset_id / "manifest.jsonl"
        audit = manifest.parent / "audit.json"
        if (
            sha256_file(manifest) != metadata[f"{dataset}_manifest_sha256"]
            or sha256_file(audit) != metadata[f"{dataset}_audit_sha256"]
        ):
            raise ValueError("local source manifest/audit no longer matches the evaluated inputs")
        records = [json.loads(line) for line in manifest.read_text("utf-8").splitlines()]
        for category in sorted({row["category"] for row in records}):
            selected = [row for row in records if row["category"] == category]
            partitions = {
                split: sorted(
                    [row for row in selected if row["split"] == split],
                    key=lambda row: hashlib.sha256(
                        f"{seed}:{row['sample_id']}".encode()
                    ).hexdigest(),
                )
                for split in ("train", "validation", "test")
            }
            train = partitions["train"][:128]
            validation, test = partitions["validation"], partitions["test"]
            if any(row["label"] != "normal" for row in [*train, *validation]):
                raise ValueError("non-normal source row entered fitting or calibration")
            directory = run / "units" / dataset / category
            for mode in ("global", "tiled"):
                fit = json.loads((directory / mode / "fit.json").read_text("utf-8"))
                if fit["fit_ids"] != [row["sample_id"] for row in train] or fit[
                    "calibration_ids"
                ] != [row["sample_id"] for row in validation]:
                    raise ValueError(
                        "evaluated normal-reference IDs differ from original source partitions"
                    )
            expected = {(row["sample_id"], row["label"]) for row in test}
            for condition in settings["conditions"]:
                rows = _read_csv(directory / "conditions" / condition / "predictions.csv")
                for readout in ("global56", "global336", "tiled336"):
                    predictions = [
                        (row["sample_id"], row["label"])
                        for row in rows
                        if row["readout"] == readout
                    ]
                    if len(predictions) != len(expected) or set(predictions) != expected:
                        raise ValueError("missing/duplicate/relabelled official test images")
                total += len(rows)
            if args.verify_images:
                hashes = {}
                for row in [*train, *validation, *test]:
                    hashes[row["image_path"]] = row["image_sha256"]
                    if row["mask_path"]:
                        hashes[row["mask_path"]] = row["mask_sha256"]
                verify_files(data / "sources" / dataset_id, hashes)
            fit_count += len(train)
            calibration_count += len(validation)
            units += 1
    if units != 13 or total != 28494:
        raise ValueError("unexpected full localization audit coverage")
    files = [*sorted((project / "src").rglob("*.py")), project / "scripts/run_m13_small_defects.py"]
    code = hashlib.sha256(
        "".join(
            f"{path.relative_to(project).as_posix()}:{sha256_file(path)}\n" for path in files
        ).encode()
    ).hexdigest()
    print(
        json.dumps(
            {
                "source_run": run.name,
                "units": units,
                "fit_normals_per_model": fit_count,
                "calibration_normals_per_model": calibration_count,
                "official_test_images": 3166,
                "readout_condition_predictions": total,
                "exact_normal_and_test_membership_verified": True,
                "all_committed_output_hashes_verified": True,
                "consumed_image_mask_hashes_reverified": args.verify_images,
                "current_source_code_matches_recorded_fingerprint": code == metadata["code_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
