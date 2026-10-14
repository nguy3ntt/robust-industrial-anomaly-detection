from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from robust_industrial_anomaly_detection.data.ksdd2 import (
    KSDD2Error,
    assign_normal_splits,
    load_ksdd2_manifest,
    safe_extract_zip,
)
from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.experiments.m9 import full_frame_pair, guarded_metrics


def _records() -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rng = np.random.default_rng(12)
    rows = [
        {
            "image_path": f"train/{index}.png",
            "image_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "perceptual_hash": f"{index:016x}",
            "source_split": "train",
            "label": "normal",
        }
        for index in range(400)
    ]
    thumbnails = [rng.integers(0, 256, (32, 32), dtype=np.uint8) for _ in rows]
    return rows, thumbnails


def test_split_keeps_duplicate_components_together_and_is_order_independent() -> None:
    records, thumbnails = _records()
    records[1]["image_sha256"] = records[0]["image_sha256"]
    kwargs = {"seed": 5, "validation_fraction": 0.15, "dhash_distance": 2, "mae_limit": 0.1}

    assign_normal_splits(records, thumbnails, **kwargs)
    assignments = {row["image_path"]: row["split"] for row in records}
    assign_normal_splits(list(reversed(records)), list(reversed(thumbnails)), **kwargs)

    assert assignments == {row["image_path"]: row["split"] for row in records}
    assert records[0]["split"] == records[1]["split"]
    assert set(assignments.values()) == {"train", "validation"}


def test_train_test_duplicates_are_excluded_from_fit_without_removing_provider_test() -> None:
    records, thumbnails = _records()
    test = {**records[0], "image_path": "test/0.png", "source_split": "test"}
    records.append(test)
    thumbnails.append(thumbnails[0])

    audit = assign_normal_splits(
        records, thumbnails, seed=5, validation_fraction=0.15, dhash_distance=2, mae_limit=0.1
    )

    assert records[0]["split"] == "excluded"
    assert test["split"] == "test"
    assert audit["exclusions"] == [{"image_path": "train/0.png", "reason": "test_duplicate"}]


def test_provider_training_defects_never_enter_fit_or_calibration() -> None:
    records, thumbnails = _records()
    records[0]["label"] = "anomaly"

    audit = assign_normal_splits(
        records, thumbnails, seed=5, validation_fraction=0.15, dhash_distance=2, mae_limit=0.1
    )

    assert records[0]["split"] == "excluded"
    assert all(row["label"] == "normal" for row in records if row["split"] != "excluded")
    assert audit["exclusions"] == [
        {"image_path": "train/0.png", "reason": "provider_training_anomaly"}
    ]


@pytest.mark.parametrize(
    "member", ["../escape.png", "/absolute.png", "C:/drive.png", "train/../escape.png"]
)
def test_zip_extraction_rejects_unsafe_paths_before_writing(tmp_path: Path, member: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, b"invalid")

    with pytest.raises(KSDD2Error, match="unsafe ZIP"):
        safe_extract_zip(archive, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_full_frame_geometry_preserves_defects_at_strip_ends(tmp_path: Path) -> None:
    image = np.zeros((180, 60, 3), dtype=np.uint8)
    mask = np.zeros((180, 60), dtype=np.uint8)
    image[:12, 12:48] = 255
    mask[:12, 12:48] = 255
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")

    resized, binary = full_frame_pair(tmp_path / "image.png", tmp_path / "mask.png", 64)

    assert resized.shape == (64, 64, 3)
    assert binary[:5].any()
    assert (resized[binary].mean(axis=1) > 180).all()


def test_recapture_is_a_deferral_and_does_not_count_as_an_anomaly_detection() -> None:
    metrics = guarded_metrics(
        [0, 0, 1, 1], ["anomalous", "normal", "anomalous", "normal"], [True, False, True, True]
    )

    assert metrics["guarded_normal_fpr"] == 0
    assert metrics["guarded_auto_anomaly_recall"] == 0
    assert metrics["guarded_anomaly_actionable_rate"] == 1
    assert metrics["guarded_auto_coverage"] == 0.25
    assert metrics["false_alerts_intercepted"] == 1


def test_audited_loader_rejects_modified_source_images(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("dataset:\n  id: test-dataset\n", encoding="utf-8")
    source = tmp_path / "sources" / "test-dataset"
    manifest_dir = tmp_path / "manifests" / "test-dataset"
    source.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    image = source / "image.png"
    mask = source / "mask.png"
    image.write_bytes(b"source")
    mask.write_bytes(b"mask")
    manifest = manifest_dir / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "split": "train",
                "image_path": "image.png",
                "mask_path": "mask.png",
                "image_sha256": sha256_file(image),
                "mask_sha256": sha256_file(mask),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "audit.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "config_sha256": sha256_file(config),
                "manifest_sha256": sha256_file(manifest),
            }
        ),
        encoding="utf-8",
    )
    image.write_bytes(b"modified")

    with pytest.raises(KSDD2Error, match="modified dataset file"):
        load_ksdd2_manifest(config, tmp_path)
