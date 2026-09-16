from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robust_industrial_anomaly_detection.data.visa import (
    DatasetLayout,
    VisaConfig,
    VisaDataError,
    build_manifest,
    derive_splits,
    download_archive,
    extract_archive,
    load_trusted_manifest,
    parse_split_table,
    sha256_file,
    verify_archive,
)


def _config(
    *, archive_url: str = "https://example.invalid/visa.tar", digest: str = "0" * 64
) -> VisaConfig:
    return VisaConfig(
        dataset_id="visa-test",
        version="test",
        archive_url=archive_url,
        archive_sha256=digest,
        accessed_on="2026-09-16",
        expected_images=4,
        expected_normal=3,
        expected_anomalous=1,
        expected_categories=1,
        perceptual_max_dhash_distance=2,
        perceptual_max_normalized_mae=0.1,
        official_split_csv="split_csv/1cls.csv",
        validation_fraction=0.25,
        seed=42,
        categories=("widget",),
    )


def _pattern(kind: int) -> np.ndarray:
    y, x = np.mgrid[0:32, 0:32]
    if kind == 0:
        values = (x * 7 + y * 2) % 256
    elif kind == 1:
        values = (y * 9 + (x // 4) * 31) % 256
    elif kind == 2:
        values = ((x // 3 + y // 5) % 2) * 220 + 20
    else:
        values = ((x - 16) ** 2 + (y - 16) ** 2) * 2 % 256
    return np.stack((values, np.roll(values, kind + 1, axis=0), 255 - values), axis=2).astype(
        np.uint8
    )


def _write_image(path: Path, kind: int, *, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_pattern(kind), mode="RGB").save(path, quality=quality)


def _dataset(
    tmp_path: Path, duplicate: str | None = None
) -> tuple[DatasetLayout, VisaConfig, Path]:
    source_root = tmp_path / "source"
    image_root = source_root / "VisA"
    category_root = image_root / "widget"
    normal_root = category_root / "Data" / "Images" / "Normal"
    anomaly_root = category_root / "Data" / "Images" / "Anomaly"
    mask_root = category_root / "Data" / "Masks" / "Anomaly"
    paths = {
        "train_1": normal_root / "001.JPG",
        "train_2": normal_root / "002.JPG",
        "test_normal": normal_root / "003.JPG",
        "test_anomaly": anomaly_root / "001.JPG",
    }
    _write_image(paths["train_1"], 0)
    _write_image(paths["train_2"], 1)
    _write_image(paths["test_normal"], 2)
    _write_image(paths["test_anomaly"], 3)
    if duplicate == "exact":
        shutil.copyfile(paths["train_1"], paths["test_normal"])
    elif duplicate == "perceptual":
        image = Image.fromarray(_pattern(0), mode="RGB")
        image.save(paths["train_1"], format="PNG", compress_level=0)
        image.save(paths["test_normal"], format="PNG", compress_level=9)

    mask_root.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:20, 10:22] = 255
    Image.fromarray(mask, mode="L").save(mask_root / "001.png")
    (category_root / "image_anno.csv").write_text(
        "image,label\n001.JPG,anomaly\n", encoding="utf-8"
    )

    split_root = source_root / "split_csv"
    split_root.mkdir(parents=True)
    split_path = split_root / "1cls.csv"
    split_path.write_text(
        "object,split,label,image,mask\n"
        "widget,train,normal,widget/Data/Images/Normal/001.JPG,\n"
        "widget,train,normal,widget/Data/Images/Normal/002.JPG,\n"
        "widget,test,normal,widget/Data/Images/Normal/003.JPG,\n"
        "widget,test,anomaly,widget/Data/Images/Anomaly/001.JPG,"
        "widget/Data/Masks/Anomaly/001.png\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("synthetic: true\n", encoding="utf-8")
    return DatasetLayout(source_root, image_root, split_path), _config(), config_path


def test_verify_archive_accepts_only_the_expected_digest(tmp_path: Path) -> None:
    archive = tmp_path / "archive.tar"
    archive.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()

    assert verify_archive(archive, digest) == digest
    with pytest.raises(VisaDataError, match="mismatch"):
        verify_archive(archive, "0" * 64)


def test_download_archive_supports_a_direct_file_source(tmp_path: Path) -> None:
    source = tmp_path / "source.tar"
    source.write_bytes(b"archive payload")
    config = _config(archive_url=source.as_uri(), digest=sha256_file(source))

    downloaded = download_archive(config, tmp_path / "data")

    assert downloaded.read_bytes() == b"archive payload"
    assert not downloaded.with_name(f"{downloaded.name}.part").exists()


def test_safe_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("../escape.txt")
        payload = b"escape"
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))

    with pytest.raises(VisaDataError, match="unsafe archive member"):
        extract_archive(archive, tmp_path / "data", "unsafe")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extraction_publishes_complete_tree_atomically(tmp_path: Path) -> None:
    archive = tmp_path / "safe.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("wrapper/file.txt")
        payload = b"safe"
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))

    extracted = extract_archive(archive, tmp_path / "data", "safe")

    assert (extracted / "wrapper" / "file.txt").read_bytes() == b"safe"


def test_calibration_split_is_deterministic_and_preserves_test(tmp_path: Path) -> None:
    layout, config, _ = _dataset(tmp_path)
    rows = parse_split_table(layout.split_csv, config)

    forward = derive_splits(rows, config)
    reversed_order = derive_splits(tuple(reversed(rows)), config)

    assert forward == reversed_order
    assert forward["widget/Data/Images/Normal/003.JPG"] == "test"
    assert forward["widget/Data/Images/Anomaly/001.JPG"] == "test"
    assert Counter(forward.values()) == Counter(train=1, validation=1, test=2)


def test_manifest_build_and_trusted_load(tmp_path: Path) -> None:
    layout, config, config_path = _dataset(tmp_path)
    output = tmp_path / "manifest"

    result = build_manifest(layout, config, config_path, output, workers=2)
    records = load_trusted_manifest(
        result.manifest_path, result.audit_path, config_path, layout.image_root
    )
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))

    assert result.record_count == len(records) == 4
    assert audit["passed"] is True
    assert audit["checks"]["official_test_preserved"] is True
    assert {record["split"] for record in records} == {"train", "validation", "test"}

    with result.manifest_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(VisaDataError, match="manifest hash"):
        load_trusted_manifest(result.manifest_path, result.audit_path, config_path)


@pytest.mark.parametrize("duplicate", ["exact", "perceptual"])
def test_manifest_rejects_content_leakage(tmp_path: Path, duplicate: str) -> None:
    layout, config, config_path = _dataset(tmp_path, duplicate)

    with pytest.raises(VisaDataError, match="duplicate"):
        build_manifest(layout, config, config_path, tmp_path / "manifest", workers=1)


def test_pending_access_date_blocks_publication(tmp_path: Path) -> None:
    layout, config, config_path = _dataset(tmp_path)

    with pytest.raises(VisaDataError, match="PENDING"):
        build_manifest(
            layout,
            replace(config, accessed_on="PENDING"),
            config_path,
            tmp_path / "manifest",
        )
