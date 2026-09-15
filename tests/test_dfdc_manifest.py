from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from robust_deepfake_detection.data.dfdc import (
    DfdcConfig,
    DfdcConfigurationError,
    build_manifest,
    load_config,
    load_verified_manifest,
    write_build_outputs,
)
from robust_deepfake_detection.data.faceforensics import ManifestVerificationError, ProbeMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(root: Path, entries: dict[str, dict[str, object]]) -> None:
    root.mkdir(parents=True)
    for relative_path in entries:
        path = root.joinpath(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"content:{relative_path}".encode())
    (root / "dataset.json").write_text(json.dumps(entries), encoding="utf-8")


def _valid_entries() -> dict[str, dict[str, object]]:
    return {
        "original_videos/a.mp4": {"label": "real", "split": "train", "target": "a"},
        "original_videos/b.mp4": {"label": "real", "split": "train", "target": "b"},
        "method_A/a_b_A_001.mp4": {
            "label": "fake",
            "split": "train",
            "target": "b",
            "swapped": "a",
        },
        "original_videos/c.mp4": {"label": "real", "split": "train", "target": "c"},
        "original_videos/d.mp4": {"label": "real", "split": "train", "target": "d"},
        "method_B/c_d_B_001.mp4": {
            "label": "fake",
            "split": "train",
            "target": "d",
            "swapped": "c",
            "augmentations": [],
        },
        "original_videos/e.mp4": {"label": "real", "split": "test", "target": "e"},
        "original_videos/f.mp4": {"label": "real", "split": "test", "target": "f"},
        "method_A/e_f_A_001.mp4": {
            "label": "fake",
            "split": "test",
            "target": "f",
            "swapped": "e",
            "augmentations": ["fps_15"],
        },
    }


def _config(tmp_path: Path, root: Path, *, probe_mode: ProbeMode = "none") -> DfdcConfig:
    return DfdcConfig(
        dataset_root=root,
        metadata_path=root / "dataset.json",
        dataset_release="synthetic-preview-v1",
        access_date=date(2026, 9, 15),
        output_dir=tmp_path / "output",
        validation_fraction=0.5,
        split_seed=7,
        minimum_records=9,
        expected_identity_count=6,
        checksum_mode="sha256",
        probe_mode=probe_mode,
        workers=2,
        manifest_filename="manifest.jsonl",
        audit_filename="audit.json",
    )


def test_builder_preserves_test_and_derives_identity_disjoint_validation(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root, _valid_entries())
    result = build_manifest(_config(tmp_path, root))

    assert result.audit.passed
    assert len(result.records) == 9
    assert {record.split for record in result.records} == {"train", "val", "test"}
    assert all(
        record.split == "test" for record in result.records if record.provider_split == "test"
    )
    identity_splits: dict[str, set[str]] = {}
    for record in result.records:
        for identity in record.lineage_identity_ids:
            identity_splits.setdefault(identity, set()).add(record.split)
    assert all(len(splits) == 1 for splits in identity_splits.values())


def test_manifest_fingerprint_and_split_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root, _valid_entries())
    config = _config(tmp_path, root)

    first = build_manifest(config)
    second = build_manifest(config)

    assert first.records == second.records
    assert first.audit.manifest_sha256 == second.audit.manifest_sha256
    assert first.audit.split_policy == second.audit.split_policy


def test_provider_identity_crossing_train_and_test_rejects_manifest(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    entries = _valid_entries()
    entries["original_videos/e.mp4"]["target"] = "a"
    _write_dataset(root, entries)

    result = build_manifest(_config(tmp_path, root))

    assert not result.audit.passed
    assert "provider_identity_leakage" in {issue.code for issue in result.audit.issues}


def test_metadata_and_media_must_match_exactly(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    entries = _valid_entries()
    _write_dataset(root, entries)
    (root / "method_B" / "extra.mp4").write_bytes(b"unlisted")
    (root / "method_A" / "e_f_A_001.mp4").unlink()

    result = build_manifest(_config(tmp_path, root))
    codes = {issue.code for issue in result.audit.issues}

    assert not result.audit.passed
    assert {"metadata_media_missing", "unlisted_media"}.issubset(codes)


def test_byte_identical_files_across_splits_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    entries = _valid_entries()
    _write_dataset(root, entries)
    shared = b"duplicate"
    (root / "original_videos" / "a.mp4").write_bytes(shared)
    (root / "original_videos" / "e.mp4").write_bytes(shared)

    result = build_manifest(_config(tmp_path, root))

    assert not result.audit.passed
    assert "cross_split_content_duplicate" in {issue.code for issue in result.audit.issues}


def test_passing_output_is_verified_and_tampering_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root, _valid_entries())
    config = _config(tmp_path, root)
    result = build_manifest(config)
    manifest_path, audit_path, quarantined = write_build_outputs(result, config)

    assert manifest_path is not None
    assert quarantined is None
    assert len(load_verified_manifest(manifest_path, audit_path)) == 9
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="manifest hash mismatch"):
        load_verified_manifest(manifest_path, audit_path)


def test_invalid_label_directory_pair_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    entries = _valid_entries()
    entries["method_A/a_b_A_001.mp4"]["label"] = "real"
    _write_dataset(root, entries)

    with pytest.raises(DfdcConfigurationError, match="label/directory disagreement"):
        build_manifest(_config(tmp_path, root))


def test_default_config_requires_real_access_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", "D:/deepfake-data")

    with pytest.raises(DfdcConfigurationError, match="access_date must use YYYY-MM-DD"):
        load_config(PROJECT_ROOT / "configs" / "data" / "dfdc-preview.yaml")


def test_required_ffprobe_absence_rejects_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root, _valid_entries())
    monkeypatch.setattr("robust_deepfake_detection.data.dfdc.shutil.which", lambda _: None)

    result = build_manifest(_config(tmp_path, root, probe_mode="required"))

    assert not result.audit.passed
    assert "ffprobe_missing" in {issue.code for issue in result.audit.issues}
