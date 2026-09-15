from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from robust_deepfake_detection.data.faceforensics import (
    FaceForensicsConfig,
    ManifestConfigurationError,
    ManifestVerificationError,
    ProbeMode,
    build_manifest,
    load_config,
    load_official_splits,
    load_verified_manifest,
    write_build_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SPLITS = PROJECT_ROOT / "configs" / "data" / "faceforensicspp" / "splits"
EXPECTED_SPLIT_HASHES = {
    "train": "e59386911255e6fb0a79a7808ec2210536b8c0474268342694c4bca766d1b347",
    "val": "b48cc511f66938e05356aaa9c67e150b1e3638c355db3d745225b0022884b36e",
    "test": "886f5a0da623c25820692e0d8dc33d197ddb1db527a7f1cfcb9bcbca60fe4f40",
}


def _write_splits(root: Path, split_pairs: dict[str, list[list[str]]]) -> Path:
    split_dir = root / "splits"
    split_dir.mkdir(parents=True)
    for split_name in ("train", "val", "test"):
        payload = split_pairs.get(split_name, [])
        (split_dir / f"{split_name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return split_dir


def _video_path(root: Path, relative_path: str, content: bytes) -> Path:
    path = root / Path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _config(
    tmp_path: Path,
    *,
    dataset_root: Path,
    split_dir: Path,
    expected_counts: dict[str, int],
    methods: tuple[str, ...] = ("Deepfakes",),
    identity_map: Path | None = None,
    probe_mode: ProbeMode = "none",
    require_complete: bool = True,
) -> FaceForensicsConfig:
    return FaceForensicsConfig(
        dataset_root=dataset_root,
        dataset_release="synthetic-test-v1",
        access_date=date(2026, 9, 14),
        output_dir=tmp_path / "output",
        split_dir=split_dir,
        compressions=("c23",),
        manipulation_methods=methods,
        expected_counts=expected_counts,
        identity_map_path=identity_map,
        checksum_mode="sha256",
        probe_mode=probe_mode,
        workers=2,
        require_complete_official_sources=require_complete,
        manifest_filename="manifest.jsonl",
        audit_filename="audit.json",
    )


def test_vendored_official_splits_have_expected_coverage_and_hashes() -> None:
    assignments, hashes = load_official_splits(OFFICIAL_SPLITS)

    assert len(assignments) == 1000
    assert list(assignments.values()).count("train") == 720
    assert list(assignments.values()).count("val") == 140
    assert list(assignments.values()).count("test") == 140
    assert hashes == EXPECTED_SPLIT_HASHES


def test_builder_assigns_complete_lineage_and_emits_only_relative_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    split_dir = _write_splits(
        tmp_path,
        {"train": [["000", "001"]], "val": [], "test": []},
    )
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/000.mp4", b"original-zero")
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/001.mp4", b"original-one")
    _video_path(
        dataset_root,
        "manipulated_sequences/Deepfakes/c23/videos/000_001.mp4",
        b"fake-zero-one",
    )
    _video_path(
        dataset_root,
        "manipulated_sequences/Deepfakes/c23/videos/001_000.mp4",
        b"fake-one-zero",
    )
    config = _config(
        tmp_path,
        dataset_root=dataset_root,
        split_dir=split_dir,
        expected_counts={"original": 2, "Deepfakes": 2},
    )

    result = build_manifest(config)

    assert result.audit.passed
    assert len(result.records) == 4
    assert {record.split for record in result.records} == {"train"}
    manipulated = [record for record in result.records if record.label == 1]
    assert {record.lineage_sequence_ids for record in manipulated} == {("000", "001")}
    assert all(not Path(record.relative_path).is_absolute() for record in result.records)
    assert all(str(dataset_root) not in record.relative_path for record in result.records)


def test_manifest_fingerprint_is_deterministic(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    split_dir = _write_splits(tmp_path, {"train": [["000", "001"]]})
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/000.mp4", b"zero")
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/001.mp4", b"one")
    config = _config(
        tmp_path,
        dataset_root=dataset_root,
        split_dir=split_dir,
        expected_counts={"original": 2},
        methods=(),
    )

    first = build_manifest(config)
    second = build_manifest(config)

    assert first.audit.manifest_sha256 == second.audit.manifest_sha256
    assert first.records == second.records


def test_cross_split_target_donor_lineage_rejects_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    split_dir = _write_splits(
        tmp_path,
        {
            "train": [["000", "002"]],
            "val": [["001", "003"]],
            "test": [["004", "005"]],
        },
    )
    _video_path(
        dataset_root,
        "manipulated_sequences/Deepfakes/c23/videos/000_001.mp4",
        b"cross-split",
    )
    (dataset_root / "original_sequences/youtube/c23/videos").mkdir(parents=True)
    config = _config(
        tmp_path,
        dataset_root=dataset_root,
        split_dir=split_dir,
        expected_counts={"original": 0, "Deepfakes": 1},
        require_complete=False,
    )

    result = build_manifest(config)

    assert not result.audit.passed
    assert {issue.code for issue in result.audit.issues} >= {
        "cross_split_lineage",
        "unassigned_video",
    }


def test_byte_identical_files_across_splits_are_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    split_dir = _write_splits(
        tmp_path,
        {"train": [["000", "002"]], "val": [["001", "003"]]},
    )
    shared_bytes = b"identical-video-bytes"
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/000.mp4", shared_bytes)
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/001.mp4", shared_bytes)
    config = _config(
        tmp_path,
        dataset_root=dataset_root,
        split_dir=split_dir,
        expected_counts={"original": 2},
        methods=(),
        require_complete=False,
    )

    result = build_manifest(config)

    assert not result.audit.passed
    assert "cross_split_content_duplicate" in {issue.code for issue in result.audit.issues}


def test_verified_identity_crossing_splits_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    split_dir = _write_splits(
        tmp_path,
        {"train": [["000", "002"]], "val": [["001", "003"]]},
    )
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/000.mp4", b"zero")
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/001.mp4", b"one")
    identity_map = tmp_path / "identities.csv"
    with identity_map.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sequence_id", "identity_id"))
        writer.writeheader()
        writer.writerows(
            [
                {"sequence_id": "000", "identity_id": "person-a"},
                {"sequence_id": "001", "identity_id": "person-a"},
            ]
        )
    config = _config(
        tmp_path,
        dataset_root=dataset_root,
        split_dir=split_dir,
        expected_counts={"original": 2},
        methods=(),
        identity_map=identity_map,
        require_complete=False,
    )

    result = build_manifest(config)

    assert not result.audit.passed
    assert "identity_leakage" in {issue.code for issue in result.audit.issues}


def test_failed_audit_is_written_but_manifest_is_not_published(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    split_dir = _write_splits(tmp_path, {"train": [["000", "001"]]})
    (dataset_root / "original_sequences/youtube/c23/videos").mkdir(parents=True)
    config = _config(
        tmp_path,
        dataset_root=dataset_root,
        split_dir=split_dir,
        expected_counts={"original": 1},
        methods=(),
        require_complete=False,
    )
    result = build_manifest(config)
    previous_manifest = config.output_dir / config.manifest_filename
    previous_manifest.parent.mkdir(parents=True)
    previous_manifest.write_text("previous valid manifest\n", encoding="utf-8")

    manifest_path, audit_path, quarantined_path = write_build_outputs(result, config)

    assert manifest_path is None
    assert audit_path.is_file()
    assert json.loads(audit_path.read_text(encoding="utf-8"))["passed"] is False
    assert not previous_manifest.exists()
    assert quarantined_path is not None
    assert quarantined_path.read_text(encoding="utf-8") == "previous valid manifest\n"


def test_passing_audit_publishes_hash_matched_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    split_dir = _write_splits(tmp_path, {"train": [["000", "001"]]})
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/000.mp4", b"zero")
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/001.mp4", b"one")
    config = _config(
        tmp_path,
        dataset_root=dataset_root,
        split_dir=split_dir,
        expected_counts={"original": 2},
        methods=(),
    )
    result = build_manifest(config)

    manifest_path, audit_path, quarantined_path = write_build_outputs(result, config)

    assert manifest_path is not None and manifest_path.is_file()
    assert audit_path.is_file()
    assert quarantined_path is None
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads(audit_path.read_text(encoding="utf-8"))["manifest_sha256"] == (
        result.audit.manifest_sha256
    )
    verified = load_verified_manifest(manifest_path, audit_path)
    assert len(verified) == 2


def test_verified_loader_rejects_manifest_tampering(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    split_dir = _write_splits(tmp_path, {"train": [["000", "001"]]})
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/000.mp4", b"zero")
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/001.mp4", b"one")
    config = _config(
        tmp_path,
        dataset_root=dataset_root,
        split_dir=split_dir,
        expected_counts={"original": 2},
        methods=(),
    )
    result = build_manifest(config)
    manifest_path, audit_path, _ = write_build_outputs(result, config)
    assert manifest_path is not None
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match="hash mismatch"):
        load_verified_manifest(manifest_path, audit_path)


def test_verified_loader_rejects_path_traversal_even_with_matching_hash(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    record = {
        "schema_version": "1.0",
        "dataset": "faceforensicspp",
        "video_id": "ffpp_0123456789abcdef0123",
        "relative_path": "../private/video.mp4",
        "split": "train",
        "lineage_sequence_ids": ["000"],
        "sha256": "0" * 64,
        "probe": {},
    }
    manifest_content = (json.dumps(record) + "\n").encode()
    manifest_path.write_bytes(manifest_content)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "passed": True,
                "manifest_sha256": hashlib.sha256(manifest_content).hexdigest(),
                "summary": {"records": 1},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestVerificationError, match="safe relative path"):
        load_verified_manifest(manifest_path, audit_path)


def test_required_ffprobe_missing_rejects_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    split_dir = _write_splits(tmp_path, {"train": [["000", "001"]]})
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/000.mp4", b"zero")
    _video_path(dataset_root, "original_sequences/youtube/c23/videos/001.mp4", b"one")
    config = _config(
        tmp_path,
        dataset_root=dataset_root,
        split_dir=split_dir,
        expected_counts={"original": 2},
        methods=(),
        probe_mode="required",
    )
    monkeypatch.setattr("robust_deepfake_detection.data.faceforensics.shutil.which", lambda _: None)

    result = build_manifest(config)

    assert not result.audit.passed
    assert "ffprobe_missing" in {issue.code for issue in result.audit.issues}


def test_empty_self_manipulation_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    split_dir = _write_splits(tmp_path, {"train": [["000", "001"]]})
    (dataset_root / "original_sequences/youtube/c23/videos").mkdir(parents=True)
    _video_path(
        dataset_root,
        "manipulated_sequences/Deepfakes/c23/videos/000_000.mp4",
        b"",
    )
    config = _config(
        tmp_path,
        dataset_root=dataset_root,
        split_dir=split_dir,
        expected_counts={"original": 0, "Deepfakes": 1},
        require_complete=False,
    )

    result = build_manifest(config)

    assert not result.audit.passed
    assert {"self_manipulation_lineage", "empty_video"}.issubset(
        {issue.code for issue in result.audit.issues}
    )


def test_valid_configuration_expands_environment_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_DATA_ROOT", str(tmp_path / "data"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
schema_version: 1
dataset:
  root: ${TEST_DATA_ROOT}/faceforensicspp
  release: test-release
  access_date: 2026-09-14
selection:
  compressions: [c23]
  manipulation_methods: [Deepfakes]
  expected_counts: {original: 2, Deepfakes: 2}
split:
  official_directory: splits
  identity_map: null
  require_complete_official_sources: true
integrity:
  checksum: sha256
  ffprobe: required
  workers: 2
output:
  directory: ${TEST_DATA_ROOT}/manifests
  manifest: test.jsonl
  audit: test.audit.json
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dataset_root == (tmp_path / "data" / "faceforensicspp").resolve()
    assert config.output_dir == (tmp_path / "data" / "manifests").resolve()
    assert config.split_dir == (tmp_path / "splits").resolve()


def test_configuration_rejects_output_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_DATA_ROOT", str(tmp_path / "data"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
schema_version: 1
dataset: {root: "${TEST_DATA_ROOT}/faceforensicspp", release: test, access_date: 2026-09-14}
selection:
  compressions: [c23]
  manipulation_methods: [Deepfakes]
  expected_counts: {original: 1, Deepfakes: 1}
split: {official_directory: splits, identity_map: null}
integrity: {checksum: sha256, ffprobe: required, workers: 2}
output: {directory: "${TEST_DATA_ROOT}/manifests", manifest: ../escape.jsonl, audit: audit.json}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ManifestConfigurationError, match=r"plain \.jsonl filename"):
        load_config(config_path)


def test_default_config_requires_real_access_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", "D:/deepfake-data")

    with pytest.raises(ManifestConfigurationError, match="access_date must use YYYY-MM-DD"):
        load_config(PROJECT_ROOT / "configs" / "data" / "faceforensicspp.yaml")


def test_overlapping_official_splits_are_rejected(tmp_path: Path) -> None:
    split_dir = _write_splits(
        tmp_path,
        {"train": [["000", "001"]], "val": [["000", "002"]]},
    )

    with pytest.raises(ManifestConfigurationError, match="appears in both"):
        load_official_splits(split_dir)
