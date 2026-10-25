"""Synthetic release fixtures test portability/integrity, not model accuracy."""

from __future__ import annotations

import io
import json
import shutil
import socket
import subprocess
import sys
import threading
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robust_industrial_anomaly_detection import bundle_integrity as integrity
from robust_industrial_anomaly_detection.baselines.multiscale import PyramidFeatures
from robust_industrial_anomaly_detection.prototype import release
from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionEngine,
    _CategoryCalibration,
    load_prototype_settings,
)
from robust_industrial_anomaly_detection.prototype.launch import check_port
from robust_industrial_anomaly_detection.prototype.localization import LocalizationModel
from robust_industrial_anomaly_detection.prototype.quality import (
    CaptureQualityModel,
    CategoryQualityModel,
    feature_directions,
)

ROOT = Path(__file__).parents[1]


class FixtureExtractor:
    def __init__(self, **kwargs):
        pass

    def extract_rgb(self, images):
        return tuple(
            PyramidFeatures(
                np.full((28, 28, 64), 0.3, dtype=np.float32),
                np.full((28, 28, 128), 0.5, dtype=np.float32),
                np.full((14, 14, 256), 0.2, dtype=np.float32),
            )
            for _ in images
        )


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "CATEGORIES", ("candle",))
    monkeypatch.setattr(release, "verify_checkpoint", lambda path: None)
    monkeypatch.setattr(release, "FrozenPyramidResNet18", FixtureExtractor)
    engine = InspectionEngine.__new__(InspectionEngine)
    engine.settings = load_prototype_settings(ROOT / "configs/prototype/m10.yaml")
    engine.device, engine.categories = "cpu", ("candle",)
    engine.image_size, engine.seed, engine.image_score_fraction, engine.gaussian_sigma = (
        224,
        42,
        0.01,
        4.0,
    )
    engine._lock, engine._memories = threading.RLock(), {}
    engine._calibration = {
        "candle": _CategoryCalibration(np.arange(64, dtype=float) + 100, 100.0, 0.2)
    }
    engine.m5_metadata = {"manifest_sha256": release.VISA_MANIFEST}
    engine.extractor = FixtureExtractor()
    for name in ("m4_run", "m5_run", "quality_run", "localization_run"):
        path = tmp_path / name
        path.mkdir()
        (path / "metadata.json").write_text('{"fixture":true}', encoding="utf-8")
        setattr(engine, name, path)
    monkeypatch.setattr(
        release, "M10_METADATA", integrity.digest_file(engine.localization_run / "metadata.json")
    )
    engine.checkpoint_path = tmp_path / "not-real-weights.pth"
    engine.checkpoint_path.write_bytes(b"synthetic test fixture, never research evidence")
    original_index = tmp_path / "source-index.npz"
    rng = np.random.default_rng(123)
    np.savez_compressed(
        original_index,
        bank=rng.normal(size=(512, 64)).astype(np.float32),
        projection=rng.normal(size=(448, 64)).astype(np.float32),
        stats=np.asarray("PRIVATE_SAMPLE_ID and D:/private/path must NOT be exported"),
    )
    engine._index_paths = {"candle": original_index}
    keys = feature_directions(1)
    reference = CategoryQualityModel(
        center=dict.fromkeys(keys, 0.0),
        scale=dict.fromkeys(keys, 1.0),
        calibration_scores=tuple([1e9] * 64),
        fit_samples=64,
        calibration_samples=64,
    )
    engine.quality_model = CaptureQualityModel(
        schema_version=1,
        image_size=224,
        alpha=0.02,
        manifest_sha256=release.VISA_MANIFEST,
        categories={"candle": reference},
    )
    spatial = LocalizationModel(
        center=np.zeros((224, 224), np.float32),
        scale=np.ones((224, 224), np.float32),
        raw_high=0.3,
        raw_low=0.2,
        spatial_high=0.3,
        spatial_low=0.2,
        spatial_global_high=0.4,
        minimum_region_pixels=16,
        growth_radius=2,
        fit_samples=64,
        calibration_samples=64,
        normal_pixel_budget=0.005,
        calibration_raw_support_rate=0.002,
        calibration_spatial_support_rate=0.002,
        provenance={
            "geometry": "center-crop",
            "manifest_sha256": release.VISA_MANIFEST,
            "source_index_sha256": integrity.digest_file(original_index),
            "private_note": "PRIVATE_SAMPLE_ID",
        },
    )
    engine._localizations = {"candle": spatial}
    destination = tmp_path / "candidate"
    result = release.export_release(ROOT, destination, engine)
    return engine, destination, result


def _rehash(root, relative):
    manifest = integrity.read_json(root / "manifest.json")
    path = root / relative
    manifest["files"][relative] = {
        "sha256": integrity.digest_file(path),
        "bytes": path.stat().st_size,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_export_is_deterministic_minimal_and_never_overwrites(exported, tmp_path):
    engine, root, result = exported
    second = release.export_release(ROOT, tmp_path / "another-candidate", engine)
    assert result["archive_sha256"] == second["archive_sha256"]
    assert result["manifest_sha256"] == second["manifest_sha256"]
    with pytest.raises(integrity.BundleError, match="already exist"):
        release.export_release(ROOT, root, engine)
    with zipfile.ZipFile(result["archive"]) as archive:
        assert all(info.date_time == release.FIXED_TIME for info in archive.infolist())
        assert not any(Path(name).suffix in {".png", ".jpg", ".pth"} for name in archive.namelist())
    with np.load(root / "runtime/indexes/candle.npz", allow_pickle=False) as payload:
        assert set(payload.files) == {"bank", "projection"}
    assert "PRIVATE_SAMPLE_ID" not in (root / "runtime/runtime.json").read_text()
    with np.load(root / "runtime/localization/candle.npz", allow_pickle=False) as payload:
        assert "PRIVATE_SAMPLE_ID" not in str(payload["metadata"])
    assert not result["published"]


def test_standard_library_verifier_works_relocated_without_site_packages(exported, tmp_path):
    _, root, _ = exported
    moved = tmp_path / "different path with spaces" / "demo"
    shutil.copytree(root, moved)
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(moved / "demo.py"), "--verify"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "verified"


def test_portable_inherits_full_prediction_and_rendering_parity(exported):
    original, root, _ = exported
    portable = release.load_portable_engine(root, original.checkpoint_path)
    pixels = np.random.default_rng(5).integers(20, 240, (240, 320, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    left = original.inspect_bytes(
        stream.getvalue(), filename="fixture.png", category="candle"
    ).public_dict()
    right = portable.inspect_bytes(
        stream.getvalue(), filename="fixture.png", category="candle"
    ).public_dict()
    left.pop("timing_ms"), right.pop("timing_ms")
    assert left == right
    assert right["model_evaluated"] is True
    assert portable.runtime_info()["runtime_source"] == "verified_portable_release"
    assert portable.resource_footprint()["all_category_index_bytes"] > 0
    assert root / "manifest.json" in portable.provenance_files()


def test_recapture_keeps_skipped_inference_semantics(exported, monkeypatch):
    original, root, _ = exported
    portable = release.load_portable_engine(root, original.checkpoint_path)
    model = portable.quality_model
    portable.quality_model = replace(
        model,
        categories={
            "candle": replace(model.categories["candle"], calibration_scores=tuple([0.0] * 64))
        },
    )
    monkeypatch.setattr(
        portable.extractor, "extract_rgb", lambda images: pytest.fail("inference ran")
    )
    stream = io.BytesIO()
    Image.new("RGB", (224, 224), "black").save(stream, format="PNG")
    result = portable.inspect_bytes(stream.getvalue(), filename="fixture.png", category="candle")
    assert result.decision == "recapture" and not result.model_evaluated
    assert result.anomaly_score is None and result.heatmap_data_url is None
    assert result.timing_ms["feature_extraction"] == result.timing_ms["memory_scoring"] == 0


def test_mutated_missing_and_unlisted_files_are_refused(exported):
    _, root, _ = exported
    path = root / "runtime/unlisted.txt"
    path.write_text("not permitted")
    with pytest.raises(integrity.BundleError, match="Unlisted"):
        integrity.verify_bundle(root)
    path.unlink()
    (root / "runtime/calibration.json").write_text("{}")
    with pytest.raises(integrity.BundleError, match="SHA-256"):
        integrity.verify_bundle(root)


def test_unsafe_manifest_paths_and_duplicate_json_keys_refused(exported):
    _, root, _ = exported
    manifest = integrity.read_json(root / "manifest.json")
    manifest["files"]["../escape"] = {"sha256": "0" * 64, "bytes": 1}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(integrity.BundleError, match="escapes"):
        integrity.verify_bundle(root)
    (root / "manifest.json").write_text('{"files":{},"files":{}}')
    with pytest.raises(integrity.BundleError, match="Duplicate JSON"):
        integrity.verify_bundle(root)


def test_non_finite_json_is_not_a_valid_release_contract(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"value":NaN}')
    with pytest.raises(integrity.BundleError, match="Non-finite JSON"):
        integrity.read_json(path)


def test_weights_and_private_environment_cannot_be_declared_as_public_files(exported):
    _, root, _ = exported
    relative = "runtime/forbidden.pth"
    (root / relative).write_bytes(b"should not ship")
    _rehash(root, relative)
    with pytest.raises(integrity.BundleError, match="cannot ship"):
        integrity.verify_bundle(root)


def test_rehashed_but_wrong_numeric_memory_cannot_load(exported):
    engine, root, _ = exported
    relative = "runtime/indexes/candle.npz"
    np.savez_compressed(root / relative, bank=np.zeros((2, 2)), projection=np.zeros((2, 2)))
    _rehash(root, relative)
    with pytest.raises(integrity.BundleError, match="dimensions"):
        release.load_portable_engine(root, engine.checkpoint_path)


def test_rehashed_but_changed_policy_cannot_load(exported):
    engine, root, _ = exported
    relative = "runtime/runtime.json"
    contract = integrity.read_json(root / relative)
    contract["runtime_info"]["anomaly_alpha"] = 0.1
    (root / relative).write_text(json.dumps(contract))
    _rehash(root, relative)
    with pytest.raises(integrity.BundleError, match="decision policy"):
        release.load_portable_engine(root, engine.checkpoint_path)


def test_symlinks_cannot_escape_a_release(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret").write_text("private")
    try:
        (root / "link").symlink_to(tmp_path / "secret")
    except OSError:
        # Exercise the link-rejection boundary without requesting Windows link privileges.
        link = root / "link"
        link.write_text("fixture link target")
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == link or original(path))
    with pytest.raises(integrity.BundleError):
        integrity.safe_path(root, "link")


def test_missing_or_corrupt_checkpoint_never_downloads_implicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(
        integrity, "urlopen", lambda *args, **kwargs: pytest.fail("download attempted")
    )
    path = tmp_path / "absent.pth"
    with pytest.raises(integrity.BundleError, match="Missing or modified"):
        integrity.verify_checkpoint(path)
    path.write_bytes(b"unrelated existing file")
    with pytest.raises(integrity.BundleError):
        integrity.prepare_checkpoint(path)
    assert path.read_bytes() == b"unrelated existing file"


def test_explicit_download_verifies_before_publication_and_cleans_own_temporary(
    tmp_path, monkeypatch
):
    import hashlib

    content = b"synthetic checkpoint bytes, not an actual model"
    monkeypatch.setattr(integrity, "CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest())

    class Response(io.BytesIO):
        def geturl(self):
            return integrity.CHECKPOINT_URL

    monkeypatch.setattr(integrity, "urlopen", lambda *args, **kwargs: Response(content))
    path = tmp_path / "cached.pth"
    integrity.prepare_checkpoint(path)
    assert path.read_bytes() == content and not list(tmp_path.glob("*.part"))
    path.unlink()
    monkeypatch.setattr(integrity, "urlopen", lambda *args, **kwargs: Response(b"bad bytes"))
    with pytest.raises(integrity.BundleError):
        integrity.prepare_checkpoint(path)
    assert not path.exists() and not list(tmp_path.glob("*.part"))


def test_portability_does_not_weaken_occupied_port_protection():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="Nothing was stopped"):
            check_port(listener.getsockname()[1])
