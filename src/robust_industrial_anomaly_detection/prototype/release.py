"""Minimal, deterministic public runtime export and relocation-safe loading.

Only initialization differs from the research inspector. Image decoding, quality
gating, memory scoring, calibration, localization and rendering stay inherited.
"""

from __future__ import annotations

import io
import json
import shutil
import threading
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from robust_industrial_anomaly_detection.baselines.multiscale import (
    FrozenPyramidResNet18,
    load_projected_index,
)
from robust_industrial_anomaly_detection.bundle_integrity import (
    CHECKPOINT_SHA256,
    FORMAT,
    BundleError,
    digest_file,
    read_json,
    safe_path,
    verify_bundle,
    verify_checkpoint,
)
from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionEngine,
    _CategoryCalibration,
    load_prototype_settings,
)
from robust_industrial_anomaly_detection.prototype.localization import load_localization_model
from robust_industrial_anomaly_detection.prototype.quality import (
    load_quality_model,
    quality_model_dict,
)

VISA_MANIFEST = "73283c2704ed18183e5394fff699b0f8d916113393e53ba53986adaf6f32cc6e"
M10_METADATA = "43fa655f235282152cfa29dae68acb30c608d5106858012b5e9ea8dc1a2c158b"
M10_REPORT_PACKAGE = "c095cec6d6fc1faf69766172f7ca0ab9ff3ce5837ec9da981ef61319c8f93537"
CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", "utf-8")


def _zip_item(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content, compresslevel=9)


def _numeric_archive(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x") as archive:
        for name, array in sorted(values.items()):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
            _zip_item(archive, name + ".npy", stream.getvalue())


def _checked_arrays(path: Path, *, index: bool) -> None:
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 2_000_000:
            raise BundleError("Runtime numeric archive exceeds its uncompressed size limit.")
        names = set(archive.namelist())
        expected = (
            {"bank.npy", "projection.npy"} if index else {"center.npy", "scale.npy", "metadata.npy"}
        )
        if names != expected or len(archive.infolist()) != len(expected):
            raise BundleError("Unexpected numeric runtime members.")
    if index:
        bank, projection = load_projected_index(path)
        if (
            bank.shape != (512, 64)
            or projection.shape != (448, 64)
            or not (np.all(np.isfinite(bank)) and np.all(np.isfinite(projection)))
        ):
            raise BundleError("Runtime memory dimensions or numeric values differ from M10.")


def export_release(project: Path, destination: Path, engine: InspectionEngine) -> dict[str, Any]:
    """Create a local release candidate; never upload or include a checkpoint."""

    settings = load_prototype_settings(project / "configs/prototype/m10.yaml")
    if engine.settings != settings or engine.device != "cpu" or engine.categories != CATEGORIES:
        raise BundleError("Export requires the complete, unchanged CPU M10 inspector.")
    if engine.m5_metadata.get("manifest_sha256") != VISA_MANIFEST or (
        engine.localization_run is None
        or digest_file(engine.localization_run / "metadata.json") != M10_METADATA
        or engine.quality_model is None
        or engine.quality_run is None
        or engine.quality_model.schema_version != 1
    ):
        raise BundleError(
            "Only the pinned original VisA/M8/M10 references are eligible for this release."
        )
    verify_checkpoint(engine.checkpoint_path)
    report_manifest = read_json(project / "reports/m10/manifest.json")
    if report_manifest.get("package_sha256") != M10_REPORT_PACKAGE:
        raise BundleError("The published M10 report package differs from the release policy.")
    report_paths = {
        "reports/M10_LOCALIZATION.md",
        "reports/m10/category_metrics.csv",
        "reports/m10/macro_metrics.csv",
    }
    if set(report_manifest["output_hashes"]) != report_paths or any(
        digest_file(project / name) != report_manifest["output_hashes"][name]
        for name in report_paths
    ):
        raise BundleError("Required aggregate report files failed verification.")
    archive_path = destination.with_name(destination.name + ".zip")
    checksum = destination.with_name(destination.name + ".SHA256SUMS")
    if any(path.exists() for path in (destination, archive_path, checksum)):
        raise BundleError("Release destinations already exist; choose a new --output directory.")
    destination.mkdir(parents=True, exist_ok=False)
    runtime = destination / "runtime"
    runtime.mkdir()
    calibration = {}
    source_indexes = {}
    for category in engine.categories:
        original = engine._index_paths[category]
        bank, projection = load_projected_index(original)
        index = runtime / "indexes" / f"{category}.npz"
        _numeric_archive(index, {"bank": bank, "projection": projection})
        _checked_arrays(index, index=True)
        source_indexes[category] = digest_file(original)
        reference = engine._calibration[category]
        calibration[category] = {
            "scores": reference.scores.tolist(),
            "empirical_threshold": reference.empirical_threshold,
            "pixel_threshold": reference.pixel_threshold,
        }
        spatial = engine._localizations[category]
        metadata = asdict(spatial)
        center, scale = metadata.pop("center"), metadata.pop("scale")
        metadata["schema_version"] = 3
        metadata["provenance"] = {
            key: spatial.provenance[key]
            for key in ("geometry", "manifest_sha256", "source_index_sha256")
        }
        _numeric_archive(
            runtime / "localization" / f"{category}.npz",
            {
                "center": center,
                "scale": scale,
                "metadata": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
            },
        )
    _write_json(runtime / "calibration.json", calibration)
    _write_json(runtime / "quality.json", quality_model_dict(engine.quality_model))
    _write_json(
        runtime / "runtime.json",
        {
            "format": FORMAT,
            "manifest_sha256": VISA_MANIFEST,
            "image_size": engine.image_size,
            "seed": engine.seed,
            "image_score_fraction": engine.image_score_fraction,
            "gaussian_sigma": engine.gaussian_sigma,
            "runtime_info": engine.runtime_info(),
            "source_index_sha256": source_indexes,
            "source_m5_metadata_sha256": digest_file(engine.m5_run / "metadata.json"),
            "source_m4_metadata_sha256": digest_file(engine.m4_run / "metadata.json"),
            "source_quality_metadata_sha256": digest_file(engine.quality_run / "metadata.json"),
            "source_localization_metadata_sha256": M10_METADATA,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
    )
    copied: dict[str, str] = {
        "scripts/demo_entry.py": "demo.py",
        "docs/DEMO_RELEASE_README.md": "README.md",
        "docs/DEMO_THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
        "LICENSE": "LICENSE",
        "pyproject.toml": "pyproject.toml",
        "uv.lock": "uv.lock",
        "configs/prototype/m10.yaml": "configs/prototype/m10.yaml",
        "docs/MODEL_CARD.md": "docs/MODEL_CARD.md",
        "docs/SYSTEM_CARD.md": "docs/SYSTEM_CARD.md",
        "docs/M10_LOCALIZATION_PROTOCOL.md": "docs/M10_LOCALIZATION_PROTOCOL.md",
        "docs/M10_MANUAL_TESTING.md": "docs/M10_MANUAL_TESTING.md",
        "docs/PUBLIC_DEMO_RELEASE.md": "docs/PUBLIC_DEMO_RELEASE.md",
        "docs/datasets/VISA.md": "docs/datasets/VISA.md",
        "reports/m10/manifest.json": "reports/m10/manifest.json",
        **{name: name for name in report_paths},
    }
    package = project / "src/robust_industrial_anomaly_detection"
    copied.update(
        {
            path.relative_to(project).as_posix(): path.relative_to(project).as_posix()
            for path in package.rglob("*.py")
        }
    )
    for name in (
        "py.typed",
        "prototype/static/workstation.html",
        "prototype/static/workstation.css",
        "prototype/static/app.js",
    ):
        relative = "src/robust_industrial_anomaly_detection/" + name
        copied[relative] = relative
    duplicates = [
        (name, name)
        for name in (
            "docs/DEMO_RELEASE_README.md",
            "docs/DEMO_THIRD_PARTY_NOTICES.md",
        )
    ]
    for source, target in sorted([*copied.items(), *duplicates]):
        original = safe_path(project, source)
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, output)
    files = {
        path.relative_to(destination).as_posix(): {
            "sha256": digest_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    _write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "format": FORMAT,
            "files": files,
            "source_images_included": False,
            "weights_included": False,
            "test_predictions_included": False,
            "normal_reference_features_included": True,
            "normal_calibration_scores_included": True,
            "sample_identifiers_included": False,
            "source_localization_metadata_sha256": M10_METADATA,
            "dataset_manifest_sha256": VISA_MANIFEST,
            "publication_performed": False,
        },
    )
    verify_bundle(destination)
    with zipfile.ZipFile(archive_path, "x") as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                _zip_item(
                    archive,
                    "industrial-anomaly-demo/" + path.relative_to(destination).as_posix(),
                    path.read_bytes(),
                )
    archive_hash, manifest_hash = (
        digest_file(archive_path),
        digest_file(destination / "manifest.json"),
    )
    with checksum.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{archive_hash}  {archive_path.name}\n")
        stream.write(f"{manifest_hash}  industrial-anomaly-demo/manifest.json\n")
    return {
        "directory": str(destination),
        "archive": str(archive_path),
        "checksums": str(checksum),
        "archive_sha256": archive_hash,
        "manifest_sha256": manifest_hash,
        "archive_bytes": archive_path.stat().st_size,
        "files": len(files),
        "published": False,
    }


class PortableInspectionEngine(InspectionEngine):
    """A verified export uses the SAME inherited inference, not a second model recipe."""

    def __init__(self, root: Path, checkpoint: Path) -> None:
        self._release_manifest = verify_bundle(root)
        verify_checkpoint(checkpoint)
        self._release_root = root.resolve()
        self.checkpoint_path = checkpoint.resolve()
        self.settings = load_prototype_settings(root / "configs/prototype/m10.yaml")
        if (
            self.settings.name != "m10-localization-inspector"
            or self.settings.primary_model != "multiscale_augmented_coreset_512"
            or self.settings.anomaly_alpha != 0.05
            or self.settings.review_alpha != 0.20
            or not self.settings.quality_gate_enabled
            or self.settings.quality_experiment != "m8-quality-gate"
            or self.settings.localization_experiment != "m10-localization"
        ):
            raise BundleError("The bundled settings differ from the fixed M10/M8 policy.")
        contract = read_json(root / "runtime/runtime.json")
        if (
            contract.get("format"),
            contract.get("manifest_sha256"),
            contract.get("checkpoint_sha256"),
        ) != (FORMAT, VISA_MANIFEST, CHECKPOINT_SHA256) or contract.get(
            "source_localization_metadata_sha256"
        ) != M10_METADATA:
            raise BundleError("Portable runtime does not match the fixed release source.")
        self.image_size = contract["image_size"]
        self.seed = contract["seed"]
        self.image_score_fraction = contract["image_score_fraction"]
        self.gaussian_sigma = contract["gaussian_sigma"]
        if (self.image_size, self.image_score_fraction, self.gaussian_sigma) != (224, 0.01, 4.0):
            raise BundleError(
                "Portable geometry or score aggregation differs from the frozen inspector."
            )
        self.device = "cpu"
        self._public_runtime = contract["runtime_info"]
        if not isinstance(self._public_runtime, dict) or not isinstance(
            contract["source_index_sha256"], dict
        ):
            raise BundleError("Malformed runtime information or source-index receipts.")
        self.categories = CATEGORIES
        if self._public_runtime.get("categories") != list(CATEGORIES) or (
            self._public_runtime.get("model") != self.settings.primary_model
            or self._public_runtime.get("anomaly_alpha") != self.settings.anomaly_alpha
            or self._public_runtime.get("review_alpha") != self.settings.review_alpha
            or self._public_runtime.get("quality_schema_version") != 1
        ):
            raise BundleError("Portable category coverage or decision policy differs.")
        self._lock = threading.RLock()
        self._memories = {}
        self._calibration = {}
        self._index_paths = {}
        self._localizations = {}
        self.quality_model = load_quality_model(root / "runtime/quality.json")
        if self.quality_model.manifest_sha256 != VISA_MANIFEST or (
            tuple(sorted(self.quality_model.categories)) != CATEGORIES
            or self.quality_model.schema_version != 1
            or self.quality_model.alpha != 0.02
            or self.quality_model.image_size != 224
        ):
            raise BundleError("Portable capture gate is not the original normal-only reference.")
        calibrations = read_json(root / "runtime/calibration.json")
        if tuple(sorted(calibrations)) != CATEGORIES:
            raise BundleError("Portable calibration coverage is incomplete.")
        for category in CATEGORIES:
            raw = calibrations[category]
            scores = np.asarray(raw["scores"], dtype=np.float64)
            thresholds = np.asarray(
                [raw["empirical_threshold"], raw["pixel_threshold"]], dtype=float
            )
            if (
                scores.ndim != 1
                or len(scores) < 50
                or not np.all(np.isfinite(scores))
                or (not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0))
            ):
                raise BundleError("Portable normal calibration has invalid scores or thresholds.")
            self._calibration[category] = _CategoryCalibration(scores, *thresholds.tolist())
            index = root / f"runtime/indexes/{category}.npz"
            _checked_arrays(index, index=True)
            self._index_paths[category] = index
            spatial = root / f"runtime/localization/{category}.npz"
            _checked_arrays(spatial, index=False)
            model = load_localization_model(spatial)
            if (
                model.center.shape != (224, 224)
                or model.provenance.get("manifest_sha256") != VISA_MANIFEST
                or (
                    model.provenance.get("geometry") != "center-crop"
                    or model.provenance.get("source_index_sha256")
                    != contract["source_index_sha256"][category]
                )
            ):
                raise BundleError(
                    "Portable localization references differ from their source memory."
                )
            self._localizations[category] = model
        self.extractor = FrozenPyramidResNet18(
            device="cpu",
            image_size=224,
            batch_size=1,
            checkpoint_path=self.checkpoint_path,
        )

    def runtime_info(self) -> dict[str, object]:
        return {
            **self._public_runtime,
            "runtime_source": "verified_portable_release",
            "bundle_manifest_sha256": digest_file(self._release_root / "manifest.json"),
        }

    def provenance_files(self) -> tuple[Path, ...]:
        return (
            self.checkpoint_path,
            self._release_root / "manifest.json",
            *(self._release_root / name for name in sorted(self._release_manifest["files"])),
        )

    def resource_footprint(self) -> dict[str, int]:
        root = self._release_root
        indexes = sum(path.stat().st_size for path in self._index_paths.values())
        calibration = (root / "runtime/calibration.json").stat().st_size
        quality = (root / "runtime/quality.json").stat().st_size
        localization = sum(
            path.stat().st_size for path in (root / "runtime/localization").glob("*.npz")
        )
        weights = self.checkpoint_path.stat().st_size
        return {
            "backbone_checkpoint_bytes": weights,
            "all_category_index_bytes": indexes,
            "calibration_artifact_bytes": calibration,
            "quality_gate_artifact_bytes": quality,
            "localization_artifact_bytes": localization,
            "total_runtime_artifact_bytes": weights
            + indexes
            + calibration
            + quality
            + localization,
        }


def load_portable_engine(root: Path, checkpoint: Path) -> PortableInspectionEngine:
    try:
        return PortableInspectionEngine(root, checkpoint)
    except (KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise BundleError(f"Malformed portable runtime: {exc}") from exc
