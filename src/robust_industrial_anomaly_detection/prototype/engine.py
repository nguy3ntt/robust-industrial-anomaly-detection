"""Cached calibrated inference for the M6 local inspection prototype."""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import threading
import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

from robust_industrial_anomaly_detection.baselines.multiscale import (
    FrozenPyramidResNet18,
    ProjectedPatchMemory,
    compose_feature_map,
    load_projected_index,
)
from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.evaluation.calibration import (
    conformal_p_values,
    selective_decisions,
)


class PrototypeError(RuntimeError):
    """Raised when the local prototype cannot safely inspect an input."""


@dataclass(frozen=True)
class PrototypeSettings:
    name: str
    host: str
    port: int
    max_upload_bytes: int
    max_image_pixels: int
    max_batch_files: int
    embed_images_in_reports: bool
    primary_model: str
    anomaly_alpha: float
    review_alpha: float
    benchmark_samples_per_category: int
    benchmark_warmup_samples: int


@dataclass(frozen=True)
class InspectionResult:
    filename: str
    category: str
    model: str
    decision: str
    anomaly_score: float
    conformal_p: float
    anomaly_alpha: float
    review_alpha: float
    empirical_score_threshold: float
    pixel_threshold: float
    calibration_samples: int
    anomalous_area_fraction: float
    input_width: int
    input_height: int
    model_input_size: int
    preview_data_url: str
    heatmap_data_url: str
    warnings: tuple[str, ...]
    timing_ms: Mapping[str, float]

    def public_dict(self, *, include_images: bool = True) -> dict[str, object]:
        payload = asdict(self)
        if not include_images:
            payload.pop("preview_data_url", None)
            payload.pop("heatmap_data_url", None)
        return payload


@dataclass(frozen=True)
class _CategoryCalibration:
    scores: np.ndarray
    empirical_threshold: float
    pixel_threshold: float


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PrototypeError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _positive_integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PrototypeError(f"{name} must be a positive integer")
    return value


def _probability(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PrototypeError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 < result < 1.0:
        raise PrototypeError(f"{name} must be strictly between zero and one")
    return result


def _string(section: Mapping[str, object], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise PrototypeError(f"{name} must be a non-empty string")
    return value


def load_prototype_settings(path: Path) -> PrototypeSettings:
    """Load the local prototype's frozen safety and policy settings."""

    try:
        root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "configuration")
    except (OSError, yaml.YAMLError) as exc:
        raise PrototypeError(f"cannot load prototype configuration {path}: {exc}") from exc
    prototype = _mapping(root.get("prototype"), "prototype")
    model = _mapping(root.get("model"), "model")
    benchmark = _mapping(root.get("benchmark"), "benchmark")
    embed = prototype.get("embed_images_in_reports")
    if not isinstance(embed, bool):
        raise PrototypeError("embed_images_in_reports must be boolean")
    port = _positive_integer(prototype, "port")
    if port > 65535:
        raise PrototypeError("port must be at most 65535")
    settings = PrototypeSettings(
        name=_string(prototype, "name"),
        host=_string(prototype, "host"),
        port=port,
        max_upload_bytes=_positive_integer(prototype, "max_upload_bytes"),
        max_image_pixels=_positive_integer(prototype, "max_image_pixels"),
        max_batch_files=_positive_integer(prototype, "max_batch_files"),
        embed_images_in_reports=embed,
        primary_model=_string(model, "primary_model"),
        anomaly_alpha=_probability(model, "anomaly_alpha"),
        review_alpha=_probability(model, "review_alpha"),
        benchmark_samples_per_category=_positive_integer(benchmark, "samples_per_category"),
        benchmark_warmup_samples=_positive_integer(benchmark, "warmup_samples"),
    )
    if settings.anomaly_alpha > settings.review_alpha:
        raise PrototypeError("anomaly_alpha cannot exceed review_alpha")
    if settings.embed_images_in_reports:
        raise PrototypeError("M6 forbids embedding source images in exported reports")
    return settings


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        raise PrototypeError(f"cannot read required artifact {path}: {exc}") from exc


def _verified_json(run_dir: Path, filename: str, metadata: Mapping[str, object]) -> object:
    path = run_dir / filename
    key = f"{path.stem}_sha256"
    if not path.is_file() or metadata.get(key) != sha256_file(path):
        raise PrototypeError(f"artifact failed integrity verification: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrototypeError(f"cannot parse required artifact {path}: {exc}") from exc


def _find_full_m5_run(artifact_root: Path, explicit: Path | None) -> Path:
    runs_root = (artifact_root / "runs").resolve(strict=True)
    candidates = (
        [explicit.resolve(strict=True)]
        if explicit
        else sorted(runs_root.glob("m5-calibration-full-*"), reverse=True)
    )
    for candidate in candidates:
        try:
            candidate.relative_to(runs_root)
            metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        if metadata.get("status") == "complete" and metadata.get("mode") == "full":
            return candidate
    raise PrototypeError("no complete full M5 calibration run is available")


def _safe_filename(value: str) -> str:
    candidate = Path(value.replace("\\", "/")).name.strip()
    return candidate[:180] if candidate else "inspection-image"


def _decode_image(payload: bytes, settings: PrototypeSettings) -> tuple[np.ndarray, int, int]:
    if not payload or len(payload) > settings.max_upload_bytes:
        raise PrototypeError(f"image must contain 1 to {settings.max_upload_bytes:,} bytes")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                if getattr(source, "is_animated", False):
                    raise PrototypeError("animated images are not supported")
                width, height = source.size
                if width * height > settings.max_image_pixels:
                    raise PrototypeError(
                        f"image exceeds the {settings.max_image_pixels:,}-pixel safety limit"
                    )
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                rgb = np.asarray(oriented, dtype=np.uint8)
    except PrototypeError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise PrototypeError("image exceeds Pillow's decompression safety limit") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PrototypeError("file is not a supported decodable image") from exc
    return rgb, width, height


def _fit_rgb(rgb: np.ndarray, image_size: int) -> np.ndarray:
    source = Image.fromarray(rgb, mode="RGB")
    fitted = ImageOps.fit(
        source,
        (image_size, image_size),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )
    return np.asarray(fitted, dtype=np.uint8)


def _png_data_url(rgb: np.ndarray) -> str:
    stream = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(stream, format="PNG", optimize=True)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _score_overlay(
    image: np.ndarray, score_map: np.ndarray, pixel_threshold: float
) -> tuple[np.ndarray, float]:
    minimum = float(np.min(score_map))
    upper = max(float(np.quantile(score_map, 0.995)), pixel_threshold)
    scale = max(upper - minimum, float(np.finfo(np.float32).eps))
    normalized = np.clip((score_map - minimum) / scale, 0.0, 1.0)
    heat_bgr = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.56, heat_rgb, 0.44, 0.0)
    support = (score_map > pixel_threshold).astype(np.uint8)
    contours, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay, float(np.mean(support))


class InspectionEngine:
    """Load one frozen backbone and lazily cache category indexes for inspection."""

    limitations = (
        "Decision support only; not a certified quality-control device.",
        "Calibration assumes capture conditions exchangeable with clean validation normals.",
        "Severe blur, noise, JPEG compression, or geometric shift can cause false alerts.",
        "The heatmap localizes unusual evidence but does not diagnose physical root cause.",
    )

    def __init__(
        self,
        *,
        project_root: Path,
        settings: PrototypeSettings,
        device: str,
        m5_run: Path | None = None,
    ) -> None:
        load_dotenv(project_root / ".env")
        artifact_root_raw = os.environ.get("ANOMALY_ARTIFACT_ROOT")
        if not artifact_root_raw:
            raise PrototypeError("ANOMALY_ARTIFACT_ROOT must be configured")
        self.project_root = project_root
        self.settings = settings
        self.device = device
        self.artifact_root = Path(artifact_root_raw)
        self.m5_run = _find_full_m5_run(self.artifact_root, m5_run)
        self._lock = threading.RLock()
        self._memories: dict[str, ProjectedPatchMemory] = {}
        self._calibration: dict[str, _CategoryCalibration] = {}

        try:
            self.m5_metadata = cast(
                dict[str, object],
                json.loads((self.m5_run / "metadata.json").read_text(encoding="utf-8")),
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise PrototypeError(f"cannot read M5 metadata: {exc}") from exc
        summary = _mapping(
            _verified_json(self.m5_run, "summary.json", self.m5_metadata), "M5 summary"
        )
        if summary.get("primary_model") != settings.primary_model:
            raise PrototypeError("prototype model does not match the frozen M5 primary model")
        policy = _mapping(summary.get("policy"), "M5 policy")
        if (
            policy.get("anomalous") != f"conformal p <= {settings.anomaly_alpha}"
            or policy.get("normal") != f"conformal p > {settings.review_alpha}"
        ):
            raise PrototypeError("prototype decision levels do not match the frozen M5 policy")
        source_m4_raw = summary.get("source_m4_run")
        if not isinstance(source_m4_raw, str):
            raise PrototypeError("M5 summary has no valid M4 source")
        self.m4_run = Path(source_m4_raw).resolve(strict=True)
        self._load_artifact_contract()

        self.extractor = FrozenPyramidResNet18(
            device=device, image_size=self.image_size, batch_size=1
        )
        checkpoint = (
            Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
            / "hub"
            / "checkpoints"
            / Path(urlparse(FrozenPyramidResNet18.weights_url).path).name
        )
        if not checkpoint.is_file() or sha256_file(checkpoint) != self.m4_metadata.get(
            "checkpoint_sha256"
        ):
            raise PrototypeError("frozen backbone checkpoint failed integrity verification")
        self.checkpoint_path = checkpoint

    def _load_artifact_contract(self) -> None:
        for filename in ("calibration_scores.csv", "calibration_parameters.csv"):
            path = self.m5_run / filename
            if self.m5_metadata.get(f"{path.stem}_sha256") != sha256_file(path):
                raise PrototypeError(f"M5 artifact failed integrity verification: {filename}")
        scores = _read_csv(self.m5_run / "calibration_scores.csv")
        parameters = _read_csv(self.m5_run / "calibration_parameters.csv")
        parameter_rows = {
            row["category"]: row
            for row in parameters
            if row["model"] == self.settings.primary_model and row["scope"] == "category"
        }
        score_rows: dict[str, list[float]] = {}
        for row in scores:
            if row["model"] == self.settings.primary_model:
                score_rows.setdefault(row["category"], []).append(float(row["score"]))
        if set(parameter_rows) != set(score_rows):
            raise PrototypeError("M5 category parameters and calibration scores disagree")

        try:
            self.m4_metadata = cast(
                dict[str, object],
                json.loads((self.m4_run / "metadata.json").read_text(encoding="utf-8")),
            )
            resolved = _mapping(
                yaml.safe_load((self.m4_run / "resolved_config.yaml").read_text("utf-8")),
                "M4 resolved configuration",
            )
            index_manifest = _mapping(
                json.loads((self.m4_run / "index_manifest.json").read_text("utf-8")),
                "M4 index manifest",
            )
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise PrototypeError(f"cannot read M4 source contract: {exc}") from exc
        if self.m4_metadata.get("status") != "complete":
            raise PrototypeError("M4 source is not complete")
        if self.m5_metadata.get("source_m4_metadata_sha256") != sha256_file(
            self.m4_run / "metadata.json"
        ):
            raise PrototypeError("M4 metadata no longer matches the M5 provenance")
        if self.m4_metadata.get("index_manifest_sha256") != sha256_file(
            self.m4_run / "index_manifest.json"
        ):
            raise PrototypeError("M4 index manifest failed integrity verification")
        self.seed = int(cast(int | float, resolved["seed"]))
        self.image_size = int(cast(int | float, resolved["image_size"]))
        self.image_score_fraction = float(cast(int | float, resolved["image_score_fraction"]))
        self.gaussian_sigma = float(cast(int | float, resolved["gaussian_sigma"]))
        intervention = _mapping(resolved.get("intervention"), "M4 intervention")
        if intervention.get("identifier") != self.settings.primary_model:
            raise PrototypeError("M4 intervention does not match the prototype model")

        metrics = _read_csv(self.m4_run / "category_metrics.csv")
        pixel_thresholds = {
            row["category"]: float(row["pixel_threshold"])
            for row in metrics
            if row["model"] == self.settings.primary_model and row["condition"] == "clean"
        }
        self._index_paths: dict[str, Path] = {}
        for category, category_scores in score_rows.items():
            relative = f"indexes/intervention/{category}.npz"
            entry = _mapping(index_manifest.get(relative), f"M4 index {relative}")
            path = self.m4_run / relative
            if (
                not path.is_file()
                or path.stat().st_size != entry.get("bytes")
                or sha256_file(path) != entry.get("sha256")
            ):
                raise PrototypeError(f"M4 category index failed verification: {category}")
            parameter = parameter_rows[category]
            expected_count = int(float(parameter["calibration_samples"]))
            if len(category_scores) != expected_count or category not in pixel_thresholds:
                raise PrototypeError(f"incomplete calibration contract for {category}")
            self._calibration[category] = _CategoryCalibration(
                np.asarray(category_scores, dtype=np.float64),
                float(parameter["quantile_threshold"]),
                pixel_thresholds[category],
            )
            self._index_paths[category] = path
        self.categories = tuple(sorted(self._calibration))

    def _memory(self, category: str) -> ProjectedPatchMemory:
        memory = self._memories.get(category)
        if memory is not None:
            return memory
        bank, projection = load_projected_index(self._index_paths[category])
        memory = ProjectedPatchMemory(
            seed=self.seed,
            bank_size=len(bank),
            projected_dimension=projection.shape[1],
            patches_per_image=1,
            candidate_limit=len(bank),
            image_score_fraction=self.image_score_fraction,
            selection="kcenter",
        )
        memory.fit_projected(bank, projection)
        self._memories[category] = memory
        return memory

    def inspect_bytes(self, payload: bytes, *, filename: str, category: str) -> InspectionResult:
        """Inspect one image entirely in memory and return calibrated evidence."""

        if category not in self._calibration:
            raise PrototypeError(f"unsupported category: {category}")
        total_started = time.perf_counter()
        decoded, width, height = _decode_image(payload, self.settings)
        fitted = _fit_rgb(decoded, self.image_size)
        preprocess_ms = (time.perf_counter() - total_started) * 1000.0
        with self._lock:
            feature_started = time.perf_counter()
            pyramid = self.extractor.extract_rgb([fitted])[0]
            feature_map = compose_feature_map(pyramid, "multiscale")
            feature_ms = (time.perf_counter() - feature_started) * 1000.0
            score_started = time.perf_counter()
            scores, raw_maps = self._memory(category).score([feature_map])
            score_ms = (time.perf_counter() - score_started) * 1000.0
        calibration = self._calibration[category]
        anomaly_score = float(scores[0])
        conformal_p = float(conformal_p_values(calibration.scores, [anomaly_score])[0])
        decision = str(
            selective_decisions(
                [conformal_p],
                anomaly_alpha=self.settings.anomaly_alpha,
                review_alpha=self.settings.review_alpha,
            )[0]
        )
        rendering_started = time.perf_counter()
        score_map = cv2.resize(
            raw_maps[0],
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        if self.gaussian_sigma > 0.0:
            score_map = np.asarray(
                cv2.GaussianBlur(score_map, (0, 0), sigmaX=self.gaussian_sigma),
                dtype=np.float32,
            )
        overlay, anomalous_fraction = _score_overlay(fitted, score_map, calibration.pixel_threshold)
        preview = _png_data_url(fitted)
        heatmap = _png_data_url(overlay)
        rendering_ms = (time.perf_counter() - rendering_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return InspectionResult(
            filename=_safe_filename(filename),
            category=category,
            model=self.settings.primary_model,
            decision=decision,
            anomaly_score=anomaly_score,
            conformal_p=conformal_p,
            anomaly_alpha=self.settings.anomaly_alpha,
            review_alpha=self.settings.review_alpha,
            empirical_score_threshold=calibration.empirical_threshold,
            pixel_threshold=calibration.pixel_threshold,
            calibration_samples=len(calibration.scores),
            anomalous_area_fraction=anomalous_fraction,
            input_width=width,
            input_height=height,
            model_input_size=self.image_size,
            preview_data_url=preview,
            heatmap_data_url=heatmap,
            warnings=self.limitations,
            timing_ms={
                "preprocess": preprocess_ms,
                "feature_extraction": feature_ms,
                "memory_scoring": score_ms,
                "rendering": rendering_ms,
                "total": total_ms,
            },
        )

    def runtime_info(self) -> dict[str, object]:
        """Return public model/policy information for the local UI."""

        return {
            "name": self.settings.name,
            "model": self.settings.primary_model,
            "categories": self.categories,
            "device": self.device,
            "anomaly_alpha": self.settings.anomaly_alpha,
            "review_alpha": self.settings.review_alpha,
            "max_upload_bytes": self.settings.max_upload_bytes,
            "max_batch_files": self.settings.max_batch_files,
            "images_persisted": False,
            "images_embedded_in_reports": False,
            "limitations": self.limitations,
            "source_m5_run": self.m5_run.name,
        }

    def resource_footprint(self) -> dict[str, int]:
        """Return exact on-disk sizes of runtime model artifacts."""

        indexes = sum(path.stat().st_size for path in self._index_paths.values())
        calibration = sum(
            (self.m5_run / name).stat().st_size
            for name in ("calibration_scores.csv", "calibration_parameters.csv")
        )
        checkpoint_bytes = self.checkpoint_path.stat().st_size
        return {
            "backbone_checkpoint_bytes": checkpoint_bytes,
            "all_category_index_bytes": indexes,
            "calibration_artifact_bytes": calibration,
            "total_runtime_artifact_bytes": checkpoint_bytes + indexes + calibration,
        }
