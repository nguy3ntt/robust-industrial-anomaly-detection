"""Normal-only capture-quality calibration for the local inspector."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np

from robust_industrial_anomaly_detection.evaluation.calibration import conformal_p_values

QualityDirection = Literal["two_sided", "lower", "upper"]

FEATURE_DIRECTIONS: dict[str, QualityDirection] = {
    "brightness": "two_sided",
    "contrast": "two_sided",
    "sharpness": "lower",
    "noise": "upper",
    "clipping": "upper",
    "blockiness": "upper",
    "edge_centroid_x": "two_sided",
    "edge_centroid_y": "two_sided",
}

FEATURE_LABELS = {
    "brightness": "illumination",
    "contrast": "contrast",
    "sharpness": "focus/blur",
    "noise": "sensor noise",
    "clipping": "exposure clipping",
    "blockiness": "compression artifacts",
    "edge_centroid_x": "horizontal framing",
    "edge_centroid_y": "vertical framing",
    "log_focus_ratio": "contrast-normalized focus",
    "log_flat_noise": "flat-region noise",
}

V2_FEATURE_DIRECTIONS: dict[str, QualityDirection] = {
    **FEATURE_DIRECTIONS,
    "log_focus_ratio": "lower",
    "log_flat_noise": "upper",
}


def feature_directions(schema_version: int) -> Mapping[str, QualityDirection]:
    """Keep historical measurements immutable and reject unknown recipes."""

    if schema_version == 1:
        return FEATURE_DIRECTIONS
    if schema_version == 2:
        return V2_FEATURE_DIRECTIONS
    raise QualityGateError("quality model has an unsupported schema")


class QualityGateError(ValueError):
    """Raised when a quality model or image violates the gate contract."""


@dataclass(frozen=True)
class CaptureMetrics:
    """Low-cost capture measurements computed on the evaluated RGB crop."""

    brightness: float
    contrast: float
    sharpness: float
    noise: float
    clipping: float
    blockiness: float
    edge_centroid_x: float
    edge_centroid_y: float

    def values(self) -> dict[str, float]:
        """Return metrics with stable feature ordering."""

        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class CategoryQualityModel:
    """One category's train-normal reference and validation-normal calibration."""

    center: Mapping[str, float]
    scale: Mapping[str, float]
    calibration_scores: tuple[float, ...]
    fit_samples: int
    calibration_samples: int


@dataclass(frozen=True)
class CaptureQualityModel:
    """Serializable normal-only quality model used before anomaly inference."""

    schema_version: int
    image_size: int
    alpha: float
    manifest_sha256: str
    categories: Mapping[str, CategoryQualityModel]


@dataclass(frozen=True)
class QualityDecision:
    """Capture disposition and the measurements that produced it."""

    status: Literal["pass", "recapture"]
    p_value: float
    score: float
    alpha: float
    metrics: Mapping[str, float]
    components: Mapping[str, float]
    contributors: tuple[str, ...]
    calibration_samples: int


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise QualityGateError("capture-quality input must be an RGB uint8 image")
    if min(rgb.shape[:2]) < 16:
        raise QualityGateError("capture-quality input must be at least 16 pixels per side")
    return rgb


def _blockiness(gray: np.ndarray) -> float:
    vertical = np.abs(np.diff(gray, axis=1))
    horizontal = np.abs(np.diff(gray, axis=0))
    vertical_boundary = vertical[:, 7::8]
    horizontal_boundary = horizontal[7::8, :]
    boundary_values = np.concatenate(
        (vertical_boundary.reshape(-1), horizontal_boundary.reshape(-1))
    )
    vertical_mask = np.ones(vertical.shape[1], dtype=bool)
    horizontal_mask = np.ones(horizontal.shape[0], dtype=bool)
    vertical_mask[7::8] = False
    horizontal_mask[7::8] = False
    interior_values = np.concatenate(
        (vertical[:, vertical_mask].reshape(-1), horizontal[horizontal_mask, :].reshape(-1))
    )
    boundary = float(np.mean(boundary_values)) if boundary_values.size else 0.0
    interior = float(np.mean(interior_values)) if interior_values.size else 0.0
    return boundary / max(interior, 1.0 / 255.0)


def capture_metrics(image: np.ndarray) -> CaptureMetrics:
    """Measure focus, noise, exposure, compression, and framing proxies."""

    rgb = _validate_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(horizontal * horizontal + vertical * vertical)
    weight = float(np.sum(magnitude))
    if weight <= float(np.finfo(np.float32).eps):
        centroid_x = 0.0
        centroid_y = 0.0
    else:
        x = np.linspace(-1.0, 1.0, gray.shape[1], dtype=np.float32)
        y = np.linspace(-1.0, 1.0, gray.shape[0], dtype=np.float32)
        centroid_x = float(np.sum(magnitude * x[None, :]) / weight)
        centroid_y = float(np.sum(magnitude * y[:, None]) / weight)
    return CaptureMetrics(
        brightness=float(np.mean(gray)),
        contrast=float(np.std(gray)),
        sharpness=float(np.log1p(np.var(laplacian))),
        noise=float(np.median(np.abs(residual))),
        clipping=float(np.mean((gray <= 0.02) | (gray >= 0.98))),
        blockiness=_blockiness(gray),
        edge_centroid_x=centroid_x,
        edge_centroid_y=centroid_y,
    )


def capture_measurements(image: np.ndarray, *, schema_version: int = 1) -> dict[str, float]:
    """Versioned focus/noise proxies, always on the exact anomaly-model input.

    V2 adds log fine/coarse energy (insensitive to multiplicative contrast) and
    a robust diagonal high-pass noise estimate on the least structured quarter
    of pixels. Logs avoid the bounded lower-tail deviation of raw sharpness.
    Neither proxy identifies physical camera faults or separates every defect.
    """

    feature_directions(schema_version)
    values = capture_metrics(image).values()
    if schema_version == 1:
        return values
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = float(np.mean(dx * dx + dy * dy))
    values["log_focus_ratio"] = float(
        np.log(float(np.mean(laplacian * laplacian)) / max(energy, 1e-8) + 1e-8)
    )
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    sx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    structure = (sx * sx + sy * sy)[1:-1, 1:-1]
    flat = structure <= np.quantile(structure, 0.25)
    kernel = np.asarray([[1, -1], [-1, 1]], dtype=np.float32) / 2.0
    residual = cv2.filter2D(gray, cv2.CV_32F, kernel)[1:-1, 1:-1]
    noise = float(np.median(np.abs(residual[flat]))) / 0.67449
    values["log_flat_noise"] = float(np.log(noise / max(float(np.std(gray)), 1e-4) + 1e-6))
    return values


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - center)))
    q25, q75 = np.quantile(values, (0.25, 0.75))
    iqr_scale = float(q75 - q25) / 1.349
    scale = max(mad_scale, iqr_scale, 1e-5)
    return center, scale


def quality_components(
    metrics: CaptureMetrics | Mapping[str, float],
    center: Mapping[str, float],
    scale: Mapping[str, float],
    *,
    schema_version: int = 1,
) -> dict[str, float]:
    """Convert raw measurements into directional robust deviations."""

    values = metrics.values() if isinstance(metrics, CaptureMetrics) else metrics
    directions = feature_directions(schema_version)
    if set(values) != set(directions):
        raise QualityGateError("quality metrics do not match the model feature schema")
    if set(center) != set(values) or set(scale) != set(values):
        raise QualityGateError("quality reference does not match the feature schema")
    result: dict[str, float] = {}
    for name, direction in directions.items():
        observed = float(values[name])
        reference_center = float(center[name])
        denominator = float(scale[name])
        if not np.isfinite(observed) or not np.isfinite(reference_center):
            raise QualityGateError(f"non-finite quality value for {name}")
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise QualityGateError(f"invalid quality scale for {name}")
        standardized = (observed - reference_center) / denominator
        if direction == "lower":
            component = max(0.0, -standardized)
        elif direction == "upper":
            component = max(0.0, standardized)
        else:
            component = abs(standardized)
        result[name] = float(component)
    return result


def fit_category_quality_model(
    fit_metrics: Sequence[CaptureMetrics | Mapping[str, float]],
    calibration_metrics: Sequence[CaptureMetrics | Mapping[str, float]],
    *,
    schema_version: int = 1,
) -> CategoryQualityModel:
    """Fit robust train-normal references and validation-normal gate scores."""

    if len(fit_metrics) < 16 or len(calibration_metrics) < 16:
        raise QualityGateError(
            "quality calibration requires at least 16 fit and 16 calibration samples"
        )
    directions = feature_directions(schema_version)
    fit_values = [
        metrics.values() if isinstance(metrics, CaptureMetrics) else metrics
        for metrics in fit_metrics
    ]
    if any(set(row) != set(directions) for row in fit_values):
        raise QualityGateError("quality fitting metrics do not match the feature schema")
    center: dict[str, float] = {}
    scale: dict[str, float] = {}
    for name in directions:
        values = np.asarray([row[name] for row in fit_values], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise QualityGateError("quality fitting measurements must be finite")
        center[name], scale[name] = _robust_location_scale(values)
    calibration_scores = tuple(
        max(quality_components(metrics, center, scale, schema_version=schema_version).values())
        for metrics in calibration_metrics
    )
    return CategoryQualityModel(
        center=center,
        scale=scale,
        calibration_scores=calibration_scores,
        fit_samples=len(fit_metrics),
        calibration_samples=len(calibration_metrics),
    )


def evaluate_capture_quality(
    image: np.ndarray,
    *,
    category: str,
    model: CaptureQualityModel,
) -> QualityDecision:
    """Apply the category gate and return a fail-closed capture disposition."""

    reference = model.categories.get(category)
    if reference is None:
        raise QualityGateError(f"quality model has no category {category!r}")
    image = _validate_rgb(image)
    if image.shape[:2] != (model.image_size, model.image_size):
        raise QualityGateError("capture-quality input geometry does not match the fitted model")
    metrics = capture_measurements(image, schema_version=model.schema_version)
    components = quality_components(
        metrics, reference.center, reference.scale, schema_version=model.schema_version
    )
    score = max(components.values())
    calibration = np.asarray(reference.calibration_scores, dtype=np.float64)
    p_value = float(conformal_p_values(calibration, [score])[0])
    status: Literal["pass", "recapture"] = "recapture" if p_value <= model.alpha else "pass"
    ordered = sorted(components, key=components.__getitem__, reverse=True)
    contributors = tuple(FEATURE_LABELS[name] for name in ordered[:2])
    return QualityDecision(
        status=status,
        p_value=p_value,
        score=score,
        alpha=model.alpha,
        metrics=metrics,
        components=components,
        contributors=contributors,
        calibration_samples=reference.calibration_samples,
    )


def quality_model_dict(model: CaptureQualityModel) -> dict[str, object]:
    """Return a stable JSON-ready quality-model representation."""

    return {
        "schema_version": model.schema_version,
        "image_size": model.image_size,
        "alpha": model.alpha,
        "manifest_sha256": model.manifest_sha256,
        "feature_directions": dict(feature_directions(model.schema_version)),
        "categories": {
            category: {
                "center": dict(reference.center),
                "scale": dict(reference.scale),
                "calibration_scores": list(reference.calibration_scores),
                "fit_samples": reference.fit_samples,
                "calibration_samples": reference.calibration_samples,
            }
            for category, reference in sorted(model.categories.items())
        },
    }


def load_quality_model(path: Path) -> CaptureQualityModel:
    """Load and strictly validate a serialized quality model."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityGateError(f"cannot load quality model {path}: {exc}") from exc
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise QualityGateError("quality model has an unsupported schema")
    version = payload["schema_version"]
    directions = feature_directions(version)
    if payload.get("feature_directions") != directions:
        raise QualityGateError("quality model feature directions have changed")
    image_size = payload.get("image_size")
    alpha = payload.get("alpha")
    manifest_sha256 = payload.get("manifest_sha256")
    categories_raw = payload.get("categories")
    if not isinstance(image_size, int) or image_size < 16:
        raise QualityGateError("quality model image_size is invalid")
    if (
        not isinstance(alpha, (int, float))
        or not np.isfinite(float(alpha))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise QualityGateError("quality model alpha is invalid")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise QualityGateError("quality model manifest hash is invalid")
    if not isinstance(categories_raw, dict) or not categories_raw:
        raise QualityGateError("quality model has no categories")
    categories: dict[str, CategoryQualityModel] = {}
    for category, raw in categories_raw.items():
        if not isinstance(category, str) or not isinstance(raw, dict):
            raise QualityGateError("quality model category entry is invalid")
        center = raw.get("center")
        scale = raw.get("scale")
        scores = raw.get("calibration_scores")
        fit_samples = raw.get("fit_samples")
        calibration_samples = raw.get("calibration_samples")
        if (
            not isinstance(center, dict)
            or not isinstance(scale, dict)
            or not isinstance(scores, list)
        ):
            raise QualityGateError(f"quality model category {category} is incomplete")
        if type(fit_samples) is not int or type(calibration_samples) is not int:
            raise QualityGateError(f"quality model category {category} counts are invalid")
        if set(center) != set(directions) or set(scale) != set(directions):
            raise QualityGateError(f"quality model category {category} features are invalid")
        try:
            reference = CategoryQualityModel(
                center={
                    name: float(cast(str | int | float, value))
                    for name, value in cast(dict[str, object], center).items()
                },
                scale={
                    name: float(cast(str | int | float, value))
                    for name, value in cast(dict[str, object], scale).items()
                },
                calibration_scores=tuple(float(cast(str | int | float, value)) for value in scores),
                fit_samples=fit_samples,
                calibration_samples=calibration_samples,
            )
        except (TypeError, ValueError) as exc:
            raise QualityGateError(
                f"quality model category {category} contains non-numeric values"
            ) from exc
        quality_components(
            {name: float(reference.center[name]) for name in directions},
            reference.center,
            reference.scale,
            schema_version=version,
        )
        if len(reference.calibration_scores) != reference.calibration_samples:
            raise QualityGateError(f"quality model category {category} score count is invalid")
        if (
            not reference.calibration_scores
            or not np.all(np.isfinite(reference.calibration_scores))
            or min(reference.calibration_scores) < 0
        ):
            raise QualityGateError(f"quality model category {category} scores are invalid")
        if reference.fit_samples < 16 or reference.calibration_samples < 16:
            raise QualityGateError(f"quality model category {category} has too few samples")
        if version == 2 and 1.0 / (reference.calibration_samples + 1) > float(alpha):
            raise QualityGateError("v2 quality calibration cannot resolve the declared alpha")
        categories[category] = reference
    return CaptureQualityModel(
        schema_version=version,
        image_size=image_size,
        alpha=float(alpha),
        manifest_sha256=manifest_sha256,
        categories=categories,
    )
