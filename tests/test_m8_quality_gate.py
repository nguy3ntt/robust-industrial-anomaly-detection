from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionEngine,
    _CategoryCalibration,
    load_prototype_settings,
)
from robust_industrial_anomaly_detection.prototype.quality import (
    CaptureQualityModel,
    CategoryQualityModel,
    QualityGateError,
    capture_metrics,
    evaluate_capture_quality,
    fit_category_quality_model,
    load_quality_model,
    quality_model_dict,
)

ROOT = Path(__file__).parents[1]


def _pattern(offset: int = 0) -> np.ndarray:
    y, x = np.indices((64, 64))
    values = ((x * 7 + y * 5 + offset) % 256).astype(np.uint8)
    return np.stack((values, np.roll(values, 3, axis=0), values), axis=2)


def _model() -> CaptureQualityModel:
    fit = [capture_metrics(_pattern(index)) for index in range(16)]
    calibration = [capture_metrics(_pattern(index + 16)) for index in range(16)]
    return CaptureQualityModel(
        schema_version=1,
        image_size=64,
        alpha=0.05,
        manifest_sha256="a" * 64,
        categories={"part": fit_category_quality_model(fit, calibration)},
    )


def test_quality_metrics_react_to_blur_and_noise() -> None:
    image = _pattern()
    clean = capture_metrics(image)
    blurred = capture_metrics(cv2.GaussianBlur(image, (0, 0), sigmaX=2.5))
    rng = np.random.default_rng(7)
    noisy = np.clip(image.astype(np.float32) + rng.normal(0, 35, image.shape), 0, 255).astype(
        np.uint8
    )

    assert blurred.sharpness < clean.sharpness
    assert capture_metrics(noisy).noise > clean.noise


def test_quality_model_round_trip_and_decision(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "quality-model.json"
    path.write_text(json.dumps(quality_model_dict(model)), encoding="utf-8")

    loaded = load_quality_model(path)
    decision = evaluate_capture_quality(_pattern(4), category="part", model=loaded)

    assert decision.status in {"pass", "recapture"}
    assert 0.0 < decision.p_value <= 1.0
    assert set(decision.metrics) == set(decision.components)
    assert len(decision.contributors) == 2


def test_quality_model_rejects_unknown_category_and_bad_input() -> None:
    model = _model()

    with pytest.raises(QualityGateError, match="no category"):
        evaluate_capture_quality(_pattern(), category="unknown", model=model)
    with pytest.raises(QualityGateError, match="RGB uint8"):
        capture_metrics(np.zeros((64, 64), dtype=np.uint8))


def test_quality_model_loader_fails_closed_on_feature_schema_drift(tmp_path: Path) -> None:
    payload = quality_model_dict(_model())
    categories = payload["categories"]
    assert isinstance(categories, dict)
    reference = categories["part"]
    assert isinstance(reference, dict)
    scale = reference["scale"]
    assert isinstance(scale, dict)
    scale.pop("brightness")
    path = tmp_path / "invalid-quality-model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(QualityGateError, match="features are invalid"):
        load_quality_model(path)


def test_failed_quality_gate_skips_anomaly_inference() -> None:
    reference_metrics = capture_metrics(_pattern()).values()
    reference = CategoryQualityModel(
        center=reference_metrics,
        scale={name: 1e-5 for name in reference_metrics},
        calibration_scores=(0.0,) * 100,
        fit_samples=100,
        calibration_samples=100,
    )
    engine = InspectionEngine.__new__(InspectionEngine)
    engine.settings = load_prototype_settings(ROOT / "configs/prototype/m8.yaml")
    engine.image_size = 64
    engine._localizations = {}
    engine.quality_model = CaptureQualityModel(
        schema_version=1,
        image_size=64,
        alpha=0.02,
        manifest_sha256="a" * 64,
        categories={"part": reference},
    )
    engine._calibration = {"part": _CategoryCalibration(np.asarray([0.1, 0.2]), 0.2, 0.3)}
    stream = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(stream, format="PNG")

    result = engine.inspect_bytes(stream.getvalue(), filename="bad.png", category="part")

    assert result.decision == "recapture"
    assert result.model_evaluated is False
    assert result.heatmap_data_url is None
    assert result.timing_ms["feature_extraction"] == 0.0
