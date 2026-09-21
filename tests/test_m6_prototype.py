from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionResult,
    PrototypeError,
    _decode_image,
    _safe_filename,
    load_prototype_settings,
)
from robust_industrial_anomaly_detection.prototype.reporting import (
    ReportError,
    build_inspection_report,
)
from robust_industrial_anomaly_detection.prototype.server import validate_loopback_host

ROOT = Path(__file__).parents[1]


def _result() -> InspectionResult:
    return InspectionResult(
        filename="part.png",
        category="candle",
        model="model",
        decision="review",
        anomaly_score=0.4,
        conformal_p=0.1,
        anomaly_alpha=0.05,
        review_alpha=0.2,
        empirical_score_threshold=0.5,
        pixel_threshold=0.3,
        calibration_samples=100,
        anomalous_area_fraction=0.1,
        input_width=640,
        input_height=480,
        model_input_size=224,
        preview_data_url="data:image/png;base64,private-preview",
        heatmap_data_url="data:image/png;base64,derived-private-image",
        warnings=("decision support only",),
        timing_ms={"total": 10.0},
    )


def test_m6_configuration_is_local_and_freezes_m5_policy() -> None:
    settings = load_prototype_settings(ROOT / "configs/prototype/m6.yaml")

    assert settings.host == "127.0.0.1"
    assert settings.embed_images_in_reports is False
    assert settings.primary_model == "multiscale_augmented_coreset_512"
    assert settings.anomaly_alpha == 0.05
    assert settings.review_alpha == 0.2


def test_public_result_can_remove_all_image_content() -> None:
    redacted = _result().public_dict(include_images=False)

    assert "preview_data_url" not in redacted
    assert "heatmap_data_url" not in redacted
    assert redacted["decision"] == "review"


def test_report_export_is_metadata_only_even_if_client_sends_images() -> None:
    source = _result().public_dict()
    report = build_inspection_report(
        [source],
        runtime_info={
            "model": "model",
            "device": "cpu",
            "anomaly_alpha": 0.05,
            "review_alpha": 0.2,
            "source_m5_run": "m5-run",
            "limitations": ["decision support only"],
        },
        max_results=2,
    )

    serialized = str(report)
    assert report["source_images_embedded"] is False
    assert "private-preview" not in serialized
    assert "derived-private-image" not in serialized
    assert report["result_count"] == 1


def test_report_rejects_empty_and_oversized_batches() -> None:
    with pytest.raises(ReportError):
        build_inspection_report([], runtime_info={}, max_results=1)
    with pytest.raises(ReportError):
        build_inspection_report(
            [_result().public_dict(), _result().public_dict()],
            runtime_info={},
            max_results=1,
        )


def test_decode_image_enforces_byte_and_pixel_limits() -> None:
    settings = load_prototype_settings(ROOT / "configs/prototype/m6.yaml")
    stream = io.BytesIO()
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8), mode="RGB").save(stream, format="PNG")

    decoded, width, height = _decode_image(stream.getvalue(), settings)

    assert decoded.shape == (12, 16, 3)
    assert (width, height) == (16, 12)
    with pytest.raises(PrototypeError):
        _decode_image(b"", settings)


def test_filename_is_reduced_to_a_bounded_basename() -> None:
    assert _safe_filename("../../private/folder/part.png") == "part.png"
    assert len(_safe_filename("x" * 500)) == 180


def test_static_interface_has_required_evidence_and_no_external_dependencies() -> None:
    static = ROOT / "src/robust_industrial_anomaly_detection/prototype/static"
    html = (static / "workstation.html").read_text(encoding="utf-8")

    assert "Anomaly score" in html
    assert "Conformal p" in html
    assert "Spatial evidence" in html
    assert "Export metadata report" in html
    assert 'id="theme-toggle"' in html
    assert "http://" not in html and "https://" not in html


def test_server_rejects_network_visible_bindings() -> None:
    validate_loopback_host("127.0.0.1")
    validate_loopback_host("localhost")
    with pytest.raises(PrototypeError, match="loopback"):
        validate_loopback_host("0.0.0.0")
