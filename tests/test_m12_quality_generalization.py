from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.experiments.m12 import (
    aggregate_comparison,
    normal_partitions,
    verify_files,
)
from robust_industrial_anomaly_detection.prototype.engine import (
    PrototypeError,
    _find_full_quality_run,
    load_prototype_settings,
)
from robust_industrial_anomaly_detection.prototype.quality import (
    CaptureQualityModel,
    QualityGateError,
    capture_measurements,
    capture_metrics,
    evaluate_capture_quality,
    fit_category_quality_model,
    load_quality_model,
    quality_model_dict,
)
from robust_industrial_anomaly_detection.prototype.reporting import build_inspection_report


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = cv2.GaussianBlur(rng.normal(128, 30, (64, 64)).astype(np.float32), (0, 0), 0.6)
    return np.repeat(np.clip(gray, 20, 230).astype(np.uint8)[:, :, None], 3, axis=2)


def _v2() -> CaptureQualityModel:
    metrics = [capture_measurements(_texture(seed), schema_version=2) for seed in range(80)]
    return CaptureQualityModel(
        2,
        64,
        0.02,
        "a" * 64,
        {"part": fit_category_quality_model(metrics[:30], metrics[30:], schema_version=2)},
    )


def test_v1_measurements_remain_identical() -> None:
    image = _texture(1)
    assert capture_measurements(image) == capture_metrics(image).values()
    v2 = capture_measurements(image, schema_version=2)
    assert {key: v2[key] for key in capture_metrics(image).values()} == capture_metrics(
        image
    ).values()


def test_v2_focus_and_noise_proxies_react_in_declared_directions() -> None:
    image = _texture(90)
    blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
    noisy = np.clip(
        image.astype(float) + np.random.default_rng(8).normal(0, 15, image.shape), 0, 255
    ).astype(np.uint8)
    clean = capture_measurements(image, schema_version=2)
    assert (
        capture_measurements(blurred, schema_version=2)["log_focus_ratio"]
        < clean["log_focus_ratio"]
    )
    assert capture_measurements(noisy, schema_version=2)["log_flat_noise"] > clean["log_flat_noise"]
    model = _v2()
    assert evaluate_capture_quality(blurred, category="part", model=model).status == "recapture"
    assert evaluate_capture_quality(noisy, category="part", model=model).status == "recapture"


def test_v2_focus_ratio_is_approximately_contrast_invariant() -> None:
    image = _texture(2)
    darker = np.clip((image.astype(float) - 128) * 0.6 + 100, 0, 255).astype(np.uint8)
    first = capture_measurements(image, schema_version=2)["log_focus_ratio"]
    second = capture_measurements(darker, schema_version=2)["log_focus_ratio"]
    assert abs(first - second) < 0.02


def test_v2_roundtrip_and_strict_versioning(tmp_path: Path) -> None:
    model = _v2()
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(quality_model_dict(model)), "utf-8")
    loaded = load_quality_model(path)
    assert loaded == model
    assert evaluate_capture_quality(
        _texture(90), category="part", model=loaded
    ) == evaluate_capture_quality(_texture(90), category="part", model=model)
    with pytest.raises(QualityGateError, match="unsupported schema"):
        quality_model_dict(replace(model, schema_version=3))
    payload = quality_model_dict(model)
    payload["schema_version"] = 1
    path.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(QualityGateError, match="directions"):
        load_quality_model(path)


def test_v2_blank_and_low_contrast_are_finite_and_fail_closed() -> None:
    for value in (0, 127, 255):
        decision = evaluate_capture_quality(
            np.full((64, 64, 3), value, np.uint8), category="part", model=_v2()
        )
        assert decision.status == "recapture"
        assert all(np.isfinite(value) for value in decision.metrics.values())
    with pytest.raises(QualityGateError, match="geometry"):
        evaluate_capture_quality(_texture(1)[:32], category="part", model=_v2())


def test_v2_fit_rejects_invalid_schema_and_nonfinite_measurements() -> None:
    metrics = [capture_measurements(_texture(seed), schema_version=2) for seed in range(16)]
    invalid = [{**item, "log_focus_ratio": float("nan")} for item in metrics]
    with pytest.raises(QualityGateError, match="finite"):
        fit_category_quality_model(invalid, metrics, schema_version=2)
    with pytest.raises(QualityGateError, match="schema"):
        fit_category_quality_model(metrics, metrics)


def test_normal_partitions_reject_defects_leakage_and_small_calibration() -> None:
    records = [
        {"sample_id": str(index), "split": split, "label": label}
        for index, (split, label) in enumerate(
            [("train", "normal")] * 20
            + [("validation", "normal")] * 50
            + [("test", "normal"), ("test", "anomaly")]
        )
    ]
    assert [len(group) for group in normal_partitions(records)] == [20, 50, 2]
    with pytest.raises(ValueError, match="only normals"):
        normal_partitions([{**records[0], "label": "anomaly"}, *records[1:]])
    with pytest.raises(ValueError, match="overlapping"):
        normal_partitions([*records, records[0]])
    with pytest.raises(ValueError, match="at least 49"):
        normal_partitions([row for row in records if row["sample_id"] not in {"20", "21"}])


def test_recapture_does_not_masquerade_as_reduced_operator_burden() -> None:
    rows = [
        {
            "gate": gate,
            "category": "part",
            "sample_id": str(index),
            "label": label,
            "condition": "clean",
            "family": "clean",
            "severity": 0,
            "quality_status": "pass" if gate == "v1" else "recapture",
            "source_decision": decision,
        }
        for gate in ("v1", "v2")
        for index, (label, decision) in enumerate(
            [("normal", "anomalous"), ("normal", "normal"), ("anomaly", "anomalous")]
        )
    ]
    result = aggregate_comparison(rows)
    normal = {row["gate"]: row for row in result if row["label"] == "normal"}
    assert normal["v1"]["guarded_auto_anomalous_rate"] == 0.5
    assert normal["v2"]["guarded_auto_anomalous_rate"] == 0
    assert normal["v1"]["guarded_actionable_rate"] == 0.5
    assert normal["v2"]["guarded_actionable_rate"] == 1.0
    anomaly = next(row for row in result if row["gate"] == "v2" and row["label"] == "anomaly")
    assert anomaly["guarded_auto_anomalous_rate"] == 0
    assert anomaly["quality_rejection_rate"] == 1


def test_committed_checkpoints_reject_corruption_and_escape(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    path = root / "model.json"
    path.write_text("original", "utf-8")
    hashes = {"model.json": sha256_file(path)}
    verify_files(root, hashes)
    path.write_text("changed", "utf-8")
    with pytest.raises(ValueError, match="modified"):
        verify_files(root, hashes)
    outside = tmp_path / "outside.json"
    outside.write_text("outside", "utf-8")
    with pytest.raises(ValueError):
        verify_files(root, {"../outside.json": sha256_file(outside)})


def test_quality_publication_refuses_smoke_and_wrong_source_pin(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m12_report", project / "scripts/build_m12_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"status": "complete", "mode": "smoke"}), "utf-8")
    config = {"source_run": str(tmp_path), "metadata_sha256": "a" * 64}
    with pytest.raises(ValueError, match="publication pin"):
        module.verified_source(config, project)
    config["metadata_sha256"] = sha256_file(metadata)
    with pytest.raises(ValueError, match="complete full"):
        module.verified_source(config, project)


def test_v2_inspector_is_explicit_and_export_identifies_gate_without_images() -> None:
    project = Path(__file__).resolve().parents[1]
    current = load_prototype_settings(project / "configs/prototype/m10.yaml")
    candidate = load_prototype_settings(project / "configs/prototype/m12.yaml")
    assert current.quality_experiment == "m8-quality-gate"
    assert candidate.quality_experiment == "m12-quality-generalization"
    assert candidate.port == 8767
    assert candidate.primary_model == current.primary_model
    assert candidate.localization_experiment == current.localization_experiment
    assert candidate.anomaly_alpha == current.anomaly_alpha
    report = build_inspection_report(
        [
            {
                "filename": "blank.png",
                "category": "candle",
                "decision": "recapture",
                "model_evaluated": False,
                "heatmap_data_url": "private",
                "preview_data_url": "private",
            }
        ],
        runtime_info={"quality_schema_version": 2, "source_quality_run": "m12-full"},
        max_results=1,
    )
    assert report["runtime"]["quality_schema_version"] == 2
    assert "private" not in json.dumps(report)


def test_live_gate_selection_refuses_incomplete_and_smoke_artifacts(tmp_path: Path) -> None:
    run = tmp_path / "runs/m12-quality-generalization-full-20260101T000000Z"
    run.mkdir(parents=True)
    metadata = run / "metadata.json"
    for status, mode in (("running", "full"), ("complete", "smoke")):
        metadata.write_text(json.dumps({"status": status, "mode": mode}), "utf-8")
        with pytest.raises(PrototypeError, match="no complete full"):
            _find_full_quality_run(tmp_path, "m12-quality-generalization", run)
    metadata.write_text(json.dumps({"status": "complete", "mode": "full"}), "utf-8")
    assert _find_full_quality_run(tmp_path, "m12-quality-generalization", run) == run
