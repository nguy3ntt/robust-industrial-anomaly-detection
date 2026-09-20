from __future__ import annotations

import numpy as np
import pytest

from robust_industrial_anomaly_detection.evaluation.metrics import (
    MetricError,
    bootstrap_image_metrics,
    calibrate_threshold,
    image_metrics,
    pixel_metrics,
    pro_auc,
)


def test_calibration_uses_higher_empirical_quantile() -> None:
    assert calibrate_threshold([0.1, 0.2, 0.3, 0.4], 0.75) == pytest.approx(0.4)


def test_image_metrics_include_frozen_operating_point() -> None:
    metrics = image_metrics([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], threshold=0.5)
    assert metrics["image_auroc"] == pytest.approx(1.0)
    assert metrics["image_ap"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["test_normal_fpr"] == pytest.approx(0.0)
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2


def test_bootstrap_is_deterministic_and_stratified() -> None:
    first = bootstrap_image_metrics([0, 0, 0, 1, 1], [0.1, 0.4, 0.2, 0.8, 0.9], seed=7, samples=20)
    second = bootstrap_image_metrics([0, 0, 0, 1, 1], [0.1, 0.4, 0.2, 0.8, 0.9], seed=7, samples=20)
    assert first == second
    assert first["image_auroc"]["low"] == pytest.approx(1.0)


def test_pixel_and_region_metrics_reward_perfect_localization() -> None:
    anomaly = np.zeros((8, 8), dtype=bool)
    anomaly[2:6, 2:6] = True
    normal = np.zeros((8, 8), dtype=bool)
    anomaly_score = anomaly.astype(np.float32)
    normal_score = np.zeros((8, 8), dtype=np.float32)
    metrics = pixel_metrics([normal, anomaly], [normal_score, anomaly_score], threshold=0.5)
    assert metrics["pixel_auroc"] == pytest.approx(1.0)
    assert metrics["pixel_ap"] == pytest.approx(1.0)
    assert metrics["pixel_iou_mean"] == pytest.approx(1.0)
    assert pro_auc(
        [normal, anomaly], [normal_score, anomaly_score], thresholds=10
    ) == pytest.approx(1.0)


def test_metrics_reject_degenerate_labels() -> None:
    with pytest.raises(MetricError, match="both normal and anomalous"):
        image_metrics([0, 0], [0.1, 0.2], threshold=0.15)
