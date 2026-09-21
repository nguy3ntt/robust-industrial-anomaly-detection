from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robust_industrial_anomaly_detection.evaluation.calibration import (
    binary_decisions,
    conformal_p_values,
    empirical_quantile_threshold,
    exact_binomial_interval,
    selective_decisions,
    selective_metrics,
)


def test_conformal_p_values_are_plus_one_corrected_and_tie_conservative() -> None:
    calibration = [1.0, 2.0, 2.0, 4.0]

    values = conformal_p_values(calibration, [0.0, 2.0, 3.0, 5.0])

    assert values.tolist() == pytest.approx([1.0, 0.8, 0.4, 0.2])


def test_three_way_boundaries_are_explicit() -> None:
    decisions = selective_decisions(
        [0.01, 0.05, 0.051, 0.2, 0.201], anomaly_alpha=0.05, review_alpha=0.2
    )

    assert decisions.tolist() == ["anomalous", "anomalous", "review", "review", "normal"]


def test_selective_metrics_measure_coverage_and_only_automatic_errors() -> None:
    labels = [0, 0, 1, 1, 1]
    decisions = ["normal", "review", "anomalous", "review", "normal"]

    metrics = selective_metrics(labels, decisions)

    assert metrics["review_rate"] == pytest.approx(0.4)
    assert metrics["auto_coverage"] == pytest.approx(0.6)
    assert metrics["selective_risk"] == pytest.approx(1 / 3)
    assert metrics["normal_fpr"] == 0.0
    assert metrics["anomaly_miss_rate"] == pytest.approx(1 / 3)


def test_selective_risk_is_undefined_when_every_sample_is_reviewed() -> None:
    metrics = selective_metrics([0, 1], ["review", "review"])

    assert metrics["auto_coverage"] == 0.0
    assert np.isnan(metrics["selective_risk"])


def test_quantile_is_conservative_and_binary_threshold_is_strict() -> None:
    threshold = empirical_quantile_threshold([1.0, 2.0, 3.0, 4.0], 0.6)

    assert threshold == 3.0
    assert binary_decisions([3.0, 3.1], threshold).tolist() == ["normal", "anomalous"]


def test_exact_interval_handles_boundaries() -> None:
    assert exact_binomial_interval(0, 10)[0] == 0.0
    assert exact_binomial_interval(10, 10)[1] == 1.0
    low, high = exact_binomial_interval(5, 10)
    assert low < 0.5 < high


def test_m5_configuration_freezes_policy_before_evaluation() -> None:
    import yaml

    root = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/experiments/m5-calibration.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert root["calibration"] == {
        "quantile": 0.95,
        "anomaly_alpha": 0.05,
        "review_alpha": 0.2,
        "scopes": ["category", "pooled"],
    }
    assert root["coverage_risk"]["review_alphas"] == [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
