"""Distribution-free score calibration and selective-review metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.stats import beta  # type: ignore[import-untyped]

Decision = Literal["normal", "review", "anomalous"]


class CalibrationError(ValueError):
    """Raised when calibration inputs cannot support a valid estimate."""


def _scores(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise CalibrationError(f"{name} must be a non-empty finite one-dimensional array")
    return array


def empirical_quantile_threshold(
    calibration_scores: Sequence[float] | np.ndarray, quantile: float
) -> float:
    """Return a conservative empirical upper-tail threshold."""

    scores = _scores(calibration_scores, "calibration_scores")
    if not 0.0 < quantile < 1.0:
        raise CalibrationError("quantile must be strictly between zero and one")
    return float(np.quantile(scores, quantile, method="higher"))


def conformal_p_values(
    calibration_scores: Sequence[float] | np.ndarray,
    test_scores: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return conservative upper-tail split-conformal p-values.

    Larger nonconformity scores are more anomalous. Ties count against rejection,
    and the plus-one correction prevents zero p-values.
    """

    calibration = np.sort(_scores(calibration_scores, "calibration_scores"))
    test = _scores(test_scores, "test_scores")
    first_equal_or_greater = np.searchsorted(calibration, test, side="left")
    greater_or_equal = calibration.size - first_equal_or_greater
    return (greater_or_equal + 1.0) / (calibration.size + 1.0)


def selective_decisions(
    p_values: Sequence[float] | np.ndarray,
    *,
    anomaly_alpha: float,
    review_alpha: float,
) -> np.ndarray:
    """Map p-values to normal, review, or anomalous decisions."""

    values = _scores(p_values, "p_values")
    if not 0.0 < anomaly_alpha <= review_alpha < 1.0:
        raise CalibrationError("decision levels must satisfy 0 < anomaly_alpha <= review_alpha < 1")
    if np.any((values < 0.0) | (values > 1.0)):
        raise CalibrationError("p_values must lie in [0, 1]")
    decisions = np.full(values.shape, "review", dtype="<U9")
    decisions[values <= anomaly_alpha] = "anomalous"
    decisions[values > review_alpha] = "normal"
    return decisions


def exact_binomial_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a two-sided Clopper-Pearson interval."""

    if total < 1 or successes < 0 or successes > total:
        raise CalibrationError("binomial counts must satisfy 0 <= successes <= total and total > 0")
    if not 0.0 < confidence < 1.0:
        raise CalibrationError("confidence must be strictly between zero and one")
    tail = (1.0 - confidence) / 2.0
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    high = (
        1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    )
    return low, high


def selective_metrics(
    labels: Sequence[int] | np.ndarray,
    decisions: Sequence[str] | np.ndarray,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Summarize a three-way policy with exact binomial uncertainty."""

    target = np.asarray(labels, dtype=np.int8)
    action = np.asarray(decisions, dtype="<U9")
    if target.ndim != 1 or action.ndim != 1 or target.shape != action.shape or not target.size:
        raise CalibrationError("labels and decisions must be equal non-empty vectors")
    if not set(np.unique(target)).issubset({0, 1}):
        raise CalibrationError("labels must contain only zero and one")
    allowed = {"normal", "review", "anomalous"}
    if not set(np.unique(action)).issubset(allowed):
        raise CalibrationError("decisions contain an unsupported state")

    normal = target == 0
    anomaly = target == 1
    if not np.any(normal) or not np.any(anomaly):
        raise CalibrationError("selective metrics require both normal and anomalous samples")
    automatic = action != "review"
    errors = ((action == "anomalous") & normal) | ((action == "normal") & anomaly)

    counts = {
        "normal_false_positive": int(np.sum((action == "anomalous") & normal)),
        "normal_review": int(np.sum((action == "review") & normal)),
        "anomaly_detected": int(np.sum((action == "anomalous") & anomaly)),
        "anomaly_review": int(np.sum((action == "review") & anomaly)),
        "anomaly_missed": int(np.sum((action == "normal") & anomaly)),
        "review": int(np.sum(action == "review")),
        "automatic": int(np.sum(automatic)),
        "selective_error": int(np.sum(errors & automatic)),
    }
    denominators = {
        "normal_false_positive": int(np.sum(normal)),
        "normal_review": int(np.sum(normal)),
        "anomaly_detected": int(np.sum(anomaly)),
        "anomaly_review": int(np.sum(anomaly)),
        "anomaly_missed": int(np.sum(anomaly)),
        "review": int(target.size),
        "automatic": int(target.size),
        "selective_error": counts["automatic"],
    }
    names = {
        "normal_false_positive": "normal_fpr",
        "normal_review": "normal_review_rate",
        "anomaly_detected": "anomaly_confident_recall",
        "anomaly_review": "anomaly_review_rate",
        "anomaly_missed": "anomaly_miss_rate",
        "review": "review_rate",
        "automatic": "auto_coverage",
        "selective_error": "selective_risk",
    }
    result: dict[str, float | int] = {
        "samples": int(target.size),
        "normal_samples": int(np.sum(normal)),
        "anomaly_samples": int(np.sum(anomaly)),
        "automatic_samples": counts["automatic"],
    }
    for key, name in names.items():
        denominator = denominators[key]
        if denominator == 0:
            result[name] = float("nan")
            result[f"{name}_ci_low"] = float("nan")
            result[f"{name}_ci_high"] = float("nan")
            continue
        successes = counts[key]
        result[name] = successes / denominator
        low, high = exact_binomial_interval(successes, denominator, confidence=confidence)
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
    return result


def binary_decisions(scores: Sequence[float] | np.ndarray, threshold: float) -> np.ndarray:
    """Return normal/anomalous decisions for a fixed score threshold."""

    values = _scores(scores, "scores")
    if not np.isfinite(threshold):
        raise CalibrationError("threshold must be finite")
    return np.where(values > threshold, "anomalous", "normal")
