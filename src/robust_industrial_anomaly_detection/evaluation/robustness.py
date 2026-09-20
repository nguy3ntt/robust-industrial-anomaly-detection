"""Paired uncertainty and stability metrics for controlled image shifts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]

from robust_industrial_anomaly_detection.evaluation.metrics import MetricError


def spearman_stability(
    first: Sequence[float] | np.ndarray, second: Sequence[float] | np.ndarray
) -> float:
    """Tie-aware Spearman correlation with deterministic constant handling."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 1 or left.shape != right.shape or left.size < 2:
        raise MetricError("Spearman inputs must be equal one-dimensional arrays")
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 1.0 if np.array_equal(left_rank, right_rank) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _normal_fpr(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    normal = labels == 0
    return float(np.mean(scores[normal] > threshold))


def paired_image_delta_intervals(
    labels: Sequence[int] | np.ndarray,
    clean_scores: Sequence[float] | np.ndarray,
    shifted_scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    seed: int,
    samples: int,
    confidence: float,
) -> dict[str, dict[str, float]]:
    """Stratified paired-bootstrap CIs for clean-to-shift image metric deltas."""

    target = np.asarray(labels, dtype=np.int8)
    clean = np.asarray(clean_scores, dtype=np.float64)
    shifted = np.asarray(shifted_scores, dtype=np.float64)
    if target.ndim != 1 or target.shape != clean.shape or clean.shape != shifted.shape:
        raise MetricError("paired score inputs must have identical one-dimensional shapes")
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise MetricError("invalid paired bootstrap settings")
    negative = np.flatnonzero(target == 0)
    positive = np.flatnonzero(target == 1)
    if not len(negative) or not len(positive):
        raise MetricError("paired bootstrap requires both classes")
    functions: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "image_auroc": lambda y, x: float(roc_auc_score(y, x)),
        "image_ap": lambda y, x: float(average_precision_score(y, x)),
        "test_normal_fpr": lambda y, x: _normal_fpr(y, x, threshold),
    }
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {name: [] for name in functions}
    for _ in range(samples):
        indexes = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        for name, function in functions.items():
            estimates[name].append(
                function(target[indexes], shifted[indexes])
                - function(target[indexes], clean[indexes])
            )
    tail = (1.0 - confidence) / 2.0
    return {
        name: {
            "low": float(np.quantile(values, tail)),
            "high": float(np.quantile(values, 1.0 - tail)),
        }
        for name, values in estimates.items()
    }


def heatmap_stability(
    clean_maps: Sequence[np.ndarray],
    shifted_maps: Sequence[np.ndarray],
    *,
    top_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-image heatmap correlation and top-support IoU."""

    if len(clean_maps) != len(shifted_maps) or not clean_maps:
        raise MetricError("heatmap collections must be non-empty and equally sized")
    if not 0.0 < top_fraction < 1.0:
        raise MetricError("top_fraction must be strictly between zero and one")
    correlations: list[float] = []
    overlaps: list[float] = []
    for clean_map, shifted_map in zip(clean_maps, shifted_maps, strict=True):
        clean = np.asarray(clean_map, dtype=np.float32)
        shifted = np.asarray(shifted_map, dtype=np.float32)
        if clean.shape != shifted.shape or clean.ndim != 2:
            raise MetricError("heatmap pairs must have identical two-dimensional shapes")
        left = clean.reshape(-1).astype(np.float64)
        right = shifted.reshape(-1).astype(np.float64)
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            correlation = 1.0 if np.array_equal(left, right) else 0.0
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
        count = max(1, int(np.ceil(left.size * top_fraction)))
        left_index = np.argpartition(left, -count)[-count:]
        right_index = np.argpartition(right, -count)[-count:]
        intersection = len(np.intersect1d(left_index, right_index, assume_unique=False))
        overlaps.append(float(intersection / (2 * count - intersection)))
        correlations.append(correlation)
    return np.asarray(correlations), np.asarray(overlaps)


def bootstrap_mean_interval(
    values: Sequence[float] | np.ndarray, *, seed: int, samples: int, confidence: float
) -> tuple[float, float]:
    """Bootstrap a mean over independent category or sample units."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise MetricError("bootstrap values must be finite and one-dimensional")
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise MetricError("invalid bootstrap settings")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [float(np.mean(rng.choice(array, size=array.size, replace=True))) for _ in range(samples)]
    )
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))
