"""Leakage-safe metrics and uncertainty estimates for anomaly detection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


class MetricError(ValueError):
    """Raised when a metric input violates the evaluation contract."""


def _one_dimensional(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise MetricError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise MetricError(f"{name} contains non-finite values")
    return array


def _binary(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = _one_dimensional(values, name).astype(np.int8, copy=False)
    if not np.all((array == 0) | (array == 1)):
        raise MetricError(f"{name} must contain only zero and one")
    return array


def calibrate_threshold(normal_scores: Sequence[float] | np.ndarray, quantile: float) -> float:
    """Freeze a decision threshold using normal calibration data only."""

    scores = _one_dimensional(normal_scores, "normal_scores").astype(np.float64, copy=False)
    if not 0.0 < quantile < 1.0:
        raise MetricError("quantile must be strictly between zero and one")
    return float(np.quantile(scores, quantile, method="higher"))


def _fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    candidates = fpr[tpr >= target_tpr]
    return float(np.min(candidates)) if candidates.size else 1.0


def image_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Compute ranking and frozen-operating-point image metrics."""

    target = _binary(labels, "labels")
    prediction_scores = _one_dimensional(scores, "scores").astype(np.float64, copy=False)
    if target.shape != prediction_scores.shape:
        raise MetricError("labels and scores must have the same shape")
    if np.unique(target).size != 2:
        raise MetricError("image metrics require both normal and anomalous samples")
    predicted = prediction_scores > threshold
    positive = target == 1
    negative = ~positive
    tp = int(np.count_nonzero(predicted & positive))
    tn = int(np.count_nonzero(~predicted & negative))
    fp = int(np.count_nonzero(predicted & negative))
    fn = int(np.count_nonzero(~predicted & positive))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "image_auroc": float(roc_auc_score(target, prediction_scores)),
        "image_ap": float(average_precision_score(target, prediction_scores)),
        "image_fpr_at_95_tpr": _fpr_at_tpr(target, prediction_scores),
        "threshold": float(threshold),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "test_normal_fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "test_count": int(target.size),
    }


def bootstrap_image_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    seed: int,
    samples: int,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Return deterministic stratified bootstrap intervals for ranking metrics."""

    target = _binary(labels, "labels")
    prediction_scores = _one_dimensional(scores, "scores").astype(np.float64, copy=False)
    if target.shape != prediction_scores.shape:
        raise MetricError("labels and scores must have the same shape")
    if samples < 1:
        raise MetricError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise MetricError("confidence must be strictly between zero and one")
    negative = np.flatnonzero(target == 0)
    positive = np.flatnonzero(target == 1)
    if negative.size == 0 or positive.size == 0:
        raise MetricError("bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {
        "image_auroc": [],
        "image_ap": [],
        "image_fpr_at_95_tpr": [],
    }
    for _ in range(samples):
        indexes = np.concatenate(
            (
                rng.choice(negative, size=negative.size, replace=True),
                rng.choice(positive, size=positive.size, replace=True),
            )
        )
        sampled_labels = target[indexes]
        sampled_scores = prediction_scores[indexes]
        estimates["image_auroc"].append(float(roc_auc_score(sampled_labels, sampled_scores)))
        estimates["image_ap"].append(float(average_precision_score(sampled_labels, sampled_scores)))
        estimates["image_fpr_at_95_tpr"].append(_fpr_at_tpr(sampled_labels, sampled_scores))
    tail = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float]] = {}
    for name, values in estimates.items():
        result[name] = {
            "low": float(np.quantile(values, tail)),
            "high": float(np.quantile(values, 1.0 - tail)),
            "confidence": confidence,
        }
    return result


def pixel_metrics(
    masks: Sequence[np.ndarray],
    score_maps: Sequence[np.ndarray],
    threshold: float,
) -> dict[str, float | int]:
    """Compute exact pixel ranking metrics and mean per-image IoU."""

    if not masks or len(masks) != len(score_maps):
        raise MetricError("masks and score_maps must be non-empty and equally sized")
    flat_masks: list[np.ndarray] = []
    flat_scores: list[np.ndarray] = []
    ious: list[float] = []
    for index, (mask, score_map) in enumerate(zip(masks, score_maps, strict=True)):
        binary_mask = np.asarray(mask, dtype=bool)
        scores = np.asarray(score_map, dtype=np.float32)
        if binary_mask.ndim != 2 or binary_mask.shape != scores.shape:
            raise MetricError(f"pixel pair {index} has incompatible shapes")
        if not np.all(np.isfinite(scores)):
            raise MetricError(f"score map {index} contains non-finite values")
        predicted = scores > threshold
        union = np.count_nonzero(binary_mask | predicted)
        intersection = np.count_nonzero(binary_mask & predicted)
        ious.append(float(intersection / union) if union else 1.0)
        flat_masks.append(binary_mask.reshape(-1))
        flat_scores.append(scores.reshape(-1))
    target = np.concatenate(flat_masks).astype(np.int8, copy=False)
    scores = np.concatenate(flat_scores).astype(np.float64, copy=False)
    if np.unique(target).size != 2:
        raise MetricError("pixel metrics require positive and negative pixels")
    return {
        "pixel_auroc": float(roc_auc_score(target, scores)),
        "pixel_ap": float(average_precision_score(target, scores)),
        "pixel_iou_mean": float(np.mean(ious)),
        "pixel_threshold": float(threshold),
        "pixel_count": int(target.size),
    }


def pro_auc(
    masks: Sequence[np.ndarray],
    score_maps: Sequence[np.ndarray],
    *,
    max_fpr: float = 0.30,
    thresholds: int = 100,
) -> float:
    """Compute area under the per-region-overlap curve up to ``max_fpr``."""

    if not masks or len(masks) != len(score_maps):
        raise MetricError("masks and score_maps must be non-empty and equally sized")
    if not 0.0 < max_fpr <= 1.0 or thresholds < 2:
        raise MetricError("invalid PRO integration settings")
    minimum = min(float(np.min(score_map)) for score_map in score_maps)
    maximum = max(float(np.max(score_map)) for score_map in score_maps)
    if math.isclose(minimum, maximum):
        return 0.0
    threshold_values = np.linspace(maximum, minimum, num=thresholds)
    normal_pixels = sum(int(np.count_nonzero(~np.asarray(mask, dtype=bool))) for mask in masks)
    if normal_pixels == 0:
        raise MetricError("PRO requires background pixels")
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    structures = [ndimage.label(np.asarray(mask, dtype=bool))[0] for mask in masks]
    region_counts = [int(labels.max()) for labels in structures]
    for threshold in threshold_values:
        false_positives = 0
        overlaps: list[float] = []
        for mask, score_map, regions, region_count in zip(
            masks, score_maps, structures, region_counts, strict=True
        ):
            binary_mask = np.asarray(mask, dtype=bool)
            predicted = np.asarray(score_map) > threshold
            false_positives += int(np.count_nonzero(predicted & ~binary_mask))
            for region_id in range(1, region_count + 1):
                region = regions == region_id
                overlaps.append(
                    float(np.count_nonzero(predicted & region) / np.count_nonzero(region))
                )
        fpr = false_positives / normal_pixels
        if fpr <= max_fpr:
            points.append((float(fpr), float(np.mean(overlaps)) if overlaps else 0.0))
    if len(points) < 2:
        return 0.0
    points.sort(key=lambda pair: pair[0])
    fprs = np.asarray([pair[0] for pair in points], dtype=np.float64)
    pros = np.asarray([pair[1] for pair in points], dtype=np.float64)
    unique_fprs = np.unique(fprs)
    unique_pros = np.asarray([np.max(pros[fprs == value]) for value in unique_fprs])
    unique_pros = np.maximum.accumulate(unique_pros)
    if unique_fprs[-1] < max_fpr:
        unique_fprs = np.append(unique_fprs, max_fpr)
        unique_pros = np.append(unique_pros, unique_pros[-1])
    return float(np.trapezoid(unique_pros, unique_fprs) / max_fpr)


def json_ready(value: Any) -> Any:
    """Recursively convert NumPy values to standard JSON-compatible objects."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
