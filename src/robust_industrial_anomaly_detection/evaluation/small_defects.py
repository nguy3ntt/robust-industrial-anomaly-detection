"""Common-grid foreground and image-clustered small-region localization metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from robust_industrial_anomaly_detection.experiments.m10 import mask_measurements


def region_measurements(
    mask: np.ndarray,
    support: np.ndarray,
    *,
    tiny_fraction: float = 0.0001,
    small_fraction: float = 0.001,
    hit_threshold: float = 0.10,
) -> list[dict[str, Any]]:
    """Measure every common-GT component; size cohorts never depend on predictions."""

    target = np.asarray(mask, dtype=bool)
    predicted = np.asarray(support, dtype=bool)
    if (
        target.ndim != 2
        or target.shape != predicted.shape
        or not (0 < tiny_fraction < small_fraction < 1 and 0 < hit_threshold <= 1)
    ):
        raise ValueError("invalid small-defect masks or region settings")
    count, regions, stats, _ = cv2.connectedComponentsWithStats(
        target.astype(np.uint8), connectivity=8
    )
    intersections = np.bincount(regions[predicted].reshape(-1), minlength=count)
    result = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        fraction = area / target.size
        recall = int(intersections[index]) / area
        result.append(
            {
                "region_id": index,
                "region_pixels": area,
                "region_fraction": fraction,
                "stratum": "tiny"
                if fraction <= tiny_fraction
                else "small"
                if fraction <= small_fraction
                else "large",
                "region_recall": recall,
                "region_hit": int(recall >= hit_threshold),
            }
        )
    return result


def capture_measurements(
    mask: np.ndarray,
    support: np.ndarray,
    label: str,
    *,
    small_fraction: float,
    regions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values: dict[str, Any] = dict(mask_measurements(mask, support, label))
    selected = [row for row in regions if row["region_fraction"] <= small_fraction]
    visible = label == "anomaly" and int(values["mask_pixels"]) > 0
    values.update(
        {
            "small_image": int(
                visible and int(values["mask_pixels"]) / mask.size <= small_fraction
            ),
            "small_regions": len(selected),
            "small_hits": sum(row["region_hit"] for row in selected),
            "small_recall_sum": sum(row["region_recall"] for row in selected),
        }
    )
    return values


def aggregate_captures(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("localization aggregation requires non-empty captures")
    normals = [row for row in rows if row["label"] == "normal"]
    visible = [row for row in rows if row.get("foreground_iou") is not None]
    small = [row for row in visible if row["small_image"]]

    def average(group: Sequence[Mapping[str, Any]], key: str) -> float | None:
        return float(np.mean([row[key] for row in group])) if group else None

    region_count = sum(row["small_regions"] for row in rows)
    return {
        "samples": len(rows),
        "normal_samples": len(normals),
        "anomaly_samples": sum(row["label"] == "anomaly" for row in rows),
        "visible_anomaly_samples": len(visible),
        "empty_anomaly_masks": sum(
            row["label"] == "anomaly" and row["mask_pixels"] == 0 for row in rows
        ),
        **{
            key: average(visible, key)
            for key in (
                "foreground_iou",
                "foreground_dice",
                "foreground_recall",
                "foreground_precision",
            )
        },
        "small_images": len(small),
        "small_image_iou": average(small, "foreground_iou"),
        "small_image_recall": average(small, "foreground_recall"),
        "small_regions": region_count,
        "small_region_hit_rate": sum(row["small_hits"] for row in rows) / region_count
        if region_count
        else None,
        "small_region_pixel_recall": sum(row["small_recall_sum"] for row in rows) / region_count
        if region_count
        else None,
        "normal_pixel_fpr": sum(row["flagged_pixels"] for row in normals)
        / sum(row["pixel_count"] for row in normals)
        if normals
        else None,
        "normal_image_support_rate": average(normals, "has_support"),
        "background_pixel_fpr": sum(row["false_positive_pixels"] for row in rows)
        / sum(row["background_pixels"] for row in rows),
        "coarse_lost_positive_masks": sum(row.get("coarse_mask_lost", 0) for row in rows),
        "source_positive_fine_empty": sum(
            row.get("source_positive_pixels", 0) > 0 and row["mask_pixels"] == 0 for row in rows
        ),
    }


def aggregate_regions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for stratum in ("tiny", "small", "large", "small_inclusive", "all"):
        selected = [
            row
            for row in rows
            if stratum == "all"
            or row["stratum"] == stratum
            or (stratum == "small_inclusive" and row["stratum"] in {"tiny", "small"})
        ]
        result.append(
            {
                "stratum": stratum,
                "regions": len(selected),
                "images": len({row["sample_id"] for row in selected}),
                "region_hit_rate": float(np.mean([row["region_hit"] for row in selected]))
                if selected
                else None,
                "region_pixel_recall": float(np.mean([row["region_recall"] for row in selected]))
                if selected
                else None,
            }
        )
    return result


def paired_deltas(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    samples: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Resample whole images, including all their small regions, with shared paired indexes."""

    if not baseline or len(baseline) != len(candidate) or samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid paired localization comparison")
    if [
        (row["sample_id"], row["label"], row["mask_pixels"], row["small_regions"])
        for row in baseline
    ] != [
        (row["sample_id"], row["label"], row["mask_pixels"], row["small_regions"])
        for row in candidate
    ]:
        raise ValueError("paired localization identities or common ground truth differ")
    labels = np.asarray([row["label"] == "anomaly" for row in baseline])
    negative, positive = np.flatnonzero(~labels), np.flatnonzero(labels)
    if not len(negative) or not len(positive):
        raise ValueError("paired intervals require both official test labels")
    rng = np.random.default_rng(seed)
    indexes = np.concatenate(
        (
            rng.choice(negative, (samples, len(negative))),
            rng.choice(positive, (samples, len(positive))),
        ),
        axis=1,
    )
    output: dict[str, Any] = {"paired_images": len(baseline), "bootstrap_unit": "image"}
    for key in ("foreground_iou", "small_image_iou", "small_region_hit_rate", "normal_fpr"):
        estimates = []
        points = []
        for rows in (baseline, candidate):
            if key in {"foreground_iou", "small_image_iou"}:
                valid = np.asarray(
                    [
                        row.get("foreground_iou") is not None
                        and (key != "small_image_iou" or bool(row["small_image"]))
                        for row in rows
                    ]
                )
                numerator = (
                    np.asarray([float(row.get("foreground_iou") or 0) for row in rows]) * valid
                )
                denominator = valid.astype(float)
            elif key == "normal_fpr":
                denominator = (~labels).astype(float)
                numerator = (
                    np.asarray([row["decision"] == "anomalous" for row in rows]) * denominator
                )
            else:
                numerator = np.asarray([row["small_hits"] for row in rows], dtype=float)
                denominator = np.asarray([row["small_regions"] for row in rows], dtype=float)
            totals = np.sum(denominator[indexes], axis=1)
            values = np.full(samples, np.nan)
            np.divide(np.sum(numerator[indexes], axis=1), totals, out=values, where=totals > 0)
            estimates.append(values)
            points.append(
                float(np.sum(numerator) / np.sum(denominator)) if np.sum(denominator) else None
            )
        differences = estimates[1] - estimates[0]
        finite = differences[np.isfinite(differences)]
        tail = (1 - confidence) / 2
        output[key + "_delta"] = (
            points[1] - points[0] if points[1] is not None and points[0] is not None else None
        )
        output[key + "_delta_ci_low"] = float(np.quantile(finite, tail)) if len(finite) else None
        output[key + "_delta_ci_high"] = (
            float(np.quantile(finite, 1 - tail)) if len(finite) else None
        )
        output[key + "_finite_resamples"] = len(finite)
    return output
