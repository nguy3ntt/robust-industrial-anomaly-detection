from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robust_industrial_anomaly_detection.experiments.m10 import mask_measurements
from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionEngine,
    PrototypeError,
    _score_overlay,
)
from robust_industrial_anomaly_detection.prototype.localization import (
    LocalizationError,
    LocalizationModel,
    fit_localization_model,
    load_localization_model,
    localization_support,
    save_localization_model,
    seeded_regions,
)
from robust_industrial_anomaly_detection.prototype.reporting import build_inspection_report


def test_region_growth_requires_strong_seed_and_removes_isolated_evidence() -> None:
    scores = np.zeros((20, 20), dtype=np.float32)
    scores[3:7, 3:7] = 2.0
    scores[4, 4] = 4.0
    scores[10:14, 10:14] = 2.0
    scores[18, 18] = 4.0

    support = seeded_regions(scores, high=3.0, low=1.0, minimum_pixels=4)

    assert support[3:7, 3:7].all()
    assert not support[10:14, 10:14].any()
    assert not support[18, 18]
    assert support.sum() == 16


def test_weak_bridge_cannot_grow_beyond_the_feature_cell_bound() -> None:
    scores = np.zeros((20, 20), np.float32)
    scores[10, 2:19] = 2.0
    scores[10, 3] = 4.0
    support = seeded_regions(scores, high=3.0, low=1.0, minimum_pixels=1, growth_radius=2)

    assert support[10, 3]
    assert not support[10, 6:].any()
    assert support.sum() == 4


def _fit() -> LocalizationModel:
    rng = np.random.default_rng(8)
    fit = [rng.normal(0.1, 0.01, (20, 20)).astype(np.float32) for _ in range(24)]
    calibration = [rng.normal(0.1, 0.01, (20, 20)).astype(np.float32) for _ in range(24)]
    return fit_localization_model(
        fit,
        calibration,
        high_quantile=0.995,
        low_quantile=0.99,
        minimum_region_fraction=0.0025,
        scale_floor_fraction=0.1,
        normal_pixel_budget=0.005,
        search_steps=24,
        provenance={"category": "part"},
    )


def test_normal_only_calibration_respects_support_budget_and_round_trips(tmp_path: Path) -> None:
    model = _fit()
    path = tmp_path / "localization.npz"
    save_localization_model(path, model)
    loaded = load_localization_model(path)

    assert loaded.calibration_spatial_support_rate <= loaded.normal_pixel_budget
    assert loaded.calibration_raw_support_rate <= loaded.normal_pixel_budget
    np.testing.assert_array_equal(loaded.center, model.center)
    np.testing.assert_array_equal(loaded.scale, model.scale)
    assert loaded.spatial_high >= loaded.spatial_low
    assert loaded.spatial_global_high <= loaded.spatial_high
    assert loaded.spatial_global_high >= loaded.spatial_low


def test_seed_only_threshold_is_calibrated_separately_from_growth() -> None:
    rng = np.random.default_rng(28)
    fit = [rng.uniform(0, 1, (20, 20)).astype(np.float32) for _ in range(24)]
    calibration = [rng.uniform(0, 1, (20, 20)).astype(np.float32) for _ in range(24)]
    model = fit_localization_model(
        fit,
        calibration,
        high_quantile=0.995,
        low_quantile=0.8,
        minimum_region_fraction=0.0025,
        scale_floor_fraction=0.1,
        normal_pixel_budget=0.005,
        search_steps=24,
        provenance={},
    )
    expected = float(
        np.quantile((np.stack(calibration) - model.center) / model.scale, 0.995, method="higher")
    )

    assert model.spatial_global_high == expected
    assert model.spatial_high >= model.spatial_global_high


def test_rendered_contours_use_the_exact_cleaned_support_mask() -> None:
    scores = np.full((20, 20), 5.0, np.float32)
    image = np.zeros((20, 20, 3), np.uint8)
    support = np.zeros((20, 20), bool)
    support[3:7, 3:7] = True

    _, fraction = _score_overlay(image, scores, 1.0, support=support)

    assert fraction == 16 / 400


def test_spatial_support_rejects_mismatched_geometry() -> None:
    model = _fit()
    with pytest.raises(LocalizationError, match="geometry"):
        localization_support(
            np.zeros((21, 20)), model, "spatial_hysteresis", baseline_threshold=0.5
        )


def test_spatial_reference_recovers_signal_in_normally_quiet_area() -> None:
    rng = np.random.default_rng(17)
    maps = []
    for _ in range(48):
        image = rng.normal(0.1, 0.01, (20, 20)).astype(np.float32)
        image[:, :5] += 1.0
        maps.append(image)
    model = fit_localization_model(
        maps[:24],
        maps[24:],
        high_quantile=0.995,
        low_quantile=0.99,
        minimum_region_fraction=0.0025,
        scale_floor_fraction=0.1,
        normal_pixel_budget=0.005,
        search_steps=24,
        provenance={},
    )
    test = np.full((20, 20), 0.1, np.float32)
    test[:, :5] += 1.0
    test[10:14, 10:14] += 0.2

    _, old, _ = localization_support(test, model, "baseline_global", baseline_threshold=1.15)
    _, new, _ = localization_support(test, model, "spatial_hysteresis", baseline_threshold=1.15)

    assert not old[10:14, 10:14].any()
    assert new[10:14, 10:14].all()
    assert not new[:, :5].any()


def test_defect_iou_cannot_be_inflated_by_empty_normal_masks() -> None:
    empty = np.zeros((10, 10), dtype=bool)
    normal = mask_measurements(empty, empty, "normal")
    unseen = mask_measurements(empty, empty, "anomaly")
    positive = empty.copy()
    positive[3:6, 3:6] = True
    missed = mask_measurements(positive, empty, "anomaly")

    assert normal["all_image_iou"] == 1.0
    assert normal["foreground_iou"] is None
    assert unseen["foreground_iou"] is None
    assert missed["foreground_iou"] == 0.0
    assert missed["foreground_precision"] == 0.0


def test_region_growth_fails_closed_on_nonfinite_maps() -> None:
    with pytest.raises(LocalizationError, match="finite"):
        seeded_regions(np.full((10, 10), np.nan), high=2.0, low=1.0, minimum_pixels=1)


def test_raw_cleanup_preserves_threshold_and_only_removes_small_components() -> None:
    model = replace(_fit(), minimum_region_pixels=4)
    scores = np.zeros((20, 20), np.float32)
    scores[3:7, 3:7] = 2.0
    scores[18, 18] = 2.0
    evidence, support, threshold = localization_support(
        scores, model, "raw_cleaned", baseline_threshold=1.0
    )

    np.testing.assert_array_equal(evidence, scores)
    assert threshold == 1.0
    assert support[3:7, 3:7].all()
    assert not support[18, 18]
    assert support.sum() == 16


def test_artifact_loader_rejects_nonpositive_spatial_scale(tmp_path: Path) -> None:
    model = replace(_fit(), scale=np.zeros((20, 20), np.float32))
    path = tmp_path / "invalid.npz"
    save_localization_model(path, model)
    with pytest.raises(LocalizationError, match="scale"):
        load_localization_model(path)


def test_live_contract_refuses_a_different_source_model(tmp_path: Path) -> None:
    run = tmp_path / "localization"
    run.mkdir()
    (run / "metadata.json").write_text('{"visa_m5_metadata_sha256":"wrong"}', "utf-8")
    source = tmp_path / "m5"
    source.mkdir()
    (source / "metadata.json").write_text("{}", "utf-8")
    engine = InspectionEngine.__new__(InspectionEngine)
    engine.localization_run = run
    engine.m5_run = source
    with pytest.raises(PrototypeError, match="selected M5 source"):
        engine._load_localization_contract()


def test_localization_export_contains_evidence_metadata_but_no_images() -> None:
    report = build_inspection_report(
        [
            {
                "filename": "part.png",
                "category": "candle",
                "decision": "review",
                "localization_policy": "raw_cleaned",
                "localized_regions": 2,
                "localization_threshold_units": "memory-distance",
                "preview_data_url": "private",
                "heatmap_data_url": "private",
            }
        ],
        runtime_info={"localization_policy": "raw_cleaned"},
        max_results=24,
    )
    result = report["results"][0]
    assert result["localization_policy"] == "raw_cleaned"
    assert result["localized_regions"] == 2
    assert result["localization_threshold_units"] == "memory-distance"
    assert "preview_data_url" not in result
    assert "heatmap_data_url" not in result
