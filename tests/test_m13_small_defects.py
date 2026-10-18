from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from robust_industrial_anomaly_detection.baselines.encoders import ENCODERS
from robust_industrial_anomaly_detection.baselines.multiscale import PyramidFeatures
from robust_industrial_anomaly_detection.baselines.tiling import (
    Tile,
    aligned_canvas,
    blend_maps,
    evaluation_support,
    tile_layout,
    view_features,
)
from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.evaluation.small_defects import (
    aggregate_captures,
    aggregate_regions,
    capture_measurements,
    paired_deltas,
    region_measurements,
)
from robust_industrial_anomaly_detection.experiments.m2 import Sample
from robust_industrial_anomaly_detection.experiments.m11 import partition
from robust_industrial_anomaly_detection.experiments.m13 import load_config, scaled_shift
from robust_industrial_anomaly_detection.robustness.transforms import ShiftSpec

ROOT = Path(__file__).resolve().parents[1]


def _report_module():
    spec = importlib.util.spec_from_file_location(
        "m13_report", ROOT / "scripts/build_m13_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_tile_geometry_and_odd_frame_edges_have_complete_coverage() -> None:
    layout = tile_layout(336, 336, 224, 112)
    assert [(tile.top, tile.left) for tile in layout] == [(0, 0), (0, 112), (112, 0), (112, 112)]
    for height, width in ((336, 336), (337, 349), (224, 224)):
        layout = tile_layout(height, width, 224, 112)
        result = blend_maps(
            [np.full((28, 28), 3.5, np.float32) for _ in layout], layout, (height, width)
        )
        assert result.shape == (height, width)
        np.testing.assert_allclose(result, 3.5)
    with pytest.raises(ValueError, match="stride"):
        tile_layout(336, 336, 224, 225)


def test_overlap_blending_preserves_linear_coordinates_and_has_no_edge_nan() -> None:
    height, width = 64, 73
    reference = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    layout = tile_layout(height, width, 40, 20)
    values = [reference[t.top : t.top + t.size, t.left : t.left + t.size] for t in layout]
    np.testing.assert_allclose(blend_maps(values, layout, (height, width)), reference, atol=1e-5)
    with pytest.raises(ValueError, match="uncovered"):
        blend_maps([values[0]], [layout[0]], (height, width))
    with pytest.raises(ValueError, match="outside"):
        blend_maps([values[0]], [Tile(50, 50, 40)], (height, width))
    with pytest.raises(ValueError, match="finite"):
        blend_maps([np.full((2, 2), np.nan)], [Tile(0, 0, 64)], (64, 64))


def test_positive_coverage_mask_retains_a_one_pixel_defect_lost_by_nearest() -> None:
    image = Image.fromarray(np.zeros((1024, 1024, 3), np.uint8))
    source = np.zeros((1024, 1024), np.uint8)
    source[7, 7] = 255
    mask = Image.fromarray(source)
    _, target, positives = aligned_canvas(image, mask, dataset="visa", size=336)
    assert positives == 1 and target.any()
    assert not np.asarray(mask.resize((56, 56), Image.Resampling.NEAREST)).any()
    np.testing.assert_array_equal(aligned_canvas(image, None, dataset="visa", size=336)[1], False)


def test_canvas_distinguishes_crop_loss_from_raster_loss_and_full_frame() -> None:
    image = Image.fromarray(np.zeros((64, 128, 3), np.uint8))
    source = np.zeros((64, 128), np.uint8)
    source[10:12, 1:3] = 255
    mask = Image.fromarray(source)
    assert not aligned_canvas(image, mask, dataset="visa", size=336)[1].any()
    assert aligned_canvas(image, mask, dataset="ksdd2", size=336)[1].any()
    with pytest.raises(ValueError, match="dimensions"):
        aligned_canvas(
            image, Image.fromarray(np.zeros((10, 10), np.uint8)), dataset="visa", size=336
        )


class _Encoder:
    def __init__(self) -> None:
        self.count = 0

    def extract_rgb(self, images: list[np.ndarray]) -> tuple[PyramidFeatures, ...]:
        self.count = len(images)
        return tuple(
            PyramidFeatures(
                np.ones((2, 2, 64), np.float32),
                np.ones((2, 2, 128), np.float32),
                np.ones((1, 1, 256), np.float32),
            )
            for _ in images
        )


def test_view_extraction_groups_four_views_per_capture_without_extra_fit_budget() -> None:
    encoder = _Encoder()
    images = [np.zeros((336, 336, 3), np.uint8)] * 2
    values = view_features(encoder, images, mode="tiled", input_size=224, tile_size=224, stride=112)
    assert encoder.count == 8 and list(map(len, values)) == [4, 4]
    values = view_features(
        encoder, images, mode="global", input_size=224, tile_size=224, stride=112
    )
    assert encoder.count == 2 and list(map(len, values)) == [1, 1]


def test_small_region_sizes_and_hit_boundary_are_defined_on_common_gt() -> None:
    mask = np.zeros((100, 100), bool)
    mask[1, 1] = True
    mask[10:12, 10:15] = True
    mask[30:34, 30:34] = True
    support = np.zeros_like(mask)
    support[1, 1] = True
    support[10, 10] = True
    rows = region_measurements(mask, support)
    assert [row["stratum"] for row in rows] == ["tiny", "small", "large"]
    assert [row["region_hit"] for row in rows] == [1, 1, 0]
    assert [row["region_recall"] for row in rows] == [1, 0.1, 0]
    summary = aggregate_regions([{**row, "sample_id": "one"} for row in rows])
    small = next(row for row in summary if row["stratum"] == "small_inclusive")
    assert small["regions"] == 2 and small["images"] == 1 and small["region_hit_rate"] == 1


def test_coarse_empty_gt_is_not_excluded_from_common_grid_foreground_metrics() -> None:
    target = np.zeros((336, 336), bool)
    target[5, 5] = True
    assert not cv2.resize(target.astype(np.uint8), (56, 56), interpolation=cv2.INTER_NEAREST).any()
    support = evaluation_support(np.zeros((336, 336), np.float32), 0.5, grid=56)
    row = {
        "label": "anomaly",
        **capture_measurements(
            target,
            support,
            "anomaly",
            small_fraction=0.001,
            regions=region_measurements(target, support),
        ),
    }
    assert row["foreground_iou"] == 0 and row["small_regions"] == 1
    summary = aggregate_captures([row])
    assert summary["visible_anomaly_samples"] == 1 and summary["foreground_iou"] == 0


def test_empty_normals_do_not_inflate_foreground_overlap() -> None:
    target = np.zeros((100, 100), bool)
    normal = {
        "label": "normal",
        **capture_measurements(target, target, "normal", small_fraction=0.001, regions=[]),
    }
    assert aggregate_captures([normal])["foreground_iou"] is None
    assert aggregate_captures([normal])["normal_pixel_fpr"] == 0


def _paired_rows() -> list[dict]:
    rows = []
    for index in range(6):
        normal = index < 3
        target = np.zeros((100, 100), bool)
        if not normal:
            target[5, 5] = True
        support = np.zeros_like(target)
        rows.append(
            {
                "sample_id": str(index),
                "label": "normal" if normal else "anomaly",
                "decision": "normal",
                **capture_measurements(
                    target,
                    support,
                    "normal" if normal else "anomaly",
                    small_fraction=0.001,
                    regions=region_measurements(target, support),
                ),
            }
        )
    return rows


def test_paired_bootstrap_clusters_regions_with_images_and_checks_common_gt() -> None:
    baseline = _paired_rows()
    assert paired_deltas(baseline, baseline, seed=3, samples=20)["foreground_iou_delta"] == 0
    improved = [
        {
            **row,
            "foreground_iou": 1 if row["label"] == "anomaly" else None,
            "small_hits": row["small_regions"],
        }
        for row in baseline
    ]
    result = paired_deltas(baseline, improved, seed=3, samples=20)
    assert result["foreground_iou_delta"] == 1 and result["small_region_hit_rate_delta"] == 1
    assert result["normal_fpr_delta"] == 0 and result["bootstrap_unit"] == "image"
    with pytest.raises(ValueError, match="identities"):
        paired_deltas(baseline, list(reversed(improved)), seed=3, samples=20)


def test_protocol_preserves_all_tests_and_scales_spatial_not_intensity_shifts() -> None:
    config = load_config(ROOT / "configs/experiments/m13-small-defects.yaml")
    samples = [
        Sample(
            str(index),
            "part",
            split,
            "normal" if split != "test" else "anomaly" if index % 2 else "normal",
            Path("image.png"),
            None,
        )
        for index, split in enumerate(["train"] * 200 + ["validation"] * 60 + ["test"] * 40)
    ]
    train, validation, test = partition(samples, config, False)
    assert (len(train), len(validation), len(test)) == (128, 60, 40)
    assert partition(list(reversed(samples)), config, False) == (train, validation, test)
    samples[0] = replace(samples[0], label="anomaly")
    with pytest.raises(ValueError, match="only normals"):
        partition(samples, config, False)
    blur = scaled_shift(ShiftSpec("blur", "blur", 2, {"sigma": 2.0}))
    noise = scaled_shift(ShiftSpec("noise", "noise", 2, {"standard_deviation": 0.05}))
    assert blur.parameters["sigma"] == 3 and noise.parameters["standard_deviation"] == 0.05


def test_publication_rejects_wrong_pin_smoke_and_incomplete_full_runs(tmp_path: Path) -> None:
    module = _report_module()
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"status": "complete", "mode": "smoke"}), "utf-8")
    config = {"source_run": str(tmp_path), "metadata_sha256": "a" * 64}
    with pytest.raises(ValueError, match="publication pin"):
        module.verified_source(config, ROOT)
    config["metadata_sha256"] = sha256_file(metadata)
    with pytest.raises(ValueError, match="complete full"):
        module.verified_source(config, ROOT)
    resolved = {
        **load_config(ROOT / "configs/experiments/m13-small-defects.yaml"),
        "mode": "full",
        "selected_datasets": ["ksdd2", "visa"],
        "selected_categories": [],
    }
    (tmp_path / "resolved_config.json").write_text(json.dumps(resolved), "utf-8")
    summary = {"units": 12, "metric_rows": 117, "comparison_rows": 78, "test_images": 3166}
    (tmp_path / "summary.json").write_text(json.dumps(summary), "utf-8")
    payload = {
        "status": "complete",
        "mode": "full",
        "output_hashes": {},
        "weights_sha256": ENCODERS["resnet18"].sha256,
    }
    metadata.write_text(json.dumps(payload), "utf-8")
    config["metadata_sha256"] = sha256_file(metadata)
    with pytest.raises(ValueError, match="incomplete unit"):
        module.verified_source(config, ROOT)


def test_comparison_panel_reproduces_support_and_rejects_a_changed_measurement(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, "build_m13_report", _report_module())
    spec = importlib.util.spec_from_file_location(
        "m13_panels", ROOT / "scripts/prepare_m13_manual_examples.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mask = np.zeros((336, 336), bool)
    mask[168, 168] = True
    arrays = {
        "rgb": np.full((336, 336, 3), 128, np.uint8),
        "mask": mask,
        "global_map": np.zeros((336, 336), np.float32),
        "tiled_map": np.zeros((336, 336), np.float32),
    }
    sample = {"sample_id": "synthetic", "label": "anomaly"}
    rows = [
        {
            "sample_id": "synthetic",
            "readout": name,
            "mask_pixels": 1,
            "flagged_pixels": 0,
            "foreground_iou": 0.0,
            "decision": "normal",
            "conformal_p": 0.5,
        }
        for name in ("global56", "global336", "tiled336")
    ]
    calibrations = {
        name: {"pixel_thresholds": {"56": 1.0, "336": 1.0}} for name in ("global", "tiled")
    }
    path = tmp_path / "panel.png"
    result = module.make_panel(arrays, sample, rows, calibrations, path, "Synthetic test")
    assert len(result["results"]) == 3
    with Image.open(path) as image:
        assert image.size == (1792, 920) and image.mode == "RGB"
    rows[0]["flagged_pixels"] = 1
    with pytest.raises(ValueError, match="does not reproduce"):
        module.make_panel(
            arrays, sample, rows, calibrations, tmp_path / "bad.png", "Synthetic test"
        )
    assert not (tmp_path / "bad.png").exists()


def test_aggregate_report_keeps_low_count_points_but_omits_false_precision(
    tmp_path: Path, monkeypatch
) -> None:
    module = _report_module()
    metrics, regions, comparisons, resources = [], [], [], []
    conditions = ("clean", "blur_severe", "noise_severe")
    for dataset in ("visa", "ksdd2"):
        for condition in conditions:
            for readout in ("global56", "global336", "tiled336"):
                metrics.append(
                    {
                        "dataset": dataset,
                        "category": "part",
                        "condition": condition,
                        "readout": readout,
                        "samples": 2,
                        "anomaly_samples": 1,
                        "visible_anomaly_samples": 1,
                        "empty_anomaly_masks": 0,
                        "coarse_lost_positive_masks": 1,
                        "source_positive_fine_empty": 0,
                        "small_images": 1,
                        "small_regions": 1,
                        "small_image_iou": 0.1,
                        "small_region_hit_rate": 1,
                        "foreground_iou": 0.1,
                        "normal_fpr": 0,
                        "normal_pixel_fpr": 0,
                        "normal_image_support_rate": 0,
                    }
                )
                for stratum in ("tiny", "small", "large", "small_inclusive", "all"):
                    regions.append(
                        {
                            "dataset": dataset,
                            "category": "part",
                            "condition": condition,
                            "readout": readout,
                            "stratum": stratum,
                            "regions": 1,
                            "region_hit_rate": 1,
                            "region_pixel_recall": 0.5,
                        }
                    )
            for contrast in ("resolution", "tiling"):
                row = {
                    "dataset": dataset,
                    "category": "part",
                    "condition": condition,
                    "contrast": contrast,
                }
                for metric in module.DELTAS:
                    row[metric + "_delta"] = row[metric + "_delta_ci_low"] = row[
                        metric + "_delta_ci_high"
                    ] = 0.1
                comparisons.append(row)
        for mode in ("global", "tiled"):
            resources.append(
                {
                    "dataset": dataset,
                    "category": "part",
                    "mode": mode,
                    "index_bytes": 100,
                    "post_fit_working_set_bytes": 1000,
                }
            )
    predictions = [
        {"readout": name, "small_image": int(index == 1), "small_regions": int(index == 1)}
        for name in ("global56", "global336", "tiled336")
        for index in range(2)
    ]
    tables = {
        "metrics.csv": metrics,
        "region_metrics.csv": regions,
        "comparisons.csv": comparisons,
        "resources.csv": resources,
        "predictions.csv": predictions,
        "benchmark_timings.csv": [{"mode": mode, "total_ms": 10.0} for mode in ("global", "tiled")],
    }
    monkeypatch.setattr(module, "verified_source", lambda config, project: tmp_path)
    monkeypatch.setattr(module, "_read_csv", lambda path: json.loads(json.dumps(tables[path.name])))
    config = {
        "source_run": str(tmp_path),
        "metadata_sha256": "a" * 64,
        "output_directory": "reports/m13",
    }
    first = module.build(config, tmp_path)
    assert module.build(config, tmp_path) == first
    with (tmp_path / "reports/m13/macro_deltas.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(float(row["small_image_iou_delta"]) == 0.1 for row in rows)
    assert all(
        row["small_image_iou_delta_ci_low"] == "" and row["small_image_iou_delta_ci_high"] == ""
        for row in rows
    )
    with (tmp_path / "reports/m13/category_comparisons.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        row = next(csv.DictReader(stream))
    assert row["small_image_iou_eligible_images"] == "1"
    assert row["small_image_iou_interval_note"] == "suppressed_low_image_count"
