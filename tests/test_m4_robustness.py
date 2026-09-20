from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robust_industrial_anomaly_detection.evaluation.robustness import (
    bootstrap_mean_interval,
    heatmap_stability,
    paired_image_delta_intervals,
    spearman_stability,
)
from robust_industrial_anomaly_detection.robustness.transforms import (
    ShiftSpec,
    align_score_map,
    apply_shift,
)


def test_noise_is_deterministic_and_never_changes_mask() -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:20, 12:18] = True
    spec = ShiftSpec("noise", "noise", 1, {"standard_deviation": 0.05})

    first = apply_shift(image, mask, spec, seed=7, sample_key="sample")
    second = apply_shift(image, mask, spec, seed=7, sample_key="sample")

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, mask)
    assert first.inverse_affine is None
    assert not np.array_equal(first.image, image)


def test_geometric_shift_moves_mask_and_can_align_explanation() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:18, 12:18] = True
    image[mask] = 255
    spec = ShiftSpec(
        "translation",
        "geometric",
        1,
        {"angle_degrees": 0.0, "translate_x": 3.0, "translate_y": -2.0},
    )

    shifted = apply_shift(image, mask, spec, seed=1, sample_key="sample")
    aligned = align_score_map(shifted.mask.astype(np.float32), shifted.inverse_affine)

    assert not np.array_equal(shifted.mask, mask)
    assert np.mean((aligned > 0.5) == mask) > 0.99


def test_stability_metrics_have_expected_identity_and_reversal() -> None:
    clean = np.arange(16, dtype=np.float32).reshape(4, 4)
    correlations, overlaps = heatmap_stability([clean], [clean], top_fraction=0.25)

    assert correlations.tolist() == pytest.approx([1.0])
    assert overlaps.tolist() == pytest.approx([1.0])
    assert spearman_stability([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_paired_bootstrap_preserves_pairing_and_delta_direction() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    clean = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    shifted = [0.8, 0.7, 0.6, 0.4, 0.3, 0.2]
    intervals = paired_image_delta_intervals(
        labels,
        clean,
        shifted,
        threshold=0.5,
        seed=9,
        samples=100,
        confidence=0.95,
    )

    assert intervals["image_auroc"]["high"] < 0.0
    assert intervals["test_normal_fpr"]["low"] > 0.0


def test_single_unit_mean_interval_is_exact() -> None:
    assert bootstrap_mean_interval([0.25], seed=1, samples=10, confidence=0.95) == (
        0.25,
        0.25,
    )


def test_m4_configuration_has_two_severities_for_every_shift_family() -> None:
    import yaml

    root = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/experiments/m4-robustness.yaml").read_text(
            encoding="utf-8"
        )
    )
    conditions = root["conditions"]
    grouped = {
        family: sorted(item["severity"] for item in conditions if item["family"] == family)
        for family in {item["family"] for item in conditions}
    }

    assert grouped == {
        "blur": [1, 2],
        "geometric": [1, 2],
        "jpeg": [1, 2],
        "noise": [1, 2],
        "photometric": [1, 2],
        "resize": [1, 2],
    }
    assert len(root["intervention"]["mild_augmentation_pool"]) == 6
