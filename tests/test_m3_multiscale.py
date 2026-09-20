from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from robust_industrial_anomaly_detection.baselines.multiscale import (
    ProjectedPatchMemory,
    PyramidFeatures,
    compose_feature_map,
    load_projected_index,
)
from robust_industrial_anomaly_detection.experiments.m3 import load_m3_settings


def _pyramid(seed: int) -> PyramidFeatures:
    rng = np.random.default_rng(seed)
    return PyramidFeatures(
        rng.normal(size=(4, 4, 2)).astype(np.float32),
        rng.normal(size=(4, 4, 3)).astype(np.float32),
        rng.normal(size=(2, 2, 4)).astype(np.float32),
    )


def test_feature_composition_has_expected_grid_and_unit_norm() -> None:
    pyramid = _pyramid(3)
    mid = compose_feature_map(pyramid, "mid")
    multiscale = compose_feature_map(pyramid, "multiscale")
    assert mid.shape == (2, 2, 7)
    assert multiscale.shape == (4, 4, 9)
    np.testing.assert_allclose(np.linalg.norm(mid, axis=2), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(multiscale, axis=2), 1.0, atol=1e-6)


def test_kcenter_memory_is_bounded_deterministic_and_serializable(tmp_path: Path) -> None:
    training = tuple(compose_feature_map(_pyramid(seed), "multiscale") for seed in range(5))
    query = (compose_feature_map(_pyramid(20), "multiscale"),)
    models: list[ProjectedPatchMemory] = []
    outputs: list[tuple[np.ndarray, tuple[np.ndarray, ...]]] = []
    for _ in range(2):
        model = ProjectedPatchMemory(
            seed=11,
            bank_size=12,
            projected_dimension=5,
            patches_per_image=6,
            candidate_limit=24,
            image_score_fraction=0.25,
            selection="kcenter",
        )
        stats = model.fit(training)
        assert stats.candidate_count == 24
        assert stats.selected_count == 12
        assert stats.bank_bytes == 12 * 5 * 4
        models.append(model)
        outputs.append(model.score(query))
    np.testing.assert_array_equal(models[0].bank, models[1].bank)
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1][0], outputs[1][1][0])
    assert outputs[0][1][0].shape == (4, 4)

    index_path = tmp_path / "index.npz"
    compressed_bytes = models[0].save(index_path, metadata={"category": "widget"})
    bank, projection = load_projected_index(index_path)
    assert compressed_bytes == index_path.stat().st_size
    np.testing.assert_array_equal(bank, models[0].bank)
    np.testing.assert_array_equal(projection, models[0].projection)


def test_projected_unified_memory_reuses_shared_projection() -> None:
    training = tuple(compose_feature_map(_pyramid(seed), "multiscale") for seed in range(4))
    category_model = ProjectedPatchMemory(
        seed=9,
        bank_size=8,
        projected_dimension=4,
        patches_per_image=4,
        candidate_limit=16,
        image_score_fraction=0.1,
        selection="kcenter",
    )
    category_model.fit(training)
    unified = ProjectedPatchMemory(
        seed=9,
        bank_size=8,
        projected_dimension=4,
        patches_per_image=4,
        candidate_limit=16,
        image_score_fraction=0.1,
        selection="kcenter",
    )
    unified.fit_projected(category_model.bank[:4], category_model.projection)
    scores, maps = unified.score((training[0],))
    assert scores.shape == (1,)
    assert maps[0].shape == training[0].shape[:2]


def test_random_memory_reports_finite_coverage() -> None:
    training = tuple(compose_feature_map(_pyramid(seed), "mid") for seed in range(4))
    model = ProjectedPatchMemory(
        seed=4,
        bank_size=6,
        projected_dimension=4,
        patches_per_image=4,
        candidate_limit=12,
        image_score_fraction=0.25,
        selection="random",
    )
    stats = model.fit(training)
    assert stats.selected_count == 6
    assert np.isfinite(stats.coverage_radius)
    assert stats.coverage_radius >= 0.0


def test_frozen_ablation_sequence_is_additive() -> None:
    settings = load_m3_settings(Path("configs/experiments/m3-multiscale.yaml"))
    variants = {variant.identifier: variant for variant in settings.variants}
    assert list(variants) == [
        "reference_mid_random_2048",
        "compact_mid_random_512",
        "multiscale_random_512",
        "multiscale_bounded_random_512",
        "multiscale_coreset_512",
    ]
    assert variants["reference_mid_random_2048"].candidate_limit == 50_000
    assert variants["multiscale_random_512"].candidate_limit == 50_000
    assert variants["multiscale_bounded_random_512"].candidate_limit == 4_096
    assert variants["multiscale_coreset_512"].candidate_limit == 4_096
