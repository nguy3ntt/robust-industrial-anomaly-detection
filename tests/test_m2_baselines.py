from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robust_industrial_anomaly_detection.baselines.handcrafted import (
    BaselineError,
    HandcraftedNearestNeighbour,
    extract_handcrafted,
)
from robust_industrial_anomaly_detection.experiments.m2 import Sample, _load_masks


def _write_pattern(path: Path, offset: int) -> None:
    y, x = np.mgrid[0:64, 0:64]
    values = ((x * 5 + y * 3 + offset) % 256).astype(np.uint8)
    image = np.stack((values, np.roll(values, offset % 7, axis=0), 255 - values), axis=2)
    Image.fromarray(image, mode="RGB").save(path)


def test_handcrafted_descriptor_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    _write_pattern(path, 3)
    first = extract_handcrafted(path, size=64)
    second = extract_handcrafted(path, size=64)
    assert first.shape == (400,)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))


def test_handcrafted_nearest_neighbour_scores_training_examples_at_zero() -> None:
    training = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    model = HandcraftedNearestNeighbour()
    model.fit(training)
    scores = model.score(training)
    np.testing.assert_allclose(scores, 0.0, atol=1e-7)


def test_handcrafted_baseline_requires_fit() -> None:
    with pytest.raises(BaselineError, match="fitted"):
        HandcraftedNearestNeighbour().score(np.ones((1, 2), dtype=np.float32))


def test_patch_memory_is_deterministic_and_preserves_map_shape() -> None:
    pytest.importorskip("torch")
    from robust_industrial_anomaly_detection.baselines.resnet import PatchMemoryBank

    rng = np.random.default_rng(11)
    training = tuple(rng.normal(size=(4, 4, 8)).astype(np.float32) for _ in range(4))
    query = (rng.normal(size=(4, 4, 8)).astype(np.float32),)
    outputs: list[tuple[np.ndarray, tuple[np.ndarray, ...]]] = []
    for _ in range(2):
        model = PatchMemoryBank(
            seed=19,
            bank_size=24,
            projected_dimension=5,
            patches_per_image=8,
            image_score_fraction=0.25,
        )
        model.fit(training)
        outputs.append(model.score(query))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1][0], outputs[1][1][0])
    assert outputs[0][1][0].shape == (4, 4)


def test_mask_preprocessing_matches_center_crop_geometry(tmp_path: Path) -> None:
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[:, :2] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)
    sample = Sample(
        sample_id="sample",
        category="category",
        split="test",
        label="anomaly",
        image_path=tmp_path / "unused.png",
        mask_path=mask_path,
    )

    fitted = _load_masks([sample], (8, 8))[0]

    assert fitted.shape == (8, 8)
    assert not np.any(fitted)
