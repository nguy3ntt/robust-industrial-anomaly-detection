from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision.models import convnext_tiny  # type: ignore[import-untyped]

from robust_industrial_anomaly_detection.baselines.encoders import (
    ENCODERS,
    _ConvNeXtPyramid,
    verify_checkpoint,
)
from robust_industrial_anomaly_detection.baselines.handcrafted import BaselineError
from robust_industrial_anomaly_detection.baselines.multiscale import (
    PyramidFeatures,
    compose_feature_map,
)
from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.experiments.m2 import Sample
from robust_industrial_anomaly_detection.experiments.m11 import (
    M11Error,
    load_config,
    paired_comparison,
    partition,
    verify_commit,
)


def _config() -> dict:
    return load_config(
        Path(__file__).resolve().parents[1] / "configs/experiments/m11-encoder-comparison.yaml"
    )


def _samples() -> list[Sample]:
    return [
        Sample(
            f"{split}-{index}",
            "part",
            split,
            "normal" if split != "test" or index % 2 else "anomaly",
            Path("image.png"),
            None,
        )
        for split, count in (("train", 300), ("validation", 60), ("test", 40))
        for index in range(count)
    ]


def test_budgeted_fit_selection_is_order_independent_and_preserves_official_tests() -> None:
    config = _config()
    normal, validation, test = partition(_samples(), config, False)
    reverse = partition(list(reversed(_samples())), config, False)
    assert (normal, validation, test) == tuple(reverse)
    assert len(normal) == 256 and len(validation) == 60 and len(test) == 40
    assert not {s.sample_id for s in normal} & {s.sample_id for s in validation}


def test_training_anomalies_cannot_enter_even_a_smoke_selection() -> None:
    samples = _samples()
    samples[0] = replace(samples[0], label="anomaly")
    with pytest.raises(M11Error, match="only normals"):
        partition(samples, _config(), True)


def test_partition_rejects_cross_split_sample_id_reuse() -> None:
    samples = _samples()
    samples[300] = replace(samples[300], sample_id=samples[0].sample_id)
    with pytest.raises(M11Error, match="overlap"):
        partition(samples, _config(), False)


def test_convnext_three_stage_geometry_matches_the_reference_patch_grid() -> None:
    torch.set_num_threads(2)
    model = _ConvNeXtPyramid(convnext_tiny(weights=None)).eval()
    with torch.inference_mode():
        layers = model(torch.rand((1, 3, 224, 224), generator=torch.Generator().manual_seed(7)))
    assert [tuple(layer.shape) for layer in layers] == [
        (1, 96, 28, 28),
        (1, 192, 28, 28),
        (1, 384, 14, 14),
    ]
    features = PyramidFeatures(*(layer[0].permute(1, 2, 0).numpy() for layer in layers))
    grid = compose_feature_map(features, "multiscale")
    assert grid.shape == (28, 28, 672)
    assert np.all(np.isfinite(grid))
    np.testing.assert_allclose(np.linalg.norm(grid, axis=2), 1.0, atol=1e-5)


def test_checkpoint_requires_the_full_pinned_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    file = tmp_path / "hub" / "checkpoints" / ENCODERS["convnext_tiny"].url.rsplit("/", 1)[-1]
    file.parent.mkdir(parents=True)
    file.write_bytes(b"untrusted weights")
    with pytest.raises(BaselineError, match="modified"):
        verify_checkpoint("convnext_tiny")


def test_resume_commit_fails_on_mutated_numeric_artifact(tmp_path: Path) -> None:
    file = tmp_path / "index.npz"
    file.write_bytes(b"original")
    commit = tmp_path / "checkpoint.json"
    commit.write_text(json.dumps({"files": {file.name: sha256_file(file)}}), "utf-8")
    assert verify_commit(tmp_path, commit)
    file.write_bytes(b"changed")
    with pytest.raises(M11Error, match="changed"):
        verify_commit(tmp_path, commit)


def test_paired_encoder_bootstrap_is_identity_zero_and_order_sensitive() -> None:
    predictions = [
        {
            "sample_id": str(i),
            "label": "normal" if i < 5 else "anomaly",
            "score": float(i),
            "decision": "normal" if i < 5 else "anomalous",
            "foreground_iou": None if i < 5 else 0.5,
        }
        for i in range(10)
    ]
    result = paired_comparison(predictions, predictions, seed=4, samples=30, confidence=0.95)
    assert all(value == 0.0 for key, value in result.items() if "delta" in key)
    # The shared artifact CSV reader omits empty normal-mask metric cells.
    restored = [
        {key: value for key, value in row.items() if value is not None} for row in predictions
    ]
    assert paired_comparison(restored, restored, seed=4, samples=30, confidence=0.95) == result
    with pytest.raises(M11Error, match="order"):
        paired_comparison(
            predictions, list(reversed(predictions)), seed=4, samples=30, confidence=0.95
        )


def test_paired_encoder_bootstrap_reports_a_known_ranking_gain() -> None:
    old = [
        {
            "sample_id": str(i),
            "label": "normal" if i < 5 else "anomaly",
            "score": 0.5,
            "decision": "normal",
            "foreground_iou": None if i < 5 else 0.1,
        }
        for i in range(10)
    ]
    new = [
        {**row, "score": float(i), "foreground_iou": None if i < 5 else 0.2}
        for i, row in enumerate(old)
    ]
    result = paired_comparison(old, new, seed=4, samples=20, confidence=0.95)
    assert result["image_auroc_delta"] == 0.5
    assert result["image_auroc_delta_ci_low"] == 0.5
    assert result["foreground_iou_delta"] == pytest.approx(0.1)


def test_publication_refuses_changed_source_and_smoke_runs(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "m11_report", Path(__file__).resolve().parents[1] / "scripts/build_m11_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"units": 13, "metric_rows": 130, "comparison_rows": 65, "test_images": 3166}),
        "utf-8",
    )
    metadata = tmp_path / "metadata.json"
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(json.dumps({**_config(), "selected_categories": []}), "utf-8")
    payload = {
        "status": "complete",
        "mode": "full",
        **{f"{name}_checkpoint_sha256": encoder.sha256 for name, encoder in ENCODERS.items()},
        "output_hashes": {
            "summary.json": sha256_file(summary),
            "resolved_config.json": sha256_file(resolved),
        },
    }
    metadata.write_text(json.dumps(payload), "utf-8")
    config = {"source_run": str(tmp_path), "metadata_sha256": sha256_file(metadata)}
    assert module.verified_source(config) == tmp_path
    summary.write_text("changed", "utf-8")
    with pytest.raises(ValueError, match="modified encoder artifact"):
        module.verified_source(config)
    payload["mode"] = "smoke"
    metadata.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(ValueError, match="publication pin"):
        module.verified_source(config)
    config["metadata_sha256"] = sha256_file(metadata)
    with pytest.raises(ValueError, match="complete full run"):
        module.verified_source(config)
