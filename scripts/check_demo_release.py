"""Verify real artifact/input parity without publishing any source image or prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from robust_industrial_anomaly_detection.baselines.multiscale import load_projected_index
from robust_industrial_anomaly_detection.prototype.launch import load_inspector
from robust_industrial_anomaly_detection.prototype.quality import quality_model_dict
from robust_industrial_anomaly_detection.prototype.release import load_portable_engine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--examples",
        type=Path,
        required=True,
        help="existing permitted M12 local fixtures, not fresh holdouts",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    original = load_inspector(project)
    portable = load_portable_engine(args.bundle.resolve(), original.checkpoint_path)
    assert original.settings == portable.settings
    for name in ("seed", "image_size", "image_score_fraction", "gaussian_sigma", "categories"):
        assert getattr(original, name) == getattr(portable, name), name
    assert original.quality_model is not None and portable.quality_model is not None
    assert quality_model_dict(original.quality_model) == quality_model_dict(portable.quality_model)
    for category in original.categories:
        left = load_projected_index(original._index_paths[category])
        right = load_projected_index(portable._index_paths[category])
        assert all(np.array_equal(a, b) for a, b in zip(left, right, strict=True)), category
        first, second = original._calibration[category], portable._calibration[category]
        assert np.array_equal(first.scores, second.scores), category
        assert (
            first.empirical_threshold == second.empirical_threshold
            and first.pixel_threshold == second.pixel_threshold
        )
        a, b = asdict(original._localizations[category]), asdict(portable._localizations[category])
        for key in ("center", "scale"):
            assert np.array_equal(a.pop(key), b.pop(key)), (category, key)
        a.pop("provenance"), b.pop("provenance")
        assert a == b, category
    evaluated = recaptures = 0
    names = (
        "clean-normal.png",
        "clean-anomaly.png",
        "blur_severe-normal.png",
        "noise_severe-normal.png",
        "blank.png",
    )
    for name in names:
        payload = (args.examples / name).read_bytes()
        left_result = original.inspect_bytes(payload, filename=name, category="candle")
        right_result = portable.inspect_bytes(payload, filename=name, category="candle")
        before, after = left_result.public_dict(), right_result.public_dict()
        before.pop("timing_ms"), after.pop("timing_ms")
        if before != after:
            changed = [key for key in before if before[key] != after[key]]
            raise AssertionError(f"Portable prediction differs in: {changed}")
        evaluated += int(right_result.model_evaluated)
        recaptures += int(right_result.decision == "recapture")
    if not evaluated or not recaptures:
        raise AssertionError("Fixed fixtures must cover both evaluated and recapture paths.")
    print(
        json.dumps(
            {
                "packaging_parity_only_not_new_accuracy_evidence": True,
                "all_categories_exact_numeric_reference_parity": len(original.categories),
                "fixed_existing_fixture_results_equal_excluding_timing": len(names),
                "model_evaluated": evaluated,
                "recapture_before_model": recaptures,
                "previews_heatmaps_thresholds_scores_and_decisions_equal": True,
                "source_images_or_predictions_published": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
