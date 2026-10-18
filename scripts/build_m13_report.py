"""Rebuild the aggregate-only small-defect study from its exact completed-run pin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robust_industrial_anomaly_detection.baselines.encoders import ENCODERS
from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.experiments.m2 import _atomic_csv, _atomic_json, _read_csv
from robust_industrial_anomaly_detection.experiments.m12 import verify_files
from robust_industrial_anomaly_detection.experiments.m13 import READOUTS, load_config
from robust_industrial_anomaly_detection.reporting.m7 import _atomic_text

METRICS = (
    "image_auroc",
    "image_ap",
    "normal_fpr",
    "anomaly_confident_recall",
    "review_rate",
    "auto_coverage",
    "foreground_iou",
    "foreground_dice",
    "foreground_recall",
    "foreground_precision",
    "small_image_iou",
    "small_image_recall",
    "small_region_hit_rate",
    "small_region_pixel_recall",
    "normal_pixel_fpr",
    "normal_image_support_rate",
    "background_pixel_fpr",
    "legacy_coarse_iou",
)
DELTAS = ("foreground_iou", "small_image_iou", "small_region_hit_rate", "normal_fpr")


def verified_source(config: dict[str, Any], project: Path) -> Path:
    run = Path(config["source_run"]).resolve(strict=True)
    if sha256_file(run / "metadata.json") != config["metadata_sha256"]:
        raise ValueError("small-defect source metadata differs from publication pin")
    metadata = json.loads((run / "metadata.json").read_text("utf-8"))
    if metadata.get("status") != "complete" or metadata.get("mode") != "full":
        raise ValueError("small-defect publication requires a complete full run")
    if metadata.get("weights_sha256") != ENCODERS["resnet18"].sha256:
        raise ValueError("small-defect publication used unexpected encoder weights")
    verify_files(run, metadata["output_hashes"])
    expected = load_config(project / "configs/experiments/m13-small-defects.yaml")
    resolved = json.loads((run / "resolved_config.json").read_text("utf-8"))
    if resolved != {
        **expected,
        "mode": "full",
        "selected_datasets": sorted(expected["datasets"]),
        "selected_categories": [],
    }:
        raise ValueError("small-defect publication differs from declared full protocol")
    summary = json.loads((run / "summary.json").read_text("utf-8"))
    if (
        summary["units"],
        summary["metric_rows"],
        summary["comparison_rows"],
        summary["test_images"],
    ) != (13, 117, 78, 3166):
        raise ValueError("small-defect publication has incomplete unit/test coverage")
    visa = yaml.safe_load((project / "configs/data/visa.yaml").read_text("utf-8"))
    units = [("visa", name) for name in visa["categories"]] + [("ksdd2", "surface")]
    rows: list[dict[str, Any]] = list(_read_csv(run / "metrics.csv"))
    expected_keys = {
        (dataset, category, condition, readout)
        for dataset, category in units
        for condition in expected["conditions"]
        for readout in READOUTS
    }
    keys = [(row["dataset"], row["category"], row["condition"], row["readout"]) for row in rows]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError("missing/duplicate small-defect metric units")
    paired = _read_csv(run / "comparisons.csv")
    pair_keys = [
        (row["dataset"], row["category"], row["condition"], row["contrast"]) for row in paired
    ]
    expected_pairs = {
        (dataset, category, condition, contrast)
        for dataset, category in units
        for condition in expected["conditions"]
        for contrast in ("resolution", "tiling")
    }
    if len(pair_keys) != len(expected_pairs) or set(pair_keys) != expected_pairs:
        raise ValueError("missing/duplicate paired small-defect comparisons")
    if (
        len(_read_csv(run / "resources.csv")) != 26
        or len(_read_csv(run / "region_metrics.csv")) != 585
    ):
        raise ValueError("incomplete small-defect resource or region coverage")
    for dataset in ("visa", "ksdd2"):
        for condition in expected["conditions"]:
            for readout in READOUTS:
                if sum(
                    row["samples"]
                    for row in rows
                    if (row["dataset"], row["condition"], row["readout"])
                    == (dataset, condition, readout)
                ) != (2162 if dataset == "visa" else 1004):
                    raise ValueError("not every official test image is included")
    for dataset, category in units:
        unit = run / "units" / dataset / category
        fits = [
            json.loads((unit / mode / "fit.json").read_text("utf-8"))
            for mode in ("global", "tiled")
        ]
        if any(fits[0][key] != fits[1][key] for key in ("fit_ids", "calibration_ids")):
            raise ValueError("global/tiled normal fit or calibration IDs differ")
        for fit in fits:
            if (
                fit["bank_entries"] != 512
                or fit["candidate_count"] != 3072
                or fit["patch_budget_per_capture"] != 24
                or len(fit["fit_ids"]) != 128
            ):
                raise ValueError("small-defect normal memory budget differs")
            if set(fit["fit_ids"]) & set(fit["calibration_ids"]):
                raise ValueError("small-defect fit/calibration IDs overlap")
        for condition in expected["conditions"]:
            predictions: list[dict[str, Any]] = list(
                _read_csv(unit / "conditions" / condition / "predictions.csv")
            )
            grouped = {
                name: [row for row in predictions if row["readout"] == name] for name in READOUTS
            }
            identity_keys = ("sample_id", "label", "mask_pixels", "small_regions", "small_image")
            identities = [
                [tuple(row[key] for key in identity_keys) for row in grouped[name]]
                for name in READOUTS
            ]
            if not identities[0] or not all(ids == identities[0] for ids in identities):
                raise ValueError(
                    "small-defect paired test IDs, labels, or common ground truth differ"
                )
            test_ids = [row["sample_id"] for row in grouped["global336"]]
            if len(test_ids) != len(set(test_ids)) or set(test_ids) & (
                set(fits[0]["fit_ids"]) | set(fits[0]["calibration_ids"])
            ):
                raise ValueError("small-defect test IDs duplicate or leak into normal references")
            for coarse, fine in zip(grouped["global56"], grouped["global336"], strict=True):
                if any(coarse[key] != fine[key] for key in ("score", "conformal_p", "decision")):
                    raise ValueError("resolution-only comparison changed image inference")
    return run


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _number(value: Any, *, percent: bool = False, signed: bool = False) -> str:
    if value is None:
        return "—"
    if percent:
        return f"{value:+.2%}" if signed else f"{value:.2%}"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def build(config: dict[str, Any], project: Path) -> dict[str, str]:
    run = verified_source(config, project)
    metrics: list[dict[str, Any]] = list(_read_csv(run / "metrics.csv"))
    comparisons: list[dict[str, Any]] = list(_read_csv(run / "comparisons.csv"))
    regions: list[dict[str, Any]] = list(_read_csv(run / "region_metrics.csv"))
    resources: list[dict[str, Any]] = list(_read_csv(run / "resources.csv"))
    for comparison in comparisons:
        unit = run / "units" / comparison["dataset"] / comparison["category"]
        predictions = _read_csv(unit / "conditions" / comparison["condition"] / "predictions.csv")
        common = [row for row in predictions if row["readout"] == "global336"]
        for metric, count in (
            ("small_image_iou", sum(bool(row["small_image"]) for row in common)),
            ("small_region_hit_rate", sum(float(str(row["small_regions"])) > 0 for row in common)),
        ):
            comparison[metric + "_eligible_images"] = count
            comparison[metric + "_interval_note"] = (
                "conditional_image_bootstrap" if count >= 2 else "suppressed_low_image_count"
            )
            if count < 2:
                comparison[metric + "_delta_ci_low"] = None
                comparison[metric + "_delta_ci_high"] = None
    macros = []
    deltas = []
    macro_regions = []
    for dataset in ("visa", "ksdd2"):
        for condition in ("clean", "blur_severe", "noise_severe"):
            for readout in READOUTS:
                group = [
                    row
                    for row in metrics
                    if (row["dataset"], row["condition"], row["readout"])
                    == (dataset, condition, readout)
                ]
                macros.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "readout": readout,
                        "categories": len(group),
                        **{key: _mean(group, key) for key in METRICS},
                        **{
                            key: int(sum(row[key] for row in group))
                            for key in (
                                "samples",
                                "anomaly_samples",
                                "visible_anomaly_samples",
                                "empty_anomaly_masks",
                                "coarse_lost_positive_masks",
                                "source_positive_fine_empty",
                                "small_images",
                                "small_regions",
                            )
                        },
                        "small_image_categories": sum(
                            row.get("small_image_iou") is not None for row in group
                        ),
                        "small_region_categories": sum(
                            row.get("small_region_hit_rate") is not None for row in group
                        ),
                    }
                )
                for stratum in ("tiny", "small", "large", "small_inclusive", "all"):
                    selected = [
                        row
                        for row in regions
                        if (row["dataset"], row["condition"], row["readout"], row["stratum"])
                        == (dataset, condition, readout, stratum)
                    ]
                    macro_regions.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "readout": readout,
                            "stratum": stratum,
                            "regions": int(sum(row["regions"] for row in selected)),
                            "eligible_categories": sum(
                                row.get("region_hit_rate") is not None for row in selected
                            ),
                            "region_hit_rate": _mean(selected, "region_hit_rate"),
                            "region_pixel_recall": _mean(selected, "region_pixel_recall"),
                        }
                    )
            for contrast in ("resolution", "tiling"):
                selected = [
                    row
                    for row in comparisons
                    if (row["dataset"], row["condition"], row["contrast"])
                    == (dataset, condition, contrast)
                ]
                result: dict[str, Any] = {
                    "dataset": dataset,
                    "condition": condition,
                    "contrast": contrast,
                }
                for metric in DELTAS:
                    key = metric + "_delta"
                    available = [row for row in selected if row.get(key) is not None]
                    values = np.asarray([row[key] for row in available], dtype=float)
                    if not len(values):
                        result[key] = result[key + "_ci_low"] = result[key + "_ci_high"] = None
                    elif dataset == "visa":
                        rng = np.random.default_rng(20260915)
                        estimates = np.mean(rng.choice(values, (1000, len(values))), axis=1)
                        result.update(
                            {
                                key: float(np.mean(values)),
                                key + "_ci_low": float(np.quantile(estimates, 0.025)),
                                key + "_ci_high": float(np.quantile(estimates, 0.975)),
                            }
                        )
                        if len(values) < 2:
                            result[key + "_ci_low"] = result[key + "_ci_high"] = None
                    else:
                        result.update(
                            {
                                key: available[0][key],
                                key + "_ci_low": available[0].get(key + "_ci_low"),
                                key + "_ci_high": available[0].get(key + "_ci_high"),
                            }
                        )
                    result[key + "_categories"] = len(values)
                deltas.append(result)
    macro_resources: list[dict[str, Any]] = []
    for dataset in ("visa", "ksdd2"):
        for mode in ("global", "tiled"):
            group = [row for row in resources if (row["dataset"], row["mode"]) == (dataset, mode)]
            raw: list[dict[str, Any]] = [
                row
                for entry in group
                for row in _read_csv(
                    run / "units" / dataset / entry["category"] / "benchmark_timings.csv"
                )
                if row["mode"] == mode
            ]
            times = [float(row["total_ms"]) for row in raw]
            macro_resources.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "samples": len(times),
                    "median_ms": float(np.median(times)),
                    "p95_ms": float(np.quantile(times, 0.95)),
                    "encoder_views": 1 if mode == "global" else 4,
                    "all_index_bytes": int(sum(row["index_bytes"] for row in group)),
                    "max_observed_post_fit_process_bytes": int(
                        max(row["post_fit_working_set_bytes"] or 0 for row in group)
                    ),
                }
            )
    index = {(row["dataset"], row["condition"], row["readout"]): row for row in macros}
    delta_index = {(row["dataset"], row["condition"], row["contrast"]): row for row in deltas}
    clean = "\n".join(
        f"| {row['dataset']} | {row['readout']} | {_number(row['foreground_iou'])} "
        f"| {_number(row['small_image_iou'])} "
        f"| {_number(row['small_region_hit_rate'], percent=True)} "
        f"| {_number(row['normal_fpr'], percent=True)} "
        f"| {_number(row['normal_pixel_fpr'], percent=True)} "
        f"| {_number(row['normal_image_support_rate'], percent=True)} |"
        for row in macros
        if row["condition"] == "clean"
    )
    delta_lines = "\n".join(
        f"| {row['dataset']} | {row['condition']} | {row['contrast']} "
        f"| {_number(row['foreground_iou_delta'], signed=True)} "
        f"| [{_number(row['foreground_iou_delta_ci_low'], signed=True)}, "
        f"{_number(row['foreground_iou_delta_ci_high'], signed=True)}] "
        f"| {_number(row['small_image_iou_delta'], signed=True)} "
        f"| {_number(row['normal_fpr_delta'], percent=True, signed=True)} |"
        for row in deltas
    )
    size_lines = "\n".join(
        f"| {row['dataset']} | {row['readout']} | {row['stratum']} | {row['regions']} "
        f"| {row['eligible_categories']} | {_number(row['region_hit_rate'], percent=True)} "
        f"| {_number(row['region_pixel_recall'], percent=True)} |"
        for row in macro_regions
        if row["condition"] == "clean"
    )
    visibility = "\n".join(
        f"| {dataset} | {index[(dataset, 'clean', 'global336')]['anomaly_samples']} "
        f"| {index[(dataset, 'clean', 'global336')]['visible_anomaly_samples']} "
        f"| {index[(dataset, 'clean', 'global336')]['coarse_lost_positive_masks']} "
        f"| {index[(dataset, 'clean', 'global336')]['source_positive_fine_empty']} "
        f"| {index[(dataset, 'clean', 'global336')]['small_images']} "
        f"| {index[(dataset, 'clean', 'global336')]['small_image_categories']} |"
        for dataset in ("visa", "ksdd2")
    )
    resource_lines = "\n".join(
        f"| {row['dataset']} | {row['mode']} | {row['encoder_views']} "
        f"| {row['median_ms']:.1f} | {row['p95_ms']:.1f} | {row['all_index_bytes'] / 1e6:.2f} |"
        for row in macro_resources
    )
    small_lines = "\n".join(
        f"| {row['dataset']} | {row['contrast']} "
        f"| {_number(row['small_image_iou_delta'], signed=True)} "
        f"| [{_number(row['small_image_iou_delta_ci_low'], signed=True)}, "
        f"{_number(row['small_image_iou_delta_ci_high'], signed=True)}] "
        f"| {_number(row['small_region_hit_rate_delta'], percent=True, signed=True)} "
        f"| [{_number(row['small_region_hit_rate_delta_ci_low'], percent=True, signed=True)}, "
        f"{_number(row['small_region_hit_rate_delta_ci_high'], percent=True, signed=True)}] |"
        for row in deltas
        if row["condition"] == "clean"
    )
    image_lines = "\n".join(
        f"| {row['dataset']} | {row['readout']} | {_number(row['image_auroc'])} "
        f"| {_number(row['image_ap'])} "
        f"| {_number(row['anomaly_confident_recall'], percent=True)} "
        f"| {_number(row['review_rate'], percent=True)} "
        f"| {_number(row['auto_coverage'], percent=True)} |"
        for row in macros
        if row["condition"] == "clean" and row["readout"] != "global56"
    )
    v, k = [delta_index[(dataset, "clean", "tiling")] for dataset in ("visa", "ksdd2")]
    clean_header = (
        "| Dataset | Readout | Foreground IoU | Small-image IoU | "
        "Small-region hit | Normal image FPR | Normal pixel FPR | Normal contour incidence |"
    )
    delta_header = (
        "| Dataset | Condition | Contrast | Foreground IoU delta | 95% interval | "
        "Small-image IoU delta | Normal FPR delta |"
    )
    visibility_header = (
        "| Dataset | Source anomaly images | Fine visible masks | "
        "Fine positives lost at 56 | Source-positive/fine-empty | Small images | "
        "Eligible small-image categories |"
    )
    region_header = (
        "| Dataset | Readout | Stratum | Regions | Eligible categories | "
        "Hit rate | Mean region pixel recall |"
    )
    small_header = (
        "| Dataset | Contrast | Small-image IoU delta | 95% interval | "
        "Small-region hit delta | 95% interval |"
    )
    image_header = (
        "| Dataset | Readout | Image AUROC | Image AP | Confident defect recall | "
        "Review rate | Automatic coverage |"
    )
    noise_v = delta_index[("visa", "noise_severe", "tiling")]["normal_fpr_delta"]
    noise_k = delta_index[("ksdd2", "noise_severe", "tiling")]["normal_fpr_delta"]
    content = f"""# Small-defect localization: resolution and overlapping tiles

## Outcome and scope

Four overlapping detail tiles change clean VisA foreground IoU by
{_number(v["foreground_iou_delta"], signed=True)} relative to the matched global
fine readout; small-image IoU changes by {_number(v["small_image_iou_delta"], signed=True)}.
KSDD2 changes by {_number(k["foreground_iou_delta"], signed=True)} and
{_number(k["small_image_iou_delta"], signed=True)}, respectively. Full trade-offs,
including false positives, region sizes, shifts and cost, are retained below.
This is a completed exploratory comparison, not automatic deployment promotion.
The historical M10 inspector and M12 capture comparison remain unchanged.

The VisA small-image gain is uncertain (its paired category interval spans zero),
and tiny-region hit rate decreases despite improved overall clean overlap.
Severe-noise normal-image FPR increases by
{_number(noise_v, percent=True, signed=True)}
on VisA and
{_number(noise_k, percent=True, signed=True)}
on KSDD2 (percentage-point changes). KSDD2 has only ONE small-defect image;
its apparent small-defect success cannot establish generalization. Tiling is
therefore not an unqualified small-defect or robustness improvement.

## Controlled method

`global56` and `global336` share the SAME global 224-input ResNet, normal memory,
image scores and decisions. Only pixel readout/normal calibration changes.
Upsampling adds no encoder information. `tiled336` processes four 224 crops at
112-pixel stride, obtaining 1.5x finer linear detail with reduced per-tile context.
Raised-cosine normalized blending covers edges and suppresses tile seams; both
methods apply the same sigma-6 smoothing in common-frame units after merging.

Both models fit the same seeded 128 normal training IDs per category, all original
calibration normals, a 512-entry/64-dimensional memory, 4,096-candidate ceiling,
and 24 patches per ORIGINAL capture (12 clean, 12 from its one assigned mild
augmentation). Tiling does not multiply its memory/patch budget by four. Both
use the identical frozen local ImageNet ResNet-18 checkpoint and projection.
This bounded-fit experiment is not a numerical M5/M9/M10/M11 reproduction.

All official test IDs remain included. Severe blur/noise act on the same 336
canvas before either view; blur units scale with frame resolution. Downsampling
and tiling therefore expose different effective noise/detail, a pipeline
trade-off rather than architecture-only causality. Other shifts are not covered.
There is no gate to hide failures, no component cleanup, threshold sweep, or
per-category test winner selection. Thresholds use only clean normals: pixel
quantile .995 per readout; conformal image levels .05/.20 per model. Image scores
use the top 1% of the unsmoothed COMMON map, unlike historical native-patch scores.

## Ground truth and visibility

All primary spatial measurements use the SAME conservative 336 mask, produced
with floating-point area sampling and any-positive coverage. This preserves
small positives but can expand subpixel boundaries; it is not native-mask
boundary precision. Coarse support is lifted to this common grid, so anomalies
with an empty 56 mask are NOT silently dropped from primary foreground overlap.
Historical-style coarse IoU remains separately named in the aggregate tables.
VisA keeps its centered crop, KSDD2 its full frame. Crop loss is still possible.
Keeping a positive annotation also does not guarantee its RGB appearance survives
canvas downsampling; the source-resolution defect signal can still be lost.

{visibility_header}
|---|---:|---:|---:|---:|---:|---:|
{visibility}

Small-image total mask area is <= .1% of the common frame. Region strata use
8-connected fine ground truth: tiny <= .01%, small > .01% and <= .1%, large > .1%.
`small_inclusive` combines tiny and small. These are relative image-area cohorts,
not physical defect dimensions. A region hit means >=10% pixel recall, not good
boundaries or a confident product-level detection. Always inspect FPR/IoU too.

## Clean operating points

VisA metrics are equal-category means; undefined cohorts are not filled with
zero. Eligible category/region/image counts remain explicit in the tables. KSDD2
is a separate single-domain result. Empty normal masks never raise foreground IoU.

{clean_header}
|---|---|---:|---:|---:|---:|---:|---:|
{clean}

The following image-level context makes reduced field-of-view trade-offs visible.
Global56 is omitted here because its image inference is identical to global336.
Review is unresolved, not a correct automatic product decision.

{image_header}
|---|---|---:|---:|---:|---:|---:|
{image_lines}

## Clean small-defect paired evidence

The same size cohorts apply to both readouts. Undefined cohorts remain undefined;
small-image and region eligibility counts are in the visibility/size tables.
Public image-bootstrap intervals are omitted with fewer than two eligible
images; category-bootstrap intervals are omitted with fewer than two eligible
categories. A single observed small defect cannot establish reliable uncertainty,
even when a conditional bootstrap of that same image gives a degenerate interval.
Points/counts remain visible and raw frozen-run estimates remain in local artifacts.

{small_header}
|---|---|---:|---|---:|---|
{small_lines}

## Paired changes and severe capture conditions

`resolution` is global336 minus global56; `tiling` is tiled336 minus global336.
Both use common fine ground truth. Each image bootstrap keeps all its regions
together; VisA macro intervals instead resample category differences. These are
conditional on the fitted models/calibration and exploratory, not simultaneous
or independent capture-session guarantees. Small-cohort finite-resample/category
counts are retained in CSVs. Image false positives are calibrated decisions, not
pixel support or quality recapture. The two global readouts have identical image
scores/p-values/decisions by construction and verified before publication.

{delta_header}
|---|---|---|---:|---|---:|---:|
{delta_lines}

Full per-condition readout tables additionally retain recall, Dice, precision,
review, coverage, normal contour incidence and background-pixel false positives.

## Clean region-size cohorts

{region_header}
|---|---|---|---:|---:|---:|---:|
{size_lines}

Within each category region rates weight regions equally; VisA summary rates
weight eligible categories equally. Regions in one image are not independent
trials. No claim of significant small-region gain is made from unclustered counts.

## Resources and deployment boundary

| Dataset | Model | Encoder views | Single-image median ms | p95 ms | All indexes MB |
|---|---|---:|---:|---:|---:|
{resource_lines}

Both models are warmed up and timed on the same three normals/three anomalies
per category in one-image calls (tiles internally batched). Timings include
image/mask decoding, features, memory search, stitching and smoothing; exclude
browser rendering, quality gating and report publication. They are NOT a new
end-to-end browser benchmark. Fixed model order, host/cache history and observed
process working sets limit resource precision. Continuous maps are interpolated;
display pixels should never be described as independent encoder features.

Do not silently replace the current inspector: this study changes geometry,
image-score aggregation, normal memories and calibration. An attractive overlap
score alone does not justify its compute, false-positive or robustness costs.
Native/grouped capture validation and an explicitly verified runtime would be
needed for deployment. Keep all category/condition regressions visible.

## Reproducibility and privacy

Authoritative source: `{run.name}`. It contains 13 units, all 3,166 official test
images, three conditions, three readouts (28,494 predictions), 117 metric rows
and 78 paired comparisons. Dataset/audit/config/code/weights, consumed source
files, normal membership, fitted indexes, calibrations, condition checkpoints,
and environment are verified before reuse. Actual smoke interruption/resume is
tested. Data, per-image/region outputs, memories and qualitative arrays remain
local. Public files are aggregates and source/output hashes only.

Read the [fixed protocol](../docs/M13_SMALL_DEFECT_PROTOCOL.md) and
[manual verification guide](../docs/M13_MANUAL_TESTING.md). Reused benchmarks,
one fit seed, conservative mask rasterization, lost crop context and unknown
capture/session dependence limit confirmation and physical interpretation.
This remains inspection decision support, not product certification or diagnosis.
"""
    output = (project / config["output_directory"]).resolve()
    output.relative_to((project / "reports").resolve())
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for filename, rows in (
        ("category_metrics.csv", metrics),
        ("macro_metrics.csv", macros),
        ("category_comparisons.csv", comparisons),
        ("macro_deltas.csv", deltas),
        ("category_regions.csv", regions),
        ("macro_regions.csv", macro_regions),
        ("category_resources.csv", resources),
        ("resources.csv", macro_resources),
    ):
        path = output / filename
        _atomic_csv(path, rows)
        files.append(path)
    report = project / "reports/M13_SMALL_DEFECT_LOCALIZATION.md"
    _atomic_text(report, content)
    files.append(report)
    hashes = {path.relative_to(project).as_posix(): sha256_file(path) for path in files}
    package_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    _atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "source_run": run.name,
            "source_metadata_sha256": config["metadata_sha256"],
            "output_hashes": hashes,
            "package_sha256": package_hash,
            "source_images_published": False,
            "per_image_predictions_published": False,
            "weights_published": False,
        },
    )
    return {"report": str(report), "package_sha256": package_hash}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reporting/m13.yaml"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / args.config).read_text("utf-8"))
    print(json.dumps(build(config, project), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
