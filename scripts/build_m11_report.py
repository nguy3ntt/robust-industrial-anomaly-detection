"""Rebuild aggregate encoder evidence from an exact full-run identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robust_industrial_anomaly_detection.baselines.encoders import ENCODERS
from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.evaluation.metrics import json_ready
from robust_industrial_anomaly_detection.experiments.m2 import _atomic_csv, _atomic_json, _read_csv
from robust_industrial_anomaly_detection.reporting.m7 import _atomic_text

METRICS = (
    "image_auroc",
    "image_ap",
    "image_fpr_at_95_tpr",
    "normal_fpr",
    "anomaly_confident_recall",
    "auto_coverage",
    "selective_risk",
    "pixel_auroc",
    "pixel_ap",
    "aupro_30",
    "foreground_iou",
    "foreground_dice",
    "foreground_precision",
    "foreground_recall",
    "normal_pixel_fpr",
    "normal_image_support_rate",
    "all_image_iou",
    "clean_score_rank_correlation",
    "mean_batch_feature_ms",
    "mean_batch_memory_ms",
    "mean_batch_pipeline_ms",
)
DELTAS = ("image_auroc", "image_ap", "normal_fpr", "anomaly_confident_recall", "foreground_iou")


def verified_source(config: dict[str, Any]) -> Path:
    run = Path(config["source_run"]).resolve(strict=True)
    metadata_path = run / "metadata.json"
    if sha256_file(metadata_path) != config["metadata_sha256"]:
        raise ValueError("encoder source metadata differs from the publication pin")
    metadata = json.loads(metadata_path.read_text("utf-8"))
    if metadata.get("status") != "complete" or metadata.get("mode") != "full":
        raise ValueError("encoder publication requires a complete full run")
    if any(
        metadata.get(f"{name}_checkpoint_sha256") != spec.sha256 for name, spec in ENCODERS.items()
    ):
        raise ValueError("encoder publication requires the declared pretrained weight identities")
    for filename, expected in metadata["output_hashes"].items():
        file = (run / filename).resolve(strict=True)
        file.relative_to(run)
        if sha256_file(file) != expected:
            raise ValueError(f"modified encoder artifact: {filename}")
    summary = json.loads((run / "summary.json").read_text("utf-8"))
    if (
        summary["units"],
        summary["metric_rows"],
        summary["comparison_rows"],
        summary["test_images"],
    ) != (13, 130, 65, 3166):
        raise ValueError("publication requires both encoders and all thirteen official-test units")
    resolved = json.loads((run / "resolved_config.json").read_text("utf-8"))
    expected_model = {
        "image_size": 224,
        "localization_size": 56,
        "batch_size": 16,
        "torch_threads": 4,
        "fit_normals_cap": 256,
        "bank_size": 512,
        "projected_dimension": 64,
        "candidate_limit": 4096,
        "patch_budget_per_image": 24,
        "image_score_fraction": 0.01,
        "gaussian_sigma": 4.0,
    }
    if (
        resolved.get("model") != expected_model
        or resolved.get("calibration")
        != {"anomaly_alpha": 0.05, "review_alpha": 0.20, "pixel_quantile": 0.995}
        or resolved.get("encoders") != ["resnet18", "convnext_tiny"]
        or resolved.get("experiment", {}).get("confidence") != 0.95
        or resolved.get("experiment", {}).get("bootstrap_samples") != 1000
        or resolved.get("experiment", {}).get("seed") != 20260915
        or resolved.get("benchmark") != {"normal_samples": 5, "anomaly_samples": 5}
        or resolved.get("selected_categories") != []
    ):
        raise ValueError("publication source differs from the declared matched-budget protocol")
    return run


def build(config: dict[str, Any], project: Path) -> dict[str, str]:
    run = verified_source(config)
    rows: list[dict[str, Any]] = list(_read_csv(run / "metrics.csv"))
    comparisons: list[dict[str, Any]] = list(_read_csv(run / "comparisons.csv"))
    benchmarks = _read_csv(run / "benchmarks.csv")
    conditions = ("clean", "jpeg_severe", "blur_severe", "noise_severe", "geometric_severe")
    visa_config = yaml.safe_load((project / "configs/data/visa.yaml").read_text("utf-8"))
    units = [("visa", category) for category in visa_config["categories"]] + [("ksdd2", "surface")]
    expected = {
        (dataset, category, encoder, condition)
        for dataset, category in units
        for encoder in ("resnet18", "convnext_tiny")
        for condition in conditions
    }
    actual = [(row["dataset"], row["category"], row["encoder"], row["condition"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("missing, duplicated, or unexpected encoder metric units")
    if len(comparisons) != 65 or len(benchmarks) != 26:
        raise ValueError("incomplete comparison/resource coverage")
    expected_pairs = {
        (dataset, category, condition) for dataset, category in units for condition in conditions
    }
    actual_pairs = [(row["dataset"], row["category"], row["condition"]) for row in comparisons]
    expected_resources = {
        (dataset, category, encoder)
        for dataset, category in units
        for encoder in ("resnet18", "convnext_tiny")
    }
    actual_resources = [(row["dataset"], row["category"], row["encoder"]) for row in benchmarks]
    if set(actual_pairs) != expected_pairs or set(actual_resources) != expected_resources:
        raise ValueError("duplicated or missing paired/resource units")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["condition"], row["encoder"])].append(row)
    macros: list[dict[str, Any]] = []
    for (dataset, condition, encoder), group in sorted(groups.items()):
        means = {key: float(np.mean([float(row[key]) for row in group])) for key in METRICS}
        macros.append(
            {
                "dataset": dataset,
                "condition": condition,
                "encoder": encoder,
                "categories": len(group),
                **means,
            }
        )
    paired: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        paired[(row["dataset"], row["condition"])].append(row)
    deltas: list[dict[str, Any]] = []
    for (dataset, condition), group in sorted(paired.items()):
        result: dict[str, Any] = {
            "dataset": dataset,
            "condition": condition,
            "categories": len(group),
        }
        for metric in DELTAS:
            key = f"{metric}_delta"
            values = np.asarray([float(row[key]) for row in group])
            if dataset == "visa":
                rng = np.random.default_rng(20260915)
                estimates = np.mean(rng.choice(values, (1000, len(values))), axis=1)
                low, high = np.quantile(estimates, [0.025, 0.975])
            else:
                low, high = group[0][key + "_ci_low"], group[0][key + "_ci_high"]
            result.update(
                {
                    key: float(np.mean(values)),
                    key + "_ci_low": float(low),
                    key + "_ci_high": float(high),
                }
            )
        deltas.append(result)
    resources = []
    for dataset in ("visa", "ksdd2"):
        for encoder in ("resnet18", "convnext_tiny"):
            group = [
                row for row in benchmarks if row["dataset"] == dataset and row["encoder"] == encoder
            ]
            timings = [
                row
                for unit in group
                for row in _read_csv(
                    run / "units" / dataset / unit["category"] / encoder / "benchmark_timings.csv"
                )
            ]
            resources.append(
                {
                    "dataset": dataset,
                    "encoder": encoder,
                    "samples": len(timings),
                    "median_ms": float(np.median([row["total_ms"] for row in timings])),
                    "p95_ms": float(np.quantile([row["total_ms"] for row in timings], 0.95)),
                    "mean_feature_ms": float(np.mean([row["feature_ms"] for row in timings])),
                    "active_parameters": group[0]["active_parameters"],
                    "checkpoint_bytes": group[0]["checkpoint_bytes"],
                    "all_category_index_bytes": sum(row["index_bytes"] for row in group),
                    "mean_post_fit_process_bytes": float(
                        np.mean(
                            [
                                row["working_set_bytes"]
                                for row in group
                                if row.get("working_set_bytes") is not None
                            ]
                        )
                    ),
                }
            )
    visibility = []
    for dataset in ("visa", "ksdd2"):
        for condition in conditions:
            group = groups[(dataset, condition, "resnet18")]
            visibility.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "test_images": int(sum(row["samples"] for row in group)),
                    "anomaly_images": int(sum(row["anomaly_samples"] for row in group)),
                    "visible_anomaly_masks": int(
                        sum(row["visible_anomaly_samples"] for row in group)
                    ),
                    "empty_anomaly_masks": int(sum(row["empty_anomaly_masks"] for row in group)),
                }
            )
    visible_index = {(row["dataset"], row["condition"]): row for row in visibility}
    severe_policy = []
    for dataset in ("visa", "ksdd2"):
        for encoder in ("resnet18", "convnext_tiny"):
            group = [
                row
                for row in macros
                if row["dataset"] == dataset
                and row["encoder"] == encoder
                and row["condition"] != "clean"
            ]
            severe_policy.append(
                {
                    "dataset": dataset,
                    "encoder": encoder,
                    "conditions": len(group),
                    **{
                        key: float(np.mean([row[key] for row in group]))
                        for key in (
                            "normal_fpr",
                            "anomaly_confident_recall",
                            "auto_coverage",
                            "selective_risk",
                        )
                    },
                }
            )
    output = (project / config["output_directory"]).resolve()
    output.relative_to((project / "reports").resolve())
    output.mkdir(parents=True, exist_ok=True)
    for filename, table in (
        ("category_metrics.csv", rows),
        ("paired_comparisons.csv", comparisons),
        ("macro_metrics.csv", macros),
        ("macro_deltas.csv", deltas),
        ("resources.csv", resources),
        ("mask_visibility.csv", visibility),
        ("severe_operating_summary.csv", severe_policy),
    ):
        _atomic_csv(output / filename, json_ready(table))
    index = {(row["dataset"], row["condition"], row["encoder"]): row for row in macros}
    delta_index = {(row["dataset"], row["condition"]): row for row in deltas}
    clean = []
    for dataset in ("visa", "ksdd2"):
        for encoder in ("resnet18", "convnext_tiny"):
            row = index[(dataset, "clean", encoder)]
            clean.append(
                f"| {dataset} | {encoder} | {row['image_auroc']:.4f} | {row['image_ap']:.4f} "
                f"| {row['normal_fpr']:.2%} | {row['anomaly_confident_recall']:.2%} "
                f"| {row['aupro_30']:.4f} | {row['foreground_iou']:.4f} |"
            )
    stress = []
    for dataset in ("visa", "ksdd2"):
        for condition in conditions[1:]:
            old, new = [
                index[(dataset, condition, encoder)] for encoder in ("resnet18", "convnext_tiny")
            ]
            stress.append(
                f"| {dataset} | {condition} | {old['image_auroc']:.4f} | {new['image_auroc']:.4f} "
                f"| {old['normal_fpr']:.2%} | {new['normal_fpr']:.2%} "
                f"| {old['foreground_iou']:.4f} | {new['foreground_iou']:.4f} |"
            )
    category_lines = []
    for row in comparisons:
        if row["condition"] == "clean":
            category_lines.append(
                f"| {row['dataset']}/{row['category']} | {row['image_auroc_delta']:+.4f} "
                f"| [{row['image_auroc_delta_ci_low']:+.4f}, "
                f"{row['image_auroc_delta_ci_high']:+.4f}] "
                f"| {100 * row['normal_fpr_delta']:+.2f} | {row['foreground_iou_delta']:+.4f} |"
            )
    resource_lines = [
        f"| {row['dataset']} | {row['encoder']} | {row['active_parameters'] / 1e6:.2f} "
        f"| {row['checkpoint_bytes'] / 1e6:.2f} | {row['all_category_index_bytes'] / 1e6:.2f} "
        f"| {row['median_ms']:.1f} | {row['p95_ms']:.1f} |"
        for row in resources
    ]
    v = delta_index[("visa", "clean")]
    k = delta_index[("ksdd2", "clean")]
    visa_visibility = visible_index[("visa", "clean")]
    ksdd_visibility = visible_index[("ksdd2", "clean")]
    ksdd_policies = {row["encoder"]: row for row in severe_policy if row["dataset"] == "ksdd2"}
    severe_old, severe_new = ksdd_policies["resnet18"], ksdd_policies["convnext_tiny"]
    deployment_conclusion = (
        "The measured substitution does not support replacing ResNet: clean image "
        "ranking and visible-defect overlap are lower on both domains, while the "
        "checkpoint and executed encoder are larger. Retain the current ResNet-based "
        "inspector; do not cherry-pick per-category test winners."
        if v["image_auroc_delta"] < 0 and k["image_auroc_delta"] < 0
        else "The measured trade-offs do not automatically authorize replacing the live inspector."
    )
    clean_table, stress_table = "\n".join(clean), "\n".join(stress)
    category_table, resource_table = "\n".join(category_lines), "\n".join(resource_lines)
    operating_table = "\n".join(
        f"| {row['dataset']} | {row['encoder']} | {row['normal_fpr']:.2%} "
        f"| {row['anomaly_confident_recall']:.2%} | {row['auto_coverage']:.2%} "
        f"| {row['selective_risk']:.2%} |"
        for row in severe_policy
    )
    clean_header = (
        "| Dataset | Encoder | Image AUROC | Image AP | Conformal normal FPR | "
        "Confident anomaly recall | AUPRO@30 | Visible-defect IoU |"
    )
    category_header = (
        "| Dataset/category | Image AUROC delta | 95% paired-image interval | "
        "Normal FPR delta (pp) | Defect IoU delta |"
    )
    stress_header = (
        "| Dataset | Condition | ResNet AUROC | ConvNeXt AUROC | ResNet normal FPR | "
        "ConvNeXt normal FPR | ResNet defect IoU | ConvNeXt defect IoU |"
    )
    resource_header = (
        "| Dataset | Encoder | Active parameters (M) | Checkpoint (MB) | "
        "All indexes (MB) | Single-image median (ms) | p95 (ms) |"
    )
    content = f"""# M11 modern encoder comparison

## Outcome and scope

The frozen ConvNeXt-Tiny substitution changes clean VisA macro image AUROC by
{v["image_auroc_delta"]:+.4f} relative to matched-budget ResNet-18, with a 95%
category-bootstrap interval [{v["image_auroc_delta_ci_low"]:+.4f},
{v["image_auroc_delta_ci_high"]:+.4f}]. Its visible-defect IoU difference is
{v["foreground_iou_delta"]:+.4f}; conformal normal FPR changes by
{100 * v["normal_fpr_delta"]:+.2f} percentage points. On KSDD2 the clean image AUROC difference is
{k["image_auroc_delta"]:+.4f}, with paired-image interval
[{k["image_auroc_delta_ci_low"]:+.4f}, {k["image_auroc_delta_ci_high"]:+.4f}].

This is a completed practical representation comparison, not automatic model
promotion. The live M10 inspector, its memories/calibration, and quality gate
remain unchanged. Representation gains must be balanced against false alarms,
localization, severe-shift failures, and measured cost below. No single pooled
score establishes industrial readiness or physical defect diagnosis.

{deployment_conclusion}

There is a real trade-off under KSDD2's four selected severe conditions: mean
normal FPR falls from {severe_old["normal_fpr"]:.2%} to {severe_new["normal_fpr"]:.2%},
but confident defect recall also falls from {severe_old["anomaly_confident_recall"]:.2%}
to {severe_new["anomaly_confident_recall"]:.2%}. These benefits and losses are both
retained below; fewer alerts are not automatically better inspection.

This result applies to the common first-three-stage, 64-dimensional projected
memory recipe, initially developed around ResNet. It is not a best-tuned
ConvNeXt result, a test of every modern encoder, or proof of intrinsic
architectural inferiority. Different features/projection budgets would require
a separately declared comparison, not tuning against these test outcomes.

## Matched method

Both frozen ImageNet-1K encoders supply their first three spatial stages,
aligned to a 28x28 patch grid. ResNet yields 448 feature channels and ConvNeXt
672. Both use 224 RGB input, identical geometry and normalization, a seeded
64-dimensional normalized projection, 4,096 candidates, 512 memory entries,
24 patches per training normal split equally over clean and one frozen mild
nuisance view, top-1%-patch image scores, and sigma-4 map smoothing.

At most 256 sample-ID-ranked training normals per category are used for both
encoders. All existing validation normals independently calibrate each encoder;
all 3,166 official test images remain included (2,162 VisA and 1,004 KSDD2).
No test-based layer, threshold, augmentation, or category-winner search is used.
The first three stages are executed, not the classifier or fourth stage.
Capacity and pretraining recipe differ: this is not architecture-only causality.

The study preserves VisA center-crop and KSDD2 full-frame geometry, using a
common 56x56 mask/evidence grid for both encoders. This VisA localization grid
differs from M10's 224 grid; do not compare those IoUs directly. Sampled/cropped
away positive masks are counted explicitly in category tables and excluded only
from undefined foreground metrics, never from image detection. Historical M5/M9
numbers use different training budgets and are not the matched reference here.
On clean inputs, {visa_visibility["empty_anomaly_masks"]} of
{visa_visibility["anomaly_images"]} VisA anomaly masks and
{ksdd_visibility["empty_anomaly_masks"]} of {ksdd_visibility["anomaly_images"]}
KSDD2 anomaly masks have no positive pixels after the declared geometry/rasterization.
This is a localization-resolution limitation, not evidence that those products
are normal; both encoders use the same visible subset. Counts for every condition
are in `mask_visibility.csv`.

Thresholds use clean calibration normals only: upper-tail conformal p <= .05
alerts, .05 < p <= .20 requests review, and p > .20 is normal; pixel thresholds
use the 99.5th percentile of smoothed calibration pixels. No capture gate hides
shifted encoder failures; neither encoder uses M10 cleanup or seeded growth.
Clean false-alarm validity needs exchangeability and
does not guarantee shifted control.

See the [fixed protocol](../docs/M11_ENCODER_PROTOCOL.md), including checkpoint
sources, licence boundary, representation choices, and computational limits.

## Clean results

VisA values are equal-category means; KSDD2 is a separate domain.

{clean_header}
|---|---|---:|---:|---:|---:|---:|---:|
{clean_table}

Foreground overlap never rewards empty normal masks. Aggregate CSVs additionally
include pixel ranking/AP, Dice, precision/recall, normal-pixel FPR, selective
coverage/risk, and legacy all-image IoU under distinct names. A high ranking
score does not imply accurate binary defect boundaries or acceptable miss rates.

## Clean category differences: ConvNeXt minus ResNet

{category_header}
|---|---:|---|---:|---:|
{category_table}

Intervals use 1,000 stratified paired image resamples; VisA macro intervals
resample category differences equally. Shifted category intervals use 100
descriptive resamples. One seed and reused public benchmarks limit confirmation;
nominal intervals are not simultaneous multiple-comparison guarantees. Paired
fit/calibration IDs, test ordering, and labels are checked before aggregation.
Intervals condition on the fitted models and fixed normal calibration; they do
not include refitting uncertainty or establish independent capture-session validity.

## Severe capture shifts

{stress_header}
|---|---|---:|---:|---:|---:|---:|---:|
{stress_table}

These four severe conditions are a fixed diagnostic subset of M4, not all
severities or a representative factory capture distribution. Clean calibration
and a newer encoder do not remove the need for capture control and human review.
Score-rank stability relative to the paired clean images is in the tables.

The following are equal-weight descriptive means over the four severe conditions,
with equal category weights within VisA. Lower false alarms are not sufficient
if confident defect recall also collapses. Review cases are not automatic errors
or guaranteed normal products; selective risk applies only to automatic decisions.

| Dataset | Encoder | Normal FPR | Confident defect recall | Automatic coverage | Selective risk |
|---|---|---:|---:|---:|---:|
{operating_table}

## Resources

{resource_header}
|---|---|---:|---:|---:|---:|---:|
{resource_table}

Single-image measurements use the same five normal and five anomalous samples
per category, a warm-up, four PyTorch CPU threads, and one-image calls. They
include image/mask decoding, feature extraction, memory scoring, and map
interpolation/smoothing, but exclude browser rendering, quality gating,
publication, calibration, and metric calculation. Thus they are not the M6/M10
end-to-end browser benchmark. The saved environment identifies CPU/CUDA mode.
Batch costs are separate fields and should not be reported as single-image latency.

Checkpoint bytes include the original full classifier checkpoint; active
parameters count only the executed first three stages. Projected memory is
512x64 float32 for both encoders; different feature dimensions change projection
and compressed-index bytes. Process working set includes shared libraries,
metric buffers, and allocator/order history, so it is descriptive rather than
isolated encoder RAM. No claim of equal total compute or byte budget is made.

## Reproducibility and publication

Authoritative run: `{run.name}`. It contains 130 category/encoder/condition
metric rows, 65 paired comparisons, and 26 matched single-image resource jobs.
Config, source code, environment, manifest/audit, weight, fitted-bank, calibration,
condition, and benchmark identities are recorded. Every committed output is
hash-verified before resume/republication. Interrupted runs retain completed units.

Public [aggregate tables and manifest](m11/) exclude images, per-image
predictions, weights, and feature banks. Follow the
[manual verification guide](../docs/M11_MANUAL_TESTING.md) for local checks and
the exact report rebuild command. The existing application is intentionally not
silently replaced by this offline research comparison.
"""
    report = project / "reports/M11_ENCODER_COMPARISON.md"
    _atomic_text(report, content)
    files = [
        report,
        *(
            output / name
            for name in (
                "category_metrics.csv",
                "paired_comparisons.csv",
                "macro_metrics.csv",
                "macro_deltas.csv",
                "resources.csv",
                "mask_visibility.csv",
                "severe_operating_summary.csv",
            )
        ),
    ]
    hashes = {file.relative_to(project).as_posix(): sha256_file(file) for file in files}
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
            "weights_published": False,
        },
    )
    return {"report": str(report), "package_sha256": package_hash}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild exact hash-pinned encoder comparison evidence"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/reporting/m11.yaml"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / args.config).read_text("utf-8"))
    print(json.dumps(build(config, project), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
