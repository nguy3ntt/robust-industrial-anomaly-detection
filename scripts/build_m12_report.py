"""Rebuild aggregate capture-gate evidence from a completed, hash-pinned run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.experiments.m2 import _atomic_csv, _atomic_json, _read_csv
from robust_industrial_anomaly_detection.experiments.m8 import load_m8_settings
from robust_industrial_anomaly_detection.experiments.m12 import verify_files
from robust_industrial_anomaly_detection.reporting.m7 import _atomic_text


def verified_source(config: dict[str, Any], project: Path) -> Path:
    run = Path(config["source_run"]).resolve(strict=True)
    if sha256_file(run / "metadata.json") != config["metadata_sha256"]:
        raise ValueError("quality source metadata differs from publication pin")
    metadata = json.loads((run / "metadata.json").read_text("utf-8"))
    if metadata.get("status") != "complete" or metadata.get("mode") != "full":
        raise ValueError("quality publication requires a complete full run")
    verify_files(run, metadata["output_hashes"])
    resolved = json.loads((run / "resolved_config.json").read_text("utf-8"))
    expected = yaml.safe_load(
        (project / "configs/experiments/m12-quality-generalization.yaml").read_text("utf-8")
    )
    if resolved != {**expected, "mode": "full"}:
        raise ValueError("quality publication source differs from declared protocol")
    rows: list[dict[str, Any]] = list(_read_csv(run / "metrics.csv"))
    visa = yaml.safe_load((project / "configs/data/visa.yaml").read_text("utf-8"))
    protocol = load_m8_settings(project / expected["source_protocol"])
    categories = [("visa", category) for category in visa["categories"]] + [("ksdd2", "surface")]
    conditions = ["clean", *[item.identifier for item in protocol.conditions]]
    expected_keys = {
        (dataset, category, condition, label, gate)
        for dataset, category in categories
        for condition in conditions
        for label in ("normal", "anomaly")
        for gate in ("v1", "v2")
    }
    actual = [
        tuple(row[key] for key in ("dataset", "category", "condition", "label", "gate"))
        for row in rows
    ]
    if len(actual) != len(expected_keys) or set(actual) != expected_keys:
        raise ValueError("quality publication has missing/duplicate category conditions")
    expected_counts = {
        ("visa", "normal"): 962,
        ("visa", "anomaly"): 1200,
        ("ksdd2", "normal"): 894,
        ("ksdd2", "anomaly"): 110,
    }
    for (dataset, label), count in expected_counts.items():
        for condition in conditions:
            for gate in ("v1", "v2"):
                observed = sum(
                    int(row["samples"])
                    for row in rows
                    if (row["dataset"], row["label"], row["condition"], row["gate"])
                    == (dataset, label, condition, gate)
                )
                if observed != count:
                    raise ValueError(
                        "quality publication does not preserve every provider test image"
                    )
    return run


def build(config: dict[str, Any], project: Path) -> dict[str, str]:
    run = verified_source(config, project)
    summaries: list[dict[str, Any]] = list(_read_csv(run / "summary.csv"))
    metrics: list[dict[str, Any]] = list(_read_csv(run / "metrics.csv"))
    resources: list[dict[str, Any]] = list(_read_csv(run / "resources.csv"))
    if {(row["dataset"], row["gate"]) for row in summaries} != {
        (dataset, gate) for dataset in ("visa", "ksdd2") for gate in ("v1", "v2")
    } or len(summaries) != 4:
        raise ValueError("quality summary coverage is incomplete or duplicated")
    summary_table = "\n".join(
        f"| {row['dataset']} | {row['gate']} | {row['clean_normal_rejection_rate']:.2%} "
        f"| {row['clean_anomaly_rejection_rate']:.2%} | {row['shifted_normal_rejection_rate']:.2%} "
        f"| {row['shifted_normal_false_alert_interception_rate']:.2%} |"
        for row in summaries
    )
    summary_index = {(row["dataset"], row["gate"]): row for row in summaries}
    visa_old, visa_new = [summary_index[("visa", gate)] for gate in ("v1", "v2")]
    ksdd_old, ksdd_new = [summary_index[("ksdd2", gate)] for gate in ("v1", "v2")]
    ksdd_normals = {
        (row["condition"], row["gate"]): row
        for row in metrics
        if row["dataset"] == "ksdd2" and row["label"] == "normal"
    }
    blur_old, blur_new = [
        ksdd_normals[("blur_severe", gate)]["quality_rejection_rate"] for gate in ("v1", "v2")
    ]
    noise_old, noise_new = [
        ksdd_normals[("noise_severe", gate)]["quality_rejection_rate"] for gate in ("v1", "v2")
    ]
    frozen = load_m8_settings(project / "configs/experiments/m8-quality-gate.yaml")
    conditions = ["clean", *[condition.identifier for condition in frozen.conditions]]
    normal_table = []
    for dataset in ("visa", "ksdd2"):
        for condition in conditions:
            for gate in ("v1", "v2"):
                rows = [
                    row
                    for row in metrics
                    if (row["dataset"], row["condition"], row["label"], row["gate"])
                    == (dataset, condition, "normal", gate)
                ]
                total = sum(row["samples"] for row in rows)
                values = {
                    key: sum(row[key] * row["samples"] for row in rows) / total
                    for key in (
                        "quality_rejection_rate",
                        "guarded_auto_anomalous_rate",
                        "guarded_actionable_rate",
                    )
                }
                normal_table.append(
                    f"| {dataset} | {condition} | {gate} | {values['quality_rejection_rate']:.2%} "
                    f"| {values['guarded_auto_anomalous_rate']:.2%} "
                    f"| {values['guarded_actionable_rate']:.2%} |"
                )
    anomaly_table = "\n".join(
        f"| {row['dataset']} | {row['gate']} | {row['clean_anomaly_source_anomalous_rate']:.2%} "
        f"| {row['clean_anomaly_guarded_auto_anomalous_rate']:.2%} "
        f"| {row['clean_anomaly_rejection_rate']:.2%} "
        f"| {row['clean_anomaly_guarded_actionable_rate']:.2%} |"
        for row in summaries
    )
    normal_lines = "\n".join(normal_table)
    resource_table = "\n".join(
        f"| {row['dataset']} | {row['gate']} | {row['mean_gate_ms']:.2f} |" for row in resources
    )
    summary_header = (
        "| Dataset | Gate | Clean normal recapture | Clean anomaly recapture | "
        "Shifted normal recapture | Shifted false-alert interception |"
    )
    anomaly_header = (
        "| Dataset | Gate | Original confident defect recall | "
        "Remaining confident defect recall | Recapture | Total operator action |"
    )
    content = f"""# Capture-quality gate generalization

## Outcome and scope

This completed comparison evaluates the original v1 gate and one fixed v2 gate
on both VisA and KolektorSDD2. The detector, anomaly calibration, official splits,
all twelve capture shifts, and sample seeds are unchanged. Only gate measurements
and their normal-only fitted references change; no new backbone training occurs.

{summary_header}
|---|---|---:|---:|---:|---:|
{summary_table}

KSDD2 shifted false-alert interception changes from
{ksdd_old["shifted_normal_false_alert_interception_rate"]:.2%} to
{ksdd_new["shifted_normal_false_alert_interception_rate"]:.2%}; clean-normal
recapture changes from {ksdd_old["clean_normal_rejection_rate"]:.2%} to
{ksdd_new["clean_normal_rejection_rate"]:.2%}. VisA interception changes from
{visa_old["shifted_normal_false_alert_interception_rate"]:.2%} to
{visa_new["shifted_normal_false_alert_interception_rate"]:.2%}, and clean-defect
deferral changes from {visa_old["clean_anomaly_rejection_rate"]:.2%} to
{visa_new["clean_anomaly_rejection_rate"]:.2%}. These domain-specific trade-offs
are why v2 remains an explicit comparison, not a silent universal upgrade.

The main cross-domain problem remains unresolved: KSDD2 severe-blur normal
recapture changes only from {blur_old:.2%} to {blur_new:.2%}, and severe-noise
recapture is {noise_old:.2%} versus {noise_new:.2%}. Most such inputs still pass.
The small aggregate gain is not evidence that these proxies generalize reliably.
Retain the historical default; v2 is an auditable research comparison, not a
recommended production replacement. Additional operator action and slightly
greater clean-defect deferral must be considered alongside interception.

Interception counts original review OR anomalous calls on normal images that
are now withheld. It does not mean the products are classified correctly or that
operator workload fell. All shifted summaries are descriptive sample-weighted
totals across repeated conditions, not independent trials. VisA summaries here
are sample-weighted, not equal-category means. All category/condition/label
results, including non-target shifts and regressions, remain in the public CSV.

## Fixed intervention

V2 retains the original eight quality measurements and adds log fine/coarse
energy for contrast-normalized focus, plus a contrast-normalized robust noise
estimate from low-structure pixels. Both gates use median/MAD-IQR normal
references, maximum directional deviation, and clean-normal split-conformal
calibration at alpha .02. No anomalies or shifted images fit the gate.

Each domain/category receives its own original training/calibration normals.
VisA retains centered 224x224 crops; KSDD2 retains full-frame bilinear 224x224
geometry. This is local normal adaptation, not a VisA-to-unseen-product gate.
Historical v1 dispositions were recomputed and required to match saved results
for every evaluated image and condition. See the [fixed protocol](../docs/M12_QUALITY_PROTOCOL.md).

## Normal-input trade-offs

All rates use ALL normal inputs as denominator, including recaptures. The last
column is recapture OR review OR anomalous: unresolved required operator action,
not just model review. A low remaining automatic false-positive rate can coexist
with a very high recapture burden. The full CSV also includes exact binomial
rejection intervals by category/condition, conditional on image independence.

| Dataset | Condition | Gate | Recapture | Remaining automatic anomalous | Total operator action |
|---|---|---|---:|---:|---:|
{normal_lines}

## Defect deferral is not successful detection

The following clean-anomaly rates separately expose loss of immediate model
decisions and increased unresolved work. Recapture returns no anomaly score or
heatmap. It must not be counted as confident defect detection or as normal.

{anomaly_header}
|---|---|---:|---:|---:|---:|
{anomaly_table}

## Resources and deployment

| Dataset | Gate | Mean gate-only CPU ms |
|---|---|---:|
{resource_table}

These paired offline timings cover all test-condition gate calls, not image
decoding, transforms, anomaly inference, browser rendering, or a fresh end-to-end
latency benchmark. Fixed v1-then-v2 execution order and host load limit precision.

Historical M6/M8/M10 configurations remain reproducible. The separate M12
configuration enables v2 with the same ResNet detector and cleaned contours;
it does not silently replace the user's running server. Integrity checks bind
the v2 artifact to the same M5 calibration and trusted VisA manifest.
KSDD2 remains a separate research evaluation, not a browser product category.

## Limits and reproducibility

This follow-up was motivated by observed M9 failures on reused benchmarks.
It is exploratory development evidence, not untouched external confirmation,
proof of camera failure, or industrial safety certification. Texture, defects,
noise and blur can still confound the measurements; combined corruptions and
physically grouped camera/session captures remain unvalidated. Clean conformal
validity requires exchangeability and does not guarantee shifted false alarms.
Repeated shifts and unknown item/session dependence prevent treating pooled
counts as independent significance evidence. No post-test feature/threshold sweep
or category-specific winner selection is performed.

Authoritative run: `{run.name}`. All 3,166 official test images are evaluated in
13 conditions with both gates (82,316 gate decisions), producing 676 aggregate
category/condition/label/gate rows. Normal fit/calibration membership, source
metadata, input/code/config fingerprints, and condition checkpoints are recorded.
Interrupted runs resume only from matching, verified inputs. Public files contain
aggregates only: no images, per-image predictions, weights, or normal references.
See the [manual verification guide](../docs/M12_MANUAL_TESTING.md).
"""
    output = (project / config["output_directory"]).resolve()
    output.relative_to((project / "reports").resolve())
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for filename, table in (
        ("category_metrics.csv", metrics),
        ("summary.csv", summaries),
        ("resources.csv", resources),
    ):
        path = output / filename
        _atomic_csv(path, table)
        files.append(path)
    report = project / "reports/M12_QUALITY_GENERALIZATION.md"
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
            "weights_published": False,
            "per_image_predictions_published": False,
        },
    )
    return {"report": str(report), "package_sha256": package_hash}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reporting/m12.yaml"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / args.config).read_text("utf-8"))
    print(json.dumps(build(config, project), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
