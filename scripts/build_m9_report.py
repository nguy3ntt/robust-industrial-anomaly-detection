"""Publish aggregate evidence from an explicitly hash-pinned M9 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.experiments.m2 import _atomic_csv, _atomic_json, _read_csv
from robust_industrial_anomaly_detection.reporting.m7 import _atomic_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the aggregate M9 external validation report"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/reporting/m9.yaml"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / args.config).read_text(encoding="utf-8"))
    run = Path(config["source_run"])
    metadata_path = run / "metadata.json"
    if sha256_file(metadata_path) != config["metadata_sha256"]:
        raise ValueError("M9 source metadata does not match the publication pin")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("mode") != "full":
        raise ValueError("M9 publication requires a complete full source run")
    for filename, expected in metadata["output_hashes"].items():
        path = (run / filename).resolve(strict=True)
        path.relative_to(run.resolve(strict=True))
        if sha256_file(path) != expected:
            raise ValueError(f"M9 artifact changed: {filename}")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows = _read_csv(run / "metrics.csv")
    clean = {str(row["model"]): row for row in rows if row["condition"] == "clean"}
    if len(rows) != 26 or len(clean) != 2 or summary["official_test_samples"] != 1004:
        raise ValueError("M9 public evidence must contain both models and all thirteen conditions")
    output = (project / config["output_directory"]).resolve()
    output.relative_to((project / "reports").resolve())
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output / "external_metrics.csv", rows)

    def number(row: dict[str, object], key: str) -> float:
        value = row[key]
        if not isinstance(value, (float, int)):
            raise ValueError(f"non-numeric aggregate metric {key}")
        return float(value)

    augmented = clean["augmented_coreset_512"]
    shifted = [
        row
        for row in rows
        if row["model"] == "augmented_coreset_512" and row["condition"] != "clean"
    ]
    false_alerts = sum(number(row, "false_alerts") for row in shifted)
    intercepted = sum(number(row, "false_alerts_intercepted") for row in shifted)
    shifted_rejection = float(
        np.mean([number(row, "quality_normal_rejection_rate") for row in shifted])
    )
    shift_means = {
        model: {
            key: float(
                np.mean(
                    [
                        number(row, key)
                        for row in rows
                        if row["model"] == model and row["condition"] != "clean"
                    ]
                )
            )
            for key in ("image_auroc", "normal_fpr")
        }
        for model in clean
    }
    model_lines = []
    for model, row in clean.items():
        model_lines.append(
            f"| {model} | {number(row, 'image_auroc'):.4f} | {number(row, 'image_ap'):.4f} "
            f"| {number(row, 'normal_fpr'):.4f} | {number(row, 'anomaly_confident_recall'):.4f} "
            f"| {number(row, 'pixel_auroc'):.4f} | {number(row, 'pixel_ap'):.4f} "
            f"| {number(row, 'aupro_30'):.4f} |"
        )
    condition_lines = []
    for row in rows:
        if row["model"] != "augmented_coreset_512":
            continue
        condition_lines.append(
            f"| {row['condition']} | {number(row, 'image_auroc'):.4f} "
            f"| {number(row, 'normal_fpr'):.4f} "
            f"| {number(row, 'quality_normal_rejection_rate'):.4f} "
            f"| {number(row, 'guarded_normal_fpr'):.4f} "
            f"| {number(row, 'guarded_auto_anomaly_recall'):.4f} "
            f"| {number(row, 'guarded_anomaly_actionable_rate'):.4f} |"
        )
    clean_table = "\n".join(model_lines)
    condition_table = "\n".join(condition_lines)
    clean_header = (
        "| Model | Image AUROC | Image AP | Normal FPR | Confident recall | "
        "Pixel AUROC | Pixel AP | AUPRO@30 |"
    )
    shift_header = (
        "| Condition | Image AUROC | Ungated normal FPR | Normal recapture | "
        "Guarded normal FPR | Auto defect recall | Defect action coverage |"
    )
    content = f"""# M9 independent validation on KolektorSDD2

## Outcome

The frozen normal-only method was fitted and calibrated on a second industrial
dataset, then evaluated on all 1,004 official test images. The augmented model
achieves clean image AUROC {number(augmented, "image_auroc"):.4f}, with a stratified
95% bootstrap interval [{number(augmented, "image_auroc_ci_low"):.4f},
{number(augmented, "image_auroc_ci_high"):.4f}]. Its clean conformal normal false-positive
rate is {number(augmented, "normal_fpr"):.2%}, with exact 95% interval
[{number(augmented, "normal_fpr_ci_low"):.2%}, {number(augmented, "normal_fpr_ci_high"):.2%}].

The locally calibrated capture gate rejects {number(augmented, "quality_normal_rejection_rate"):.2%}
of clean normals and {shifted_rejection:.2%} of shifted normals across the frozen
stress conditions. It intercepts {int(intercepted):,} of {int(false_alerts):,}
shifted-normal false alerts ({intercepted / false_alerts:.2%}). False alerts here
include both anomalous and review dispositions from the ungated detector.

The current gate therefore does not generalize uniformly: severe JPEG is
reliably rejected, but severe blur, noise, and geometry often pass even when
they trigger substantial false-positive rates. The second dataset supports
normal-only adaptation of the detector and exposes a limitation of the existing
gate. It does not justify transferring the VisA 93.22% interception result to
other industrial captures. The frozen gate is retained for this validation;
any redesign requires a new protocol rather than tuning against these test labels.

This validates the method with normal adaptation to a second dataset. It does
not establish zero-shot VisA-memory transfer or factory deployment readiness.

## Data and frozen method

- Official source: [ViCoS KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/).
- Licence: CC BY-NC-SA 4.0; noncommercial portfolio/academic use.
- Fit: {summary["normal_fit_samples"]:,} provider training normals.
- Calibration: {summary["normal_calibration_samples"]:,} training-derived normals.
- Official test: 894 normals and 110 defects, all preserved.
- Provider training defects: all 246 excluded from fitting/calibration.
- Archive copies: two byte-identical training copy files explicitly excluded;
  originals remain, with their hashes recorded in the dataset audit.
- Leakage checks: exact and confirmed perceptual components before calibration
  splitting; the acquired dataset contains no confirmed duplicate pairs.
- Frozen ImageNet ResNet-18, multi-scale features, 64-dimensional projection,
  4,096 candidates, 512-entry k-center banks, and 24 total patches per image.
- Plain memory uses 24 clean patches. Augmentation splits that budget into 12
  clean and 12 mildly shifted normal patches using the pre-existing M4 pool.
- Full-frame image/mask resize to 224x224; localization evaluated at 56x56.
- Fixed conformal anomaly/review levels 0.05/0.20 and quality rejection level 0.02.

The [frozen protocol](../docs/M9_EVALUATION_PROTOCOL.md) and
[dataset card](../docs/datasets/KSDD2.md) describe the split, geometry, sampling,
licence, and provenance boundaries.

## Clean results

{clean_header}
|---|---:|---:|---:|---:|---:|---:|---:|
{clean_table}

All ranking/localization results above use ungated scores on the full official
test partition. Pixel metrics use the declared 56x56 grid, not native resolution;
their values cannot be compared directly with VisA's 224x224 localization.
The clean evaluation loses {int(number(augmented, "anomaly_masks_empty_at_evaluation_grid"))}
positive mask(s) through raster downsampling; their images remain in detection
evaluation and this count is disclosed rather than silently excluding cases.

## Capture gate and human action

Across the twelve stress conditions, equal-weight mean image AUROC is
{shift_means["plain_coreset_512"]["image_auroc"]:.4f} for the plain model and
{shift_means["augmented_coreset_512"]["image_auroc"]:.4f} for augmentation. Mean
ungated normal FPR is {shift_means["plain_coreset_512"]["normal_fpr"]:.2%} versus
{shift_means["augmented_coreset_512"]["normal_fpr"]:.2%}. These are descriptive
paired-condition summaries; conditions repeatedly transform the same test images
and are not independent samples for a significance claim.

On clean defects, the gate requests recapture for
{number(augmented, "quality_anomaly_rejection_rate"):.2%} of images. Automatic confident
defect recall changes from {number(augmented, "anomaly_confident_recall"):.2%} to
{number(augmented, "guarded_auto_anomaly_recall"):.2%}. Total required operator action,
including recapture, review, and anomalous outcomes, covers
{number(augmented, "guarded_anomaly_actionable_rate"):.2%} of clean defects.

The guarded automatic coverage is {number(augmented, "guarded_auto_coverage"):.2%} and
automatic selective risk is {number(augmented, "guarded_selective_risk"):.2%}.
Recapture is an unresolved workflow state. Capture statistics can also change
because of a defect, so rejected images must not be labelled normal or treated
as camera-only failures.

## Frozen stress conditions, augmented model

{shift_header}
|---|---:|---:|---:|---:|---:|---:|
{condition_table}

The experiment deliberately computes ungated neural scores for rejected inputs
to measure interception and missed-detection trade-offs. This is an offline
counterfactual evaluation. The live M8 interface still skips neural inference
when its quality gate fails. Recapture can reduce automatic false alerts by
withholding inputs; it does not repair image quality or prove shift invariance.

## Scope and reproducibility

This additional dataset contains one industrial surface category. Its full-frame
resize retains strip-end evidence while changing aspect ratio and reducing fine
defect detail. The source does not provide physical-item/session grouping, so
the audit establishes content-duplicate separation rather than item independence.
Bootstrap intervals resample images and may be optimistic under unobserved
production correlations. No test labels select parameters or thresholds.

The authoritative run is `{run.name}`. Model weights, source hashes, resolved
configuration, calibration arrays, memory indexes, per-image predictions, and
condition checkpoints remain in local artifact storage. The original VisA/M7
results remain frozen. Aggregate outputs and their manifest are in [m9/](m9/).

Attribution: Jakob Božič, Domen Tabernik, and Danijel Skočaj,
*Mixed supervision for surface-defect detection: from weakly to fully supervised
learning*, Computers in Industry, 2021.
"""
    report_path = project / "reports" / "M9_EXTERNAL_VALIDATION.md"
    _atomic_text(report_path, content)
    outputs = [output / "external_metrics.csv", report_path]
    output_hashes = {path.relative_to(project).as_posix(): sha256_file(path) for path in outputs}
    package_hash = hashlib.sha256(json.dumps(output_hashes, sort_keys=True).encode()).hexdigest()
    _atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "source_run": run.name,
            "source_metadata_sha256": sha256_file(metadata_path),
            "dataset_manifest_sha256": metadata["manifest_sha256"],
            "evaluation_config_sha256": metadata["config_sha256"],
            "source_code_sha256": metadata["code_sha256"],
            "output_hashes": output_hashes,
            "package_sha256": package_hash,
            "source_images_published": False,
        },
    )
    print(json.dumps({"report": str(report_path), "package_sha256": package_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
