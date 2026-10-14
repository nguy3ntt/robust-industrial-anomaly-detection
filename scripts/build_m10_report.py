"""Reconstruct aggregate localization evidence from an exact completed-run pin."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.experiments.m2 import _atomic_csv, _atomic_json, _read_csv
from robust_industrial_anomaly_detection.reporting.m7 import _atomic_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild hash-pinned M10 localization evidence")
    parser.add_argument("--config", type=Path, default=Path("configs/reporting/m10.yaml"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / args.config).read_text("utf-8"))
    run = Path(config["source_run"])
    metadata_path = run / "metadata.json"
    if sha256_file(metadata_path) != config["metadata_sha256"]:
        raise ValueError("localization source metadata differs from the publication pin")
    metadata = json.loads(metadata_path.read_text("utf-8"))
    if metadata.get("status") != "complete" or metadata.get("mode") != "full":
        raise ValueError("localization publication requires a complete full run")
    for filename, expected in metadata["output_hashes"].items():
        path = (run / filename).resolve(strict=True)
        path.relative_to(run.resolve(strict=True))
        if sha256_file(path) != expected:
            raise ValueError(f"modified localization artifact: {filename}")
    rows: list[dict[str, Any]] = list(_read_csv(run / "metrics.csv"))
    summary = json.loads((run / "summary.json").read_text("utf-8"))
    if len(rows) != 325 or summary["units"] != 13 or summary["primary_policy"] != "raw_cleaned":
        raise ValueError(
            "public localization run must cover thirteen units, five conditions, five policies"
        )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["condition"], row["policy"])].append(row)
    metric_keys = (
        "foreground_iou",
        "foreground_dice",
        "foreground_precision",
        "foreground_recall",
        "normal_pixel_fpr",
        "normal_image_support_rate",
        "all_image_iou",
        "background_pixel_fpr",
    )
    macros = []
    for (dataset, condition, policy), group in sorted(groups.items()):
        macro = {
            "dataset": dataset,
            "condition": condition,
            "policy": policy,
            "categories": len(group),
        }
        macro.update(
            {key: float(np.mean([float(row[key]) for row in group])) for key in metric_keys}
        )
        if policy == "raw_cleaned":
            if dataset == "visa":
                deltas = np.asarray([float(row["paired_foreground_iou_delta"]) for row in group])
                rng = np.random.default_rng(20260915)
                means = np.mean(rng.choice(deltas, (1000, len(deltas)), replace=True), axis=1)
                low, high = np.quantile(means, [0.025, 0.975])
                macro.update(
                    {
                        "delta": float(np.mean(deltas)),
                        "delta_ci_low": float(low),
                        "delta_ci_high": float(high),
                        "bootstrap_unit": "category",
                    }
                )
            else:
                macro.update(
                    {
                        "delta": group[0]["paired_foreground_iou_delta"],
                        "delta_ci_low": group[0]["paired_delta_ci_low"],
                        "delta_ci_high": group[0]["paired_delta_ci_high"],
                        "bootstrap_unit": "visible-anomaly-image",
                    }
                )
        macros.append(macro)
    output = (project / config["output_directory"]).resolve()
    output.relative_to((project / "reports").resolve())
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output / "category_metrics.csv", rows)
    _atomic_csv(output / "macro_metrics.csv", macros)
    index = {(row["dataset"], row["condition"], row["policy"]): row for row in macros}
    names = {
        "baseline_global": "Existing global threshold",
        "raw_cleaned": "Existing threshold + cleanup",
        "raw_hysteresis": "Raw seeded growth",
        "spatial_global": "Spatial seed + cleanup",
        "spatial_hysteresis": "Spatial seeded growth",
    }
    lines = []
    for dataset in ("visa", "ksdd2"):
        for policy, label in names.items():
            row = index[(dataset, "clean", policy)]
            lines.append(
                f"| {dataset} | {label} | {row['foreground_iou']:.4f} "
                f"| {row['foreground_dice']:.4f} | {row['foreground_precision']:.4f} "
                f"| {row['foreground_recall']:.4f} | {row['normal_pixel_fpr']:.2%} "
                f"| {row['all_image_iou']:.4f} |"
            )
    clean_table = "\n".join(lines)
    lines = []
    for dataset, condition in sorted({(row["dataset"], row["condition"]) for row in macros}):
        old = index[(dataset, condition, "baseline_global")]
        new = index[(dataset, condition, "raw_cleaned")]
        lines.append(
            f"| {dataset} | {condition} | {old['foreground_iou']:.4f} "
            f"| {new['foreground_iou']:.4f} | {old['normal_pixel_fpr']:.2%} "
            f"| {new['normal_pixel_fpr']:.2%} |"
        )
    stress_table = "\n".join(lines)
    lines = []
    for row in rows:
        if row["condition"] == "clean" and row["policy"] == "raw_cleaned":
            old = next(
                item
                for item in rows
                if item["condition"] == "clean"
                and item["dataset"] == row["dataset"]
                and item["category"] == row["category"]
                and item["policy"] == "baseline_global"
            )
            lines.append(
                f"| {row['dataset']}/{row['category']} | {old['foreground_iou']:.4f} "
                f"| {row['foreground_iou']:.4f} | {row['paired_foreground_iou_delta']:+.4f} "
                f"| [{row['paired_delta_ci_low']:+.4f}, {row['paired_delta_ci_high']:+.4f}] |"
            )
    category_table = "\n".join(lines)
    visa = index[("visa", "clean", "raw_cleaned")]
    visa_old = index[("visa", "clean", "baseline_global")]
    ksdd = index[("ksdd2", "clean", "raw_cleaned")]
    ksdd_old = index[("ksdd2", "clean", "baseline_global")]
    clean_header = (
        "| Dataset | Policy | Defect IoU | Dice | Precision | Recall | "
        "Normal pixel FPR | All-image IoU |"
    )
    stress_header = (
        "| Dataset | Condition | Existing defect IoU | Cleaned defect IoU | "
        "Existing normal pixel FPR | Cleaned normal pixel FPR |"
    )
    content = f"""# M10 thresholded localization

## Outcome

Removal of small isolated regions at the existing normal-calibrated threshold changes VisA
macro visible-defect IoU from {visa_old["foreground_iou"]:.4f} to
{visa["foreground_iou"]:.4f}, a paired category-mean difference of
{visa["delta"]:+.4f} (95% category-bootstrap interval
[{visa["delta_ci_low"]:+.4f}, {visa["delta_ci_high"]:+.4f}]). Clean normal pixel
false positives change from {visa_old["normal_pixel_fpr"]:.3%} to
{visa["normal_pixel_fpr"]:.3%}. The macro fraction of clean normal images with
any outlined region drops from {visa_old["normal_image_support_rate"]:.2%} to
{visa["normal_image_support_rate"]:.2%}; this is contour incidence, not the
image-level anomaly false-alarm rate. Foreground recall changes from
{visa_old["foreground_recall"]:.4f} to {visa["foreground_recall"]:.4f}, including
small category-specific losses. On KolektorSDD2, visible-defect IoU changes from
{ksdd_old["foreground_iou"]:.4f} to {ksdd["foreground_iou"]:.4f}, with paired
image-bootstrap difference interval [{ksdd["delta_ci_low"]:+.4f}, {ksdd["delta_ci_high"]:+.4f}].

These are measured binary contour results, separate from image-level detection
and continuous-map ranking. Image anomaly scores, M5 decisions, memory banks,
backbone weights, and the M8 capture gate remain unchanged.

## Fixed method and calibration

Up to 256 deterministic training normals estimate per-location score median and
MAD variation. A robust scale floor prevents unstable division at quiet pixels.
Separate validation normals establish 99th-percentile weak and 99.5th-percentile
strong boundaries. A monotone search can raise the strong threshold until the
region-growth calibration respects a 0.5% mean normal-pixel support budget.
The primary contour keeps the original score map and calibrated threshold and removes
regions smaller than 0.03% of the map. Spatial normalization and weak-evidence growth are separate
comparisons; growth is limited to half one feature-lattice cell (8 pixels at 224x224,
2 at 56x56). Numerical calibration never uses defect masks.

The initial engineering smoke prototype allowed unbounded growth. A weak bridge
could spread its region far from strong evidence, so the full protocol adds an
architecture-derived radius bound. The primary policy retains the existing evidence boundary;
spatial normalization and growth remain measured comparisons. The full results are comparative
evidence after development, not a fresh blind industrial acceptance test.

See the [frozen protocol](../docs/M10_LOCALIZATION_PROTOCOL.md) and
`configs/experiments/m10-localization.yaml` for the exact choices.

## Clean ablation

{clean_header}
|---|---|---:|---:|---:|---:|---:|---:|
{clean_table}

Defect IoU/Dice/precision/recall are means over anomalous images with visible
positive masks. Empty normal masks never earn credit in these metrics. The
legacy all-image IoU remains separate so normal-empty success cannot conceal
defect-localization failures. VisA uses 224x224 centered geometry; KolektorSDD2
retains M9's full-frame input and 56x56 localization grid. Compare policies
within a dataset, not absolute localization values across these grids.

## Category behavior

| Category | Existing defect IoU | Cleaned defect IoU | Paired delta | 95% paired interval |
|---|---:|---:|---:|---|
{category_table}

Category gains and losses remain visible. Small defects, object displacement,
or spatially variable normal backgrounds can change precision and recall in
different directions. A contour is unusual spatial evidence, not a physical
defect boundary or root-cause diagnosis.

## Severe capture shifts

{stress_header}
|---|---|---:|---:|---:|---:|
{stress_table}

Clean calibration does not guarantee shifted control. Spatial references assume
stable capture geometry and may degrade under displacement or image corruption.
Offline support is measured for every official test image to reveal this
failure boundary, including captures the live gate would reject.

## Runtime and reproducibility

The M10 configuration displays contours from the exact evaluated support
function and includes memory-distance threshold units, region count, policy,
and flagged area in metadata-only exports. Startup verifies all spatial
artifact hashes, original memory identity, trusted manifest, and input geometry.
The M6/M8 configurations retain their original pixel policy for reproduction.

The authoritative run is `{run.name}`. It covers 13 dataset/category units,
five conditions, and five policies, with 325 aggregate rows. All source images,
per-image support measurements, model weights, spatial numeric artifacts, and
feature memories remain in local artifact storage. Public aggregate tables and
hashes are under [m10/](m10/). Checkpointed training references and conditions
permit verified resume. Provider splits are unchanged.

The spatial alternatives were retained for comparison after their development
results showed worse visible-defect overlap in several categories. The deployment
uses component cleanup at the original threshold. This is a measured engineering
refinement rather than a claim that spatial normalization is a superior detector.
The minimum component area rounds to one pixel on KSDD2's 56x56 grid, so cleanup
does not change its baseline support. Removing more would require a different,
independently justified small-defect policy. The VisA change is modest, not a
substantial new localization accuracy gain. Intervals after development/policy
selection are exploratory, not independent confirmation.
"""
    report = project / "reports" / "M10_LOCALIZATION.md"
    _atomic_text(report, content)
    files = [report, output / "category_metrics.csv", output / "macro_metrics.csv"]
    hashes = {path.relative_to(project).as_posix(): sha256_file(path) for path in files}
    package_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    _atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "source_run": run.name,
            "source_metadata_sha256": sha256_file(metadata_path),
            "output_hashes": hashes,
            "package_sha256": package_hash,
            "source_images_published": False,
        },
    )
    print(json.dumps({"report": str(report), "package_sha256": package_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
