# Verify the modern encoder comparison

M11 is an offline research comparison. It does not replace the local inspector,
alter its thresholds, or add an uncalibrated model selector to the website.
All data and numeric model artifacts remain on local storage outside Git.

## Quick checks

From the repository directory, use the existing environment:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_m11_encoders.py -q -p no:cacheprovider
.venv\Scripts\python.exe scripts/prepare_m11_encoders.py
```

The focused tests check feature alignment, normal-only/disjoint sample selection,
paired comparison calculations, full checkpoint hashing, changed-output refusal,
and incomplete/smoke publication refusal. Weight preparation checks the local
ResNet-18 and ConvNeXt-Tiny files; it only downloads an absent official file.
It refuses an existing file with the wrong hash instead of silently replacing it.
No account or credentials are needed. The one-time ConvNeXt download is about
109 MiB; there are no new Python dependencies.

## Reproduce a small real-data comparison

```powershell
.venv\Scripts\python.exe scripts/run_m11_encoder_comparison.py `
  --quick --dataset visa --category candle
```

Expect a completed smoke run on D: with ten metric rows: two encoders times
clean/JPEG/blur/noise/geometry. It uses 24 fit normals, 50 calibration normals,
and 24 test images. Smoke numbers are diagnostic subsets, not published full
results. It does not change the authoritative report pin. Dataset integrity
checking may take longer than the small neural workload on a cold disk.

To inspect the saved results, use the exact directory printed by the command:

```powershell
Import-Csv 'D:\industrial-anomaly-artifacts\runs\<printed-run-name>\metrics.csv' |
  Select-Object dataset, category, encoder, condition, image_auroc, normal_fpr, foreground_iou
```

The masks/evidence use a declared common 56x56 grid. Foreground IoU excludes
undefined empty masks; normal empty masks must not receive perfect-defect credit.
See `empty_anomaly_masks` and `visible_anomaly_samples` alongside the metric.
Do not compare these VisA IoUs directly with the M10 224x224 evaluation.

## Verify the complete publication

```powershell
.venv\Scripts\python.exe scripts/build_m11_report.py
```

The builder reads `configs/reporting/m11.yaml`, verifies the exact completed
source metadata and every committed output hash, and regenerates only public
aggregates. Running it twice should print the same package SHA-256. It refuses
a smoke run, changed source, incomplete coverage, duplicate units, or missing
official-test images.

Check [the report](../reports/M11_ENCODER_COMPARISON.md) and `reports/m11/`:

1. `category_metrics.csv` must have 130 data rows, covering both encoders,
   twelve VisA categories plus KSDD2, and all five declared conditions.
2. `paired_comparisons.csv` must have 65 rows, with ConvNeXt-minus-ResNet
   differences and paired intervals. Gains and losses both remain visible.
3. Compare **normal_fpr**, **anomaly_confident_recall**, and foreground IoU
   as well as image AUROC. A newer encoder is not automatically a better
   operating policy, nor a guarantee of good defect contours.
4. `resources.csv` must report checkpoint/index size, active parameter count,
   and matched single-image median/p95 latency. The batch timing fields in
   category tables are a different measurement. Process memory is descriptive,
   not an isolated model-only RAM measurement.
5. The manifest must state that source images and weights are not published.
   Public files must not contain image arrays, per-image IDs/predictions, or banks.

The completed comparison retains ResNet in the live system: VisA clean AUROC is
0.8668 versus ConvNeXt's 0.8344; KSDD2 is 0.9354 versus 0.8178. Some shifted
false-alarm and localization outcomes improve, but confident defect recall can
drop sharply. `severe_operating_summary.csv` makes that trade-off explicit.
`mask_visibility.csv` counts masks lost at the declared geometry/grid; 73 VisA
and one KSDD2 clean positive masks have no evaluated positive pixels. Their
images remain included in detection. These are not full-resolution contour claims.

## Checkpoint/resume verification

The following intentionally stops after two committed conditions, leaving an
incomplete smoke run:

```powershell
.venv\Scripts\python.exe scripts/run_m11_encoder_comparison.py `
  --quick --dataset visa --category candle --max-new-conditions 2
```

Copy the printed resume command, using `.venv\Scripts\python.exe` in place of
`python` if the environment is not activated. Do not include
`--max-new-conditions` when completing the run. The exact config, device, filters,
code, environment, datasets, and weights must match. Previously committed
conditions are hash-verified and reused; a partially written uncommitted
condition is recomputed. Do not mutate the authoritative artifacts to test this
manually; the automated tests already exercise tamper refusal.

For a fresh full study, omit `--quick` and the category/dataset filters. Normal
fitting remains capped at 256 images per category **for both encoders**, while
all calibration normals and official test images are used. This is a matched
normal-budget comparison, not full historical M5/M9 retraining. Complete the
full run before creating a new publication pin; do not change the existing pin
just because another run directory is newer.

The website remains on its previously verified M10 encoder/policy. To check it
independently, follow [M10 manual testing](M10_MANUAL_TESTING.md). M11 research
results do not silently change live product decisions or capture rejection.

The authoritative completed source is
`m11-encoder-comparison-full-20261015T005134Z`. Two final report builds produced
the identical package SHA-256
`9d62d190e57010604609141c43620817bc5b5c19c44d793793c4bf8ce2e3481b`.
An independent audit verified all 31,660 predictions against official test IDs
and labels, the normal-only fitting/calibration boundaries, and the evaluated
source-code fingerprint.
