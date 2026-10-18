# Manual checks: small-defect resolution and overlapping tiles

The completed study uses the existing local data and frozen ResNet weights.
You do not need a new account, download, dependency, or training run to inspect
its results. The current browser inspectors are intentionally unchanged.

## 1. Read the results before choosing a model

Open `reports/M13_SMALL_DEFECT_LOCALIZATION.md`. Check these separately:

- `global56` -> `global336`: measurement/readout resolution only. The encoder,
  memory, image score, p-value and image decision are identical.
- `global336` -> `tiled336`: actual four-view detail/context trade-off, with the
  same original normal fit IDs and total normal-memory/patch budget.
- Small/tiny regions: read cohort counts, image-clustered intervals, foreground
  IoU, region recall and false positives together. A 10% region hit is not an
  accurate defect outline or a confident anomaly decision.
- Severe blur/noise and single-image processing time: clean overlap alone is
  not a reason to replace the current inspector.

An omitted interval on a tiny cohort is not a zero effect: fewer than two
eligible images/categories cannot support a useful bootstrap interval. A point
result of 100% on one observed small region does not establish reliable recall.

All readouts use the same fine ground truth for primary metrics. An anomaly
with a vanished coarse mask remains included. `coarse_lost_positive_masks`
counts raster losses; `source_positive_fine_empty` separately exposes crop loss.
The conservative fine masks can expand subpixel boundaries, so do not interpret
display pixels as native annotation precision or physical defect dimensions.

## 2. Create local side-by-side panels

Already generated and visually checked on this workstation:

- `artifacts/manual/m13-visa-candle-20261017T005715Z/`
- `artifacts/manual/m13-ksdd2-surface-20261017T005721Z/`

Both are labelled `full study`. The candle smallest anomaly has four fine GT
pixels: global336 reports normal (IoU .0155), tiled336 review (IoU .0046).
This is a visible regression, not a filtered success. The KSDD2 smallest anomaly
has 72 fine GT pixels: both globals miss its support, tiled IoU is .1579, but
its separate image decision is still normal. It is the only KSDD2 small image.
The normal candle has false contours even though its image decision is normal.

From the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe scripts/prepare_m13_manual_examples.py --dataset visa --category candle
.venv\Scripts\python.exe scripts/prepare_m13_manual_examples.py --dataset ksdd2
```

Each command prints an ignored folder under `artifacts/manual/` containing:

- `normal.png`: one fixed normal test image, including any false contours.
- `smallest-anomaly.png`: the smallest visible positive mask in the evaluated
  category, chosen by ground truth, not by a favorable prediction.
- `expected.json`: exact source-run identity, mask/ROI measurements, decisions,
  pixel thresholds and expected metrics.

Each panel has the same five columns: common input, ground truth, global56,
global336 and tiled336. The lower row zooms the SAME region in every column.
Red outlines are exact evaluated support; heat color is score relative to each
method's own frozen threshold, not an assertion that raw distances are equal.
The generator checks flagged-pixel counts and IoU against the saved predictions
before saving the panels. No new neural inference or threshold selection occurs.

Look for both retained tiny positives and excessive area around them. A large
red region touching a tiny defect can have high region hit/recall but poor IoU.
On the normal case, every red outline is a false spatial flag even if the
separate image decision is normal. Source labels and model decisions need not
agree; these panels are not filtered to hide failures.

These are licensed local dataset images, not public report assets. Do not add
the generated PNGs, private predictions, arrays, indexes, or source images to Git.
Do not upload them to an external service. The browser's current 224-input model
uses different memories/score aggregation; uploading panel screenshots there
would not reproduce this study.

## 3. Rebuild the exact public evidence

```powershell
.venv\Scripts\python.exe scripts/build_m13_report.py
```

Run it twice: package hashes should match. The builder refuses changed source
files, wrong metadata pins, smoke/incomplete/subset runs, missing official test
coverage, mismatched fit/calibration cohorts, or changed common ground truth.
Public output under `reports/m13/` contains only aggregate CSVs and hashes.
The verified final package SHA-256 is
`d7d814db35ba89ac5262b4c9248e454969833509876134955ad3b410856844d7`.

For an independent check of every original normal fit/calibration ID and every
official test ID/label in all three readouts and conditions:

```powershell
.venv\Scripts\python.exe scripts/audit_m13_run.py
# Optional full pixel-file integrity check; still no inference or training:
.venv\Scripts\python.exe scripts/audit_m13_run.py --verify-images
```

Expect 13 units, 3,166 official test images and 28,494 readout-condition
predictions. The code-fingerprint field may become false after later source
development; the frozen source/output hashes still identify this completed
study. Changed data membership or output artifacts are errors, not warnings.

## 4. Verify the software checks

```powershell
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\ruff.exe check .
.venv\Scripts\mypy.exe src scripts
.venv\Scripts\pytest.exe -q -p no:cacheprovider
```

The tests cover frame-edge coverage, odd dimensions, constant/coordinate map
stitching, non-finite input rejection, one-pixel positive preservation, crop vs
full-frame geometry, common size cohorts, empty-mask denominators, paired
identity checks, original-image splitting and spatial shift units.

## 5. Optional: a small real-data stop/resume check

```powershell
.venv\Scripts\python.exe scripts/run_m13_small_defects.py --quick --dataset visa --category candle --max-new-conditions 1
# Copy the exact run directory printed above:
.venv\Scripts\python.exe scripts/run_m13_small_defects.py --quick --dataset visa --category candle --resume "D:\industrial-anomaly-artifacts\runs\<printed-smoke-run>"
```

The first command commits clean results and reports `running`. Resume verifies
and reuses both fitted memories/calibrations and the clean checkpoint, then
finishes blur/noise. Source/config/environment or consumed data changes must
refuse reuse. Smoke/subset results are engineering checks, never full evidence.
Trusted source verification and complete normal calibration can take time.

For local smoke panel debugging only, the panel tool accepts `--run <completed
smoke-directory>` and clearly labels it `smoke`. The publication builder never
accepts such a run. To rerun the full fixed study, omit `--quick`, dataset and
category filters; do not change tile sizes or thresholds based on test results.
