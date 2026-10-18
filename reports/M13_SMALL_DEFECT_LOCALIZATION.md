# Small-defect localization: resolution and overlapping tiles

## Outcome and scope

Four overlapping detail tiles change clean VisA foreground IoU by
+0.0181 relative to the matched global
fine readout; small-image IoU changes by +0.0071.
KSDD2 changes by +0.0317 and
+0.1579, respectively. Full trade-offs,
including false positives, region sizes, shifts and cost, are retained below.
This is a completed exploratory comparison, not automatic deployment promotion.
The historical M10 inspector and M12 capture comparison remain unchanged.

The VisA small-image gain is uncertain (its paired category interval spans zero),
and tiny-region hit rate decreases despite improved overall clean overlap.
Severe-noise normal-image FPR increases by
+47.59%
on VisA and
+79.19%
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

| Dataset | Source anomaly images | Fine visible masks | Fine positives lost at 56 | Source-positive/fine-empty | Small images | Eligible small-image categories |
|---|---:|---:|---:|---:|---:|---:|
| visa | 1200 | 1189 | 31 | 11 | 221 | 9 |
| ksdd2 | 110 | 110 | 1 | 0 | 1 | 1 |

Small-image total mask area is <= .1% of the common frame. Region strata use
8-connected fine ground truth: tiny <= .01%, small > .01% and <= .1%, large > .1%.
`small_inclusive` combines tiny and small. These are relative image-area cohorts,
not physical defect dimensions. A region hit means >=10% pixel recall, not good
boundaries or a confident product-level detection. Always inspect FPR/IoU too.

## Clean operating points

VisA metrics are equal-category means; undefined cohorts are not filled with
zero. Eligible category/region/image counts remain explicit in the tables. KSDD2
is a separate single-domain result. Empty normal masks never raise foreground IoU.

| Dataset | Readout | Foreground IoU | Small-image IoU | Small-region hit | Normal image FPR | Normal pixel FPR | Normal contour incidence |
|---|---|---:|---:|---:|---:|---:|---:|
| visa | global56 | 0.1322 | 0.0271 | 45.58% | 4.05% | 0.51% | 76.34% |
| visa | global336 | 0.1338 | 0.0285 | 45.97% | 4.05% | 0.51% | 78.92% |
| visa | tiled336 | 0.1519 | 0.0356 | 46.81% | 3.80% | 0.47% | 88.98% |
| ksdd2 | global56 | 0.2916 | 0.0000 | 0.00% | 4.47% | 0.36% | 61.19% |
| ksdd2 | global336 | 0.2922 | 0.0000 | 0.00% | 4.47% | 0.36% | 64.99% |
| ksdd2 | tiled336 | 0.3238 | 0.1579 | 100.00% | 5.15% | 0.42% | 83.89% |

The following image-level context makes reduced field-of-view trade-offs visible.
Global56 is omitted here because its image inference is identical to global336.
Review is unresolved, not a correct automatic product decision.

| Dataset | Readout | Image AUROC | Image AP | Confident defect recall | Review rate | Automatic coverage |
|---|---|---:|---:|---:|---:|---:|
| visa | global336 | 0.8744 | 0.8953 | 54.08% | 19.68% | 80.32% |
| visa | tiled336 | 0.8931 | 0.9102 | 57.58% | 18.70% | 81.30% |
| ksdd2 | global336 | 0.9373 | 0.8466 | 79.09% | 11.06% | 88.94% |
| ksdd2 | tiled336 | 0.9024 | 0.7875 | 73.64% | 11.55% | 88.45% |

## Clean small-defect paired evidence

The same size cohorts apply to both readouts. Undefined cohorts remain undefined;
small-image and region eligibility counts are in the visibility/size tables.
Public image-bootstrap intervals are omitted with fewer than two eligible
images; category-bootstrap intervals are omitted with fewer than two eligible
categories. A single observed small defect cannot establish reliable uncertainty,
even when a conditional bootstrap of that same image gives a degenerate interval.
Points/counts remain visible and raw frozen-run estimates remain in local artifacts.

| Dataset | Contrast | Small-image IoU delta | 95% interval | Small-region hit delta | 95% interval |
|---|---|---:|---|---:|---|
| visa | resolution | +0.0015 | [+0.0002, +0.0029] | +0.39% | [-0.63%, +1.18%] |
| visa | tiling | +0.0071 | [-0.0015, +0.0191] | +0.84% | [-4.84%, +6.23%] |
| ksdd2 | resolution | +0.0000 | [—, —] | +0.00% | [—, —] |
| ksdd2 | tiling | +0.1579 | [—, —] | +100.00% | [—, —] |

## Paired changes and severe capture conditions

`resolution` is global336 minus global56; `tiling` is tiled336 minus global336.
Both use common fine ground truth. Each image bootstrap keeps all its regions
together; VisA macro intervals instead resample category differences. These are
conditional on the fitted models/calibration and exploratory, not simultaneous
or independent capture-session guarantees. Small-cohort finite-resample/category
counts are retained in CSVs. Image false positives are calibrated decisions, not
pixel support or quality recapture. The two global readouts have identical image
scores/p-values/decisions by construction and verified before publication.

| Dataset | Condition | Contrast | Foreground IoU delta | 95% interval | Small-image IoU delta | Normal FPR delta |
|---|---|---|---:|---|---:|---:|
| visa | clean | resolution | +0.0015 | [+0.0010, +0.0021] | +0.0015 | +0.00% |
| visa | clean | tiling | +0.0181 | [+0.0014, +0.0377] | +0.0071 | -0.25% |
| visa | blur_severe | resolution | +0.0002 | [+0.0001, +0.0003] | +0.0000 | +0.00% |
| visa | blur_severe | tiling | +0.0009 | [-0.0043, +0.0069] | +0.0002 | +14.75% |
| visa | noise_severe | resolution | +0.0006 | [+0.0002, +0.0012] | +0.0001 | +0.00% |
| visa | noise_severe | tiling | -0.0205 | [-0.0347, -0.0037] | -0.0025 | +47.59% |
| ksdd2 | clean | resolution | +0.0006 | [-0.0008, +0.0019] | +0.0000 | +0.00% |
| ksdd2 | clean | tiling | +0.0317 | [+0.0118, +0.0509] | +0.1579 | +0.67% |
| ksdd2 | blur_severe | resolution | -0.0000 | [-0.0003, +0.0002] | +0.0005 | +0.00% |
| ksdd2 | blur_severe | tiling | -0.0115 | [-0.0180, -0.0061] | -0.0074 | +25.95% |
| ksdd2 | noise_severe | resolution | +0.0000 | [-0.0007, +0.0008] | +0.0000 | +0.00% |
| ksdd2 | noise_severe | tiling | -0.1292 | [-0.1468, -0.1128] | +0.0031 | +79.19% |

Full per-condition readout tables additionally retain recall, Dice, precision,
review, coverage, normal contour incidence and background-pixel false positives.

## Clean region-size cohorts

| Dataset | Readout | Stratum | Regions | Eligible categories | Hit rate | Mean region pixel recall |
|---|---|---|---:|---:|---:|---:|
| visa | global56 | tiny | 731 | 12 | 42.51% | 39.08% |
| visa | global56 | small | 914 | 12 | 47.60% | 41.02% |
| visa | global56 | large | 1177 | 12 | 80.60% | 61.11% |
| visa | global56 | small_inclusive | 1645 | 12 | 45.58% | 40.11% |
| visa | global56 | all | 2822 | 12 | 63.02% | 51.36% |
| visa | global336 | tiny | 731 | 12 | 42.05% | 38.78% |
| visa | global336 | small | 914 | 12 | 48.38% | 41.52% |
| visa | global336 | large | 1177 | 12 | 80.13% | 61.56% |
| visa | global336 | small_inclusive | 1645 | 12 | 45.97% | 40.50% |
| visa | global336 | all | 2822 | 12 | 63.09% | 51.80% |
| visa | tiled336 | tiny | 731 | 12 | 40.51% | 37.03% |
| visa | tiled336 | small | 914 | 12 | 51.56% | 43.90% |
| visa | tiled336 | large | 1177 | 12 | 82.32% | 63.55% |
| visa | tiled336 | small_inclusive | 1645 | 12 | 46.81% | 40.61% |
| visa | tiled336 | all | 2822 | 12 | 64.40% | 52.78% |
| ksdd2 | global56 | tiny | 0 | 0 | — | — |
| ksdd2 | global56 | small | 1 | 1 | 0.00% | 0.00% |
| ksdd2 | global56 | large | 118 | 1 | 92.37% | 75.64% |
| ksdd2 | global56 | small_inclusive | 1 | 1 | 0.00% | 0.00% |
| ksdd2 | global56 | all | 119 | 1 | 91.60% | 75.01% |
| ksdd2 | global336 | tiny | 0 | 0 | — | — |
| ksdd2 | global336 | small | 1 | 1 | 0.00% | 0.00% |
| ksdd2 | global336 | large | 118 | 1 | 92.37% | 75.68% |
| ksdd2 | global336 | small_inclusive | 1 | 1 | 0.00% | 0.00% |
| ksdd2 | global336 | all | 119 | 1 | 91.60% | 75.05% |
| ksdd2 | tiled336 | tiny | 0 | 0 | — | — |
| ksdd2 | tiled336 | small | 1 | 1 | 100.00% | 100.00% |
| ksdd2 | tiled336 | large | 118 | 1 | 84.75% | 66.95% |
| ksdd2 | tiled336 | small_inclusive | 1 | 1 | 100.00% | 100.00% |
| ksdd2 | tiled336 | all | 119 | 1 | 84.87% | 67.22% |

Within each category region rates weight regions equally; VisA summary rates
weight eligible categories equally. Regions in one image are not independent
trials. No claim of significant small-region gain is made from unclustered counts.

## Resources and deployment boundary

| Dataset | Model | Encoder views | Single-image median ms | p95 ms | All indexes MB |
|---|---|---:|---:|---:|---:|
| visa | global | 1 | 40.1 | 49.0 | 2.75 |
| visa | tiled | 4 | 97.9 | 117.1 | 2.75 |
| ksdd2 | global | 1 | 24.1 | 27.8 | 0.23 |
| ksdd2 | tiled | 4 | 88.4 | 92.6 | 0.23 |

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

Authoritative source: `m13-small-defects-full-20261017T002537Z`. It contains 13 units, all 3,166 official test
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
