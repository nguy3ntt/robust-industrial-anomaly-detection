# Research specification

## Problem statement

Industrial anomaly detection must often learn from normal examples because
defects are rare, diverse, and expensive to label. A useful system must find
unseen visible deviations, localize them, control nuisance false alarms, and
remain interpretable when acquisition conditions differ from the benchmark.

This project studies normal-only visual anomaly detection on VisA and asks
whether robust representations plus normal-score calibration can improve the
reliability of image-level detection and pixel-level localization under shift.

## Scope

The initial scope is RGB still images, the 12 VisA categories, image-level
anomaly ranking, pixel-level localization, deterministic image perturbations,
and a local single-image/batch demonstration. It includes category-specific and
unified modelling and an explicit human-review outcome.

The scope excludes autonomous production-line control, causal defect diagnosis,
safety certification, private factory data, and claims beyond the evaluated
object/capture distribution.

## Hypotheses

- H1: Patch-level pretrained features materially outperform global statistical
  baselines for both image ranking and localization.
- H2: Multi-scale normal representations reduce robustness loss under resizing,
  blur, and small geometric changes.
- H3: Normal-only quantile or conformal score calibration controls the normal
  false-positive rate more reliably than a pooled fixed score threshold.
- H4: A selective review band reduces confident errors at useful coverage.
- H5: Explanation stability predicts failure under benign transformations and
  provides information beyond unshifted localization accuracy.

## Experimental controls

- Preserve the provider's official one-class test partition.
- Derive validation/calibration only from official training-normal images.
- Freeze all preprocessing, models, thresholds, and perturbation severities
  before final test evaluation.
- Compare every proposed component against the nearest simpler baseline.
- Report category-level distributions and confidence intervals.
- Record code revision, environment, seed, manifest hash, and configuration for
  every reportable run.

## Planned baselines

1. Colour/texture summary features with distance to the normal training set.
2. Frozen pretrained global embeddings with nearest-neighbour scoring.
3. Frozen patch embeddings with a coreset memory bank and nearest-neighbour
   anomaly maps.

Only after these baselines are reproduced will the project test multi-scale
features, representation adaptation, or learned fusion.

## Metrics

Image-level: AUROC, average precision, FPR at 95% recall, balanced accuracy and
F1 at a frozen operating point. Pixel-level: AUROC, average precision,
intersection-over-union at a frozen threshold, and region-aware overlap. Trust
metrics: normal false-positive rate, coverage-risk curves, review rate, and
bootstrap intervals. Efficiency: latency, throughput, peak memory, model size,
and memory-bank size.

## Robustness protocol

The clean test set is evaluated together with deterministic severity ladders
for JPEG compression, downscale/upscale, Gaussian blur, sensor-like noise,
brightness/contrast shifts, and small rotations/translations. Transformations
must apply consistently to images and masks. The report includes absolute
performance, change from clean, ranking stability, heatmap stability, and
failure examples.

## Completion claim

The strongest acceptable conclusion is evidence about benchmark performance
and controlled shifts. The project will not claim general industrial readiness.
