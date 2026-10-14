# M9 external validation protocol

Frozen before the full evaluation on 2026-10-13. M9 evaluates the existing method
on KolektorSDD2 using dataset-local training normals and calibration normals.
It tests recipe replication and robustness in a second domain. VisA product
memories, category thresholds, and capture references are not assumed to transfer
to an unrelated industrial surface.

## Fixed method

Use frozen ImageNet ResNet-18 layer-1/2/3 features, a seeded 64-dimensional
projection, a 4,096-candidate bound, a 512-entry k-center memory, and the highest
1% of patch distances for each image score. Compare two equal-budget models:

1. Plain memory: 24 clean normal patches per fitting image.
2. Augmented memory: 12 clean and 12 patches from one deterministic mild nuisance
   view per fitting image, using the frozen M4 augmentation pool.

Patch sampling is deterministic per sample. M9 caches sampled patches in batches
to bound memory usage; it uses the same representation, memory capacity,
candidate cap, and total patch budget as the previous method, but does not claim
byte-identical patch selection to the VisA experiment.

## Input and split boundary

- Preserve all 894 normal and 110 anomalous official test images.
- Exclude all provider training defects; only normals establish memory and gates.
- Derive the existing 15% validation fraction from whole normal content groups
  with seed 20260915: 1,772 fit images and 313 calibration images.
- Retain all raw source files; record the two exact provider copy exclusions.
- Resize the full image and mask together to 224×224 rather than cropping the
  center of the narrow strip. RGB uses bilinear; binary masks use nearest sampling.
- Smooth anomaly maps at 224×224 with sigma 4, then area-resize scores and
  nearest-resize masks to a declared 56×56 localization grid. Record any positive
  mask lost through raster downsampling rather than silently removing its image.

## Calibration and evaluation

- Anomaly conformal policy: anomalous `p <= 0.05`, review `0.05 < p <= 0.20`,
  normal `p > 0.20`, using local calibration normals only.
- Pixel support threshold: local validation-normal 99.5th percentile.
- Capture gate: unchanged M8 eight-feature robust reference and maximum-deviation
  conformal score, refitted/calibrated on local normals at `alpha = 0.02`.
- Conditions: clean plus all twelve frozen M4 shifts with unchanged seed,
  severities, parameters, and mild augmentation definitions.
- Report image AUROC/AP/FPR@95 and clean stratified 95% bootstrap intervals
  (500 replicates), pixel AUROC/AP/mean IoU, and AUPRO@30 (30 thresholds).
- Report conformal normal FPR, review, automatic coverage, selective risk,
  anomaly recall, and exact binomial intervals.
- Report quality rejection separately for normals and anomalies, false-alert
  interception, guarded automatic recall, automatic risk, and total actionable
  anomaly coverage including recapture and review.

For controlled comparison, the experiment computes both models' ungated scores
on every image, then applies the quality policy counterfactually. This permits
interception and missed-detection analysis. The live M8 prototype continues to
withhold neural inference on rejected captures.

## Provenance and interruption handling

Every run records the full resolved protocol, code fingerprint, environment,
Git state, source/config/manifest hashes, backbone checkpoint identity, normal
calibration arrays, quality model, memory indexes, per-image outcomes, and
aggregate metrics. Training artifacts and condition outputs have hash-verified
checkpoints. Resume verifies inputs and skips committed conditions.

Public exports contain only aggregate tables, this protocol, a research report,
and provenance hashes. Dataset pixels, per-image predictions, weights, and
feature memories remain in local D: storage. The M7 publication remains frozen.
