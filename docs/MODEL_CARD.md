# Model card — augmented multi-scale coreset detector

## Model summary

The deployed research model is `multiscale_augmented_coreset_512`, a normal-only
visual anomaly detector and localizer. A frozen ImageNet ResNet-18 produces
layer-1, layer-2, and layer-3 spatial features. Features are normalized,
projected to 64 dimensions, and scored by nearest-neighbour distance to a
category-specific 512-entry greedy k-center memory.

Each normal training image contributes an equal total patch budget split between
its clean view and one deterministic mild nuisance augmentation. The intervention
was selected through the controlled M4 ablation, not through test-defect tuning.

## Intended use

- Research and portfolio demonstration of normal-only industrial inspection.
- Offline or local review of individual images and small batches from one known
  VisA-compatible product category.
- Decision support in which uncertain samples are reviewed by a person.

The model is not a certified quality-control device, safety system, or physical
root-cause diagnostic. It must not autonomously release or reject products.

## Training and calibration data

- Dataset: VisA 2022-09-22, CC BY 4.0.
- Training: official one-class training normals only.
- Calibration: deterministic per-category validation normals derived only from
  training-normal rows; 68–136 images per category and 1,301 total.
- Evaluation: untouched official test partition, 962 normal and 1,200 anomalous
  images across 12 categories.
- Leakage controls: exact and perceptual cross-split duplicate rejection through
  the trusted M1 manifest.

No anomalous test image is used to fit memory, select a threshold, or calibrate
the conformal policy.

## Decision policy

Larger nearest-memory scores are more anomalous. Category-specific upper-tail
split-conformal p-values use conservative tie handling and a plus-one correction:

- anomalous: `p <= 0.05`;
- review: `0.05 < p <= 0.20`;
- normal: `p > 0.20`.

The interface also displays the empirical 95th-percentile normal-score reference
and the frozen pixel threshold used to outline heatmap support.

## Measured performance

On the clean official test partition, category-conformal macro operating results
are normal FPR 0.0430, review rate 0.1958, automatic coverage 0.8042, and
selective risk 0.1940. Category calibration improves clean false-alarm alignment
over pooled calibration by 0.0681 absolute error, 95% CI [0.00495, 0.18088].

The underlying clean detector has macro image AUROC 0.8674, image AP 0.8907,
pixel AUROC 0.9557, pixel AP 0.3226, and AUPRO@30 0.7762 on the corrected spatial
evaluation.

The 0.05–0.20 review band lowers automatic-decision risk by 0.07125, 95% CI
[0.06063, 0.08122], at 0.7790 mean coverage across clean and shifted conditions.

## Known limitations

- Conformal false-alarm validity is marginal and requires exchangeability with
  clean calibration normals. It is not a guarantee under capture shift.
- At severe shift, normal FPR reaches 0.8686 for blur, 0.6820 for noise, 0.4696
  for JPEG, and 0.2179 for geometry.
- Category identity must be known. Current zero-shot category-transfer image
  AUROC is only 0.5783.
- Heatmaps identify unusual spatial evidence but may be unstable under geometry
  and do not name the physical defect mechanism.
- Category performance varies; pooled metrics must not replace category results.
- VisA is a research benchmark and does not represent every production camera,
  material, defect, or process.

## Reproducibility and provenance

The prototype verifies the trusted manifest, M5 calibration outputs, M4 metadata,
the M4 index manifest, every category index, and the ResNet-18 checkpoint before
serving. The authoritative calibration run is
`m5-calibration-full-20260920T110316Z`. See `reports/M5_CALIBRATION.md` for the
full operating analysis.
