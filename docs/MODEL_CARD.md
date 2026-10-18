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
- Evaluation: preserved official test partition, 962 normal and 1,200 anomalous
  images across 12 categories.
- Leakage controls: exact and perceptual cross-split duplicate rejection through
  the trusted M1 manifest.

No anomalous test image is used to fit memory, select a threshold, or calibrate
the conformal policy.

## Decision policy

Before anomaly inference, the M8 configuration applies a category-conditional
capture-quality gate. It compares eight low-cost capture measurements with a
robust training-normal reference and calibrates the maximum directional
deviation using validation normals. `quality p <= 0.02` returns `recapture` and
skips the backbone, memory search, anomaly decision, and heatmap.

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

M11 compared a frozen ConvNeXt-Tiny substitution under matched normal samples,
memory capacity, calibration, and geometry. Clean VisA image AUROC falls from
0.8668 to 0.8344 and KSDD2 from 0.9354 to 0.8178; clean foreground IoU also
falls on both domains. The larger encoder is not deployed. Its lower shifted
KSDD2 false-alarm rate comes with lower confident defect recall, not a uniform
improvement. This is evidence about the common first-three-stage/64-dimensional
memory recipe, not a best-tuned ConvNeXt or every modern encoder. The declared
56x56 grid loses 73 clean VisA and one KSDD2 positive masks; their image detection
results remain included. See [M11 comparison](../reports/M11_ENCODER_COMPARISON.md).
The normal fit cap/grid differ from historical deployment studies, so these are
matched-study results rather than changed M5/M10 production metrics.

M10 optionally removes sub-16-pixel isolated components from the original VisA
224x224 thresholded support, without changing continuous evidence or image
decisions. Clean normal-image contour incidence drops from 78.17% to 73.43%,
while foreground IoU changes only from 0.1194 to 0.1196; the exploratory macro
interval includes zero. Small-defect recall can decrease. KSDD2 cleanup is
unchanged at its 56x56 grid. Spatial normalization/growth lost foreground
overlap and are not deployed. See [M10 localization](../reports/M10_LOCALIZATION.md).
This finding also distinguishes binary defect overlap from previously reported
all-image IoU, which can reward empty-normal masks.

Independent KolektorSDD2 validation of the same model recipe, with normal-only
local memories/calibration, achieves clean image AUROC 0.9393 and normal FPR
2.68%. However, its locally calibrated capture gate intercepts only 21.90% of
shifted false alerts and often misses blur/geometry failures. The gate's VisA
result must not be generalized to other product or capture domains. See
[M9 external validation](../reports/M9_EXTERNAL_VALIDATION.md) for the complete
paired-model and operating-point results and the different full-frame geometry.

- Conformal false-alarm validity is marginal and requires exchangeability with
  clean calibration normals. It is not a guarantee under capture shift.
- The capture gate is an out-of-profile screen, not proof that an accepted image
  is correctly framed or that a rejected image contains no product defect.
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

The quality-gated prototype additionally verifies the M8 quality-model hash,
manifest identity, M5 source identity, input geometry, and category coverage.
It then verifies the trusted manifest, M5 calibration outputs, M4 metadata, the
M4 index manifest, every category index, and the ResNet-18 checkpoint before
serving. The authoritative anomaly calibration run is
`m5-calibration-full-20260920T110316Z`. See `reports/M5_CALIBRATION.md` for the
full operating analysis.

The optional M12 comparison uses schema-2 capture references: the eight original
measurements plus contrast-normalized log focus and low-structure noise proxies.
Its alpha remains .02 and all references/calibration are normal-only. The
historical M8/M10 gate is unchanged; this is not an automatic deployment upgrade
or a change to detector weights/thresholds. Read the complete
[gate trade-offs](../reports/M12_QUALITY_GENERALIZATION.md), especially defect
deferral, clean recapture, and operator burden, before choosing a configuration.
This follow-up uses known benchmarks and is exploratory, not untouched capture
confirmation. Recapture never means that a defect was successfully detected.

The M10 configuration also verifies the source M5 identity, summary, every
category localization profile hash, memory identity, manifest, and center-crop
geometry before serving. The evaluated support function is shared with live
contours and flagged-area reporting. M6/M8 configurations preserve their
historical contour behavior.

M13 is a separate small-defect resolution/tiling study, not a deployment change.
It compares common-grid measurement effects with actual four-view detail under
matched normal-memory budgets and independent normal-only calibration. Fine
positive-coverage masks avoid silently dropping coarse-grid-lost defects but
can expand subpixel boundaries. Preserving annotation visibility does not add
encoder information or establish physical defect precision. Region hits must
be read alongside image/foreground overlap, false positives and compute cost.
See the [complete study](../reports/M13_SMALL_DEFECT_LOCALIZATION.md); the live
M10 model, gate and thresholds remain unchanged.

Clean tiling overlap gains do not establish reliable tiny-defect improvement:
the VisA small-image IoU interval spans zero, tiny-region hit rate declines,
and KSDD2 has only one small-defect image. Severe-noise false positives and
single-image cost increase substantially. Keep the current deployment default.
