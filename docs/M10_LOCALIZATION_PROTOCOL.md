# M10 thresholded localization protocol

M10 compares spatial normal references, bounded seeded growth, and component
cleanup on the frozen augmented-coreset detector. The deployed refinement removes
small isolated components at the original normal-calibrated threshold. Detector,
image conformal calibration, and capture gate are unchanged.

## Reference and calibration

Use up to 256 deterministically ranked training normals per category to estimate
each score-map pixel's median and MAD scale. Clamp scale to at least 10% of the
median spatial MAD and an absolute 1e-5 floor. Fit/reference samples and all
calibration images are disjoint under the trusted provider-derived partitions.

Calculate standardized evidence `(score - normal_median) / normal_scale` and
derive pooled 99th-percentile growth and 99.5th-percentile seed boundaries from
the existing validation normals only. The seed boundary can rise through a
monotone search until the mean flagged validation-normal pixel fraction is at
most 0.5%, including region-growth and island-removal effects.

The spatial seed-only policy uses its own 99.5th-percentile boundary. It does
not inherit the higher seed threshold calibrated for weak-evidence growth.
This keeps the comparisons at the intended normal-pixel operating budget.

Retain only 8-connected low-threshold regions containing a high-threshold seed.
Remove regions smaller than 0.03% of the evaluation grid (16 pixels at 224x224).
Growth is limited to half of one cell of the frozen 14x14 feature lattice:
8 pixels at 224x224 and 2 pixels at 56x56. This prevents weak connected bridges
from spreading a contour across an entire object. No hole filling is performed.

The budget is an empirical operating constraint on correlated calibration pixels,
not a per-pixel conformal guarantee or a shifted false-positive guarantee.

## Fixed comparisons

1. `baseline_global`: frozen source-model pixel threshold.
2. `raw_cleaned`: original threshold and small-component removal (primary).
3. `raw_hysteresis`: bounded seeded growth on raw distance with normal calibration.
4. `spatial_global`: spatial seed evidence with island removal.
5. `spatial_hysteresis`: spatial evidence plus bounded seeded growth.

The initial engineering smoke prototype used unbounded connected growth. Its
behavior and a synthetic weak-bridge case motivated the architecture-derived
growth bound before the full reportable evaluation. That prototype is retained
only as a local development artifact. Spatial seed regions and weaker evidence
growth remain explicit comparisons. Numerical thresholds and scales are normal-only.
An early development run also exposed accidental reuse of the growth-adjusted
seed boundary for the seed-only ablation. That run was stopped and is excluded
from publication; the authoritative comparison calibrates each policy separately.
The corrected spatial study exposed foreground-overlap losses, so the final
comparison adds isolated component cleanup at the unchanged original threshold.
The already fixed 0.03% minimum area is reused; it is not searched over test masks.
This development/model-selection history is disclosed, and the nominal intervals
are exploratory rather than independent confirmation after policy selection.
No numerical search over defect labels or category-specific test winners is performed.
Because the public benchmark images have already been evaluated during project
development, the full results are method comparisons rather than a claim of a
fresh blind industrial acceptance test.

## Datasets and conditions

- VisA: all 12 categories, 224x224 aligned center crop, verified M5/M4 primary banks.
- KolektorSDD2: independently fitted augmented M9 bank, full-frame 224x224 input,
  the unchanged 56x56 M9 localization grid.
- Clean plus frozen severe JPEG, blur, noise, and geometric conditions from M4.
- Every official-test image is included in binary-support measurement.
- The offline evaluation measures support even when capture quality would reject
  an input. The live interface still skips inference for rejected captures.

## Metrics and decisions

The main measure is per-image IoU on anomalous images with visible positive
ground truth, reported per category with Dice, precision, and recall. Empty
normal masks do not earn credit in this foreground score. Labelled anomalies
whose mask vanishes through cropping or raster sampling are explicitly counted
and have undefined foreground IoU; they remain in the overall image counts.

Report legacy all-image mean IoU separately, normal pixel false-positive rate,
normal-image contour incidence, background pixel false positives, and paired
foreground-IoU differences versus the frozen source policy with 95% bootstrap
intervals. Use 1,000 paired image resamples per category/condition. Macro VisA
summaries give categories equal weight; do not pool VisA and KSDD2 grids.

The live policy uses the same original score map and binary-support function
as evaluation, verifies model/source/input geometry hashes before serving, and
exports policy, threshold units, region count, and flagged area
as metadata. Image-level anomaly decisions remain separately calibrated.

## Reproducibility

The experiment saves config/source/code hashes, source-index identity, spatial
numeric artifacts, fit/calibration IDs, per-image support measurements, category
metrics, bootstrap comparisons, and condition checkpoints outside Git. Public
exports contain aggregate metrics, written analysis, and hashes only.
