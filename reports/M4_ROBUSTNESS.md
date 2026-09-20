# M4 robustness and generalization report

## Scope and controls

M4 evaluates controlled capture and post-processing shifts on the unchanged
official VisA test partition. All conditions, severities, models, thresholds,
and intervention settings were checked in before the full run. Image decisions
continue to use category-specific thresholds derived only from clean
training-normal validation samples.

The study compares the 2,048-entry mid-scale reference, the 512-entry
multi-scale k-center model, and an equal-capacity intervention. The intervention
uses the same 512-entry bank and the same total candidate-patch budget as the
plain coreset, but divides each normal training image's budget equally between
its clean view and one deterministically assigned mild nuisance view. No defect
images are used to construct or calibrate any memory.

## Frozen shift ladder

| Family | Severity 1 | Severity 2 |
|---|---|---|
| JPEG | quality 70 | quality 35 |
| Resize | 0.75× down/up | 0.50× down/up |
| Gaussian blur | sigma 1.0 | sigma 2.0 |
| Sensor-like Gaussian noise | standard deviation 0.02 | standard deviation 0.05 |
| Photometric | brightness/contrast 0.90/0.90 | brightness/contrast 0.75/0.75 |
| Geometric | rotation 2°, translation (+2, −2) px | rotation 5°, translation (+5, −5) px |

Noise is seeded by the experiment seed, sample identifier, and condition.
Geometric transforms apply the same affine mapping to images and masks;
shifted explanation maps are inverse-aligned before stability measurement.

## Spatial preprocessing correction

During M4 geometry validation, the earlier M2/M3 evaluator was found to center
crop non-square images but stretch masks. This affected only localization
metrics and panels, not image scores, thresholds, or memory banks. Masks now use
the identical centered crop with nearest-neighbour sampling. M2 and M3 were
regenerated without changing any model setting. All 2,162 M4 clean scores for
both source models match the corrected M3 predictions exactly.

Corrected clean macro results are:

| Model | Image AUROC | Image AP | Normal FPR | Pixel AUROC | Pixel AP | Mean IoU | AUPRO@30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mid-scale reference | 0.8283 | 0.8527 | 0.0477 | 0.9488 | 0.3038 | **0.2507** | 0.7123 |
| Multi-scale k-center | **0.8679** | 0.8876 | **0.0471** | 0.9555 | 0.3210 | 0.1665 | 0.7599 |
| Augmented multi-scale k-center | 0.8674 | **0.8907** | 0.0496 | **0.9557** | **0.3226** | 0.1674 | 0.7762* |

`*` This table uses M4's 30-threshold AUPRO integration grid; M3 uses 50
thresholds. Clean-to-shift comparisons always use the same M4 grid.

## Robustness degradation

The following table reports means across both severities and 12 categories for
the plain multi-scale coreset. False-positive-rate changes use the unchanged
clean validation threshold.

| Family | Δ image AUROC | Δ normal FPR | Δ pixel AUROC | Rank Spearman | Heatmap correlation |
|---|---:|---:|---:|---:|---:|
| JPEG | −0.0409 | +0.3642 | −0.0250 | 0.8462 | 0.7428 |
| Resize | −0.0130 | +0.1993 | −0.0100 | 0.8837 | 0.8197 |
| Blur | **−0.1018** | **+0.6426** | −0.0412 | 0.7051 | 0.6362 |
| Noise | −0.0615 | +0.4028 | **−0.0478** | 0.8283 | 0.6283 |
| Photometric | −0.0062 | −0.0026 | +0.0016 | **0.9554** | **0.9282** |
| Geometric | −0.0344 | +0.1307 | −0.0080 | 0.7883 | 0.4330 |

Blur is the largest image-ranking and false-alarm failure. Noise causes the
largest pixel-ranking loss. Photometric darkening largely preserves ranking but
can still convert anomaly decisions into false negatives. Geometric heatmap
correlation is low even after inverse alignment, showing that spatial evidence
is less stable than score ranking.

## Confirmatory tests

| Test | Estimate | 95% interval | Conclusion |
|---|---:|---:|---|
| H2: multi-scale image-robustness advantage over the reference for resize, blur, and geometry | −0.0072 | [−0.0235, +0.0083] | Inconclusive |
| Equal-budget augmentation robustness advantage | +0.0077 image AUROC | [+0.0030, +0.0126] | Supported |
| H5: heatmap-stability separation between retained-correct cases and shift failures | +0.2153 correlation | [+0.2066, +0.2253] | Supported |

The augmentation intervention also reduces shifted normal FPR by 0.0774,
improves pixel AUROC by 0.0033 and mean IoU by 0.0155 relative to the plain
coreset. It slightly reduces AUPRO by 0.0039, so it is not a universal win.

## Category transfer

The zero-shot leave-one-category-out memory contains 128 prototypes from each of
the other 11 categories and no target-category training samples. Across held-out
categories it obtains image AUROC 0.5783, image AP 0.6390, pixel AUROC 0.8322,
and pixel AP 0.0843. Image AUROC is 0.2896 below the category-specific coreset on
average. Every category loses image AUROC, with the largest gap on PCB4
(−0.5552). These results reject using the current shared feature memory as a
category-agnostic detector.

## Failure and shortcut analysis

The principal clustered failure is nuisance-driven false positives:

- plain coreset blur: 1,244 new false positives and 18 new false negatives;
- plain coreset noise: 721 new false positives and 37 new false negatives;
- plain coreset JPEG: 677 new false positives and 41 new false negatives;
- augmentation reduces those false-positive counts to 926, 679, and 513;
- photometric shift instead produces more false negatives than false positives.

Several severe blur/noise clusters convert about half of clean-correct samples
to errors, especially candle, macaroni1, PCB2, PCB3, and PCB4. Score changes have
moderate nuisance correlations in some categories: the largest absolute
Spearman correlation is 0.5275 between brightness change and coreset score
change for severe-noise cashew. This is evidence of capture-statistic sensitivity,
not proof of a causal shortcut or physical defect mechanism.

## Limitations and next use

These are controlled digital perturbations, not measurements from new cameras or
factories. Bootstrap intervals quantify sampling uncertainty within VisA; they
do not establish external validity. The intervention becomes the primary M5
calibration candidate because it improves shifted detection and false-positive
behavior with almost unchanged clean performance. The original reference and
plain coreset remain comparison models. Outputs remain inspection decision
support and must not be interpreted as product certification or root-cause
diagnosis.
