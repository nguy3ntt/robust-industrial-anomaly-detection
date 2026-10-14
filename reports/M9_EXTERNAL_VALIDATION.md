# M9 independent validation on KolektorSDD2

## Outcome

The frozen normal-only method was fitted and calibrated on a second industrial
dataset, then evaluated on all 1,004 official test images. The augmented model
achieves clean image AUROC 0.9393, with a stratified
95% bootstrap interval [0.9065,
0.9650]. Its clean conformal normal false-positive
rate is 2.68%, with exact 95% interval
[1.73%, 3.97%].

The locally calibrated capture gate rejects 1.45%
of clean normals and 12.33% of shifted normals across the frozen
stress conditions. It intercepts 1,123 of 5,128
shifted-normal false alerts (21.90%). False alerts here
include both anomalous and review dispositions from the ungated detector.

The current gate therefore does not generalize uniformly: severe JPEG is
reliably rejected, but severe blur, noise, and geometry often pass even when
they trigger substantial false-positive rates. The second dataset supports
normal-only adaptation of the detector and exposes a limitation of the existing
gate. It does not justify transferring the VisA 93.22% interception result to
other industrial captures. The frozen gate is retained for this validation;
any redesign requires a new protocol rather than tuning against these test labels.

This validates the method with normal adaptation to a second dataset. It does
not establish zero-shot VisA-memory transfer or factory deployment readiness.

## Data and frozen method

- Official source: [ViCoS KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/).
- Licence: CC BY-NC-SA 4.0; noncommercial portfolio/academic use.
- Fit: 1,772 provider training normals.
- Calibration: 313 training-derived normals.
- Official test: 894 normals and 110 defects, all preserved.
- Provider training defects: all 246 excluded from fitting/calibration.
- Archive copies: two byte-identical training copy files explicitly excluded;
  originals remain, with their hashes recorded in the dataset audit.
- Leakage checks: exact and confirmed perceptual components before calibration
  splitting; the acquired dataset contains no confirmed duplicate pairs.
- Frozen ImageNet ResNet-18, multi-scale features, 64-dimensional projection,
  4,096 candidates, 512-entry k-center banks, and 24 total patches per image.
- Plain memory uses 24 clean patches. Augmentation splits that budget into 12
  clean and 12 mildly shifted normal patches using the pre-existing M4 pool.
- Full-frame image/mask resize to 224x224; localization evaluated at 56x56.
- Fixed conformal anomaly/review levels 0.05/0.20 and quality rejection level 0.02.

The [frozen protocol](../docs/M9_EVALUATION_PROTOCOL.md) and
[dataset card](../docs/datasets/KSDD2.md) describe the split, geometry, sampling,
licence, and provenance boundaries.

## Clean results

| Model | Image AUROC | Image AP | Normal FPR | Confident recall | Pixel AUROC | Pixel AP | AUPRO@30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain_coreset_512 | 0.9392 | 0.8555 | 0.0291 | 0.8000 | 0.9673 | 0.4206 | 0.8826 |
| augmented_coreset_512 | 0.9393 | 0.8531 | 0.0268 | 0.7818 | 0.9622 | 0.4216 | 0.8717 |

All ranking/localization results above use ungated scores on the full official
test partition. Pixel metrics use the declared 56x56 grid, not native resolution;
their values cannot be compared directly with VisA's 224x224 localization.
The clean evaluation loses 1
positive mask(s) through raster downsampling; their images remain in detection
evaluation and this count is disclosed rather than silently excluding cases.

## Capture gate and human action

Across the twelve stress conditions, equal-weight mean image AUROC is
0.9157 for the plain model and
0.9194 for augmentation. Mean
ungated normal FPR is 34.87% versus
25.99%. These are descriptive
paired-condition summaries; conditions repeatedly transform the same test images
and are not independent samples for a significance claim.

On clean defects, the gate requests recapture for
5.45% of images. Automatic confident
defect recall changes from 78.18% to
73.64%. Total required operator action,
including recapture, review, and anomalous outcomes, covers
86.36% of clean defects.

The guarded automatic coverage is 84.96% and
automatic selective risk is 4.22%.
Recapture is an unresolved workflow state. Capture statistics can also change
because of a defect, so rejected images must not be labelled normal or treated
as camera-only failures.

## Frozen stress conditions, augmented model

| Condition | Image AUROC | Ungated normal FPR | Normal recapture | Guarded normal FPR | Auto defect recall | Defect action coverage |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.9393 | 0.0268 | 0.0145 | 0.0235 | 0.7364 | 0.8636 |
| jpeg_moderate | 0.9320 | 0.0794 | 0.2998 | 0.0492 | 0.5727 | 0.9364 |
| jpeg_severe | 0.9028 | 0.5951 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| resize_moderate | 0.9345 | 0.0548 | 0.0089 | 0.0537 | 0.7545 | 0.9000 |
| resize_severe | 0.9401 | 0.1286 | 0.0067 | 0.1275 | 0.8091 | 0.9727 |
| blur_moderate | 0.9412 | 0.1488 | 0.0045 | 0.1465 | 0.8364 | 0.9818 |
| blur_severe | 0.9287 | 0.7438 | 0.0034 | 0.7405 | 0.9818 | 1.0000 |
| noise_moderate | 0.9258 | 0.0537 | 0.0224 | 0.0470 | 0.7182 | 0.9091 |
| noise_severe | 0.8542 | 0.9989 | 0.1107 | 0.8881 | 0.7273 | 1.0000 |
| photometric_moderate | 0.9425 | 0.0157 | 0.0056 | 0.0145 | 0.7364 | 0.8273 |
| photometric_severe | 0.9296 | 0.0067 | 0.0000 | 0.0067 | 0.6545 | 0.7545 |
| geometric_moderate | 0.9257 | 0.0794 | 0.0089 | 0.0705 | 0.7636 | 0.9000 |
| geometric_severe | 0.8760 | 0.2136 | 0.0089 | 0.2047 | 0.7636 | 0.9000 |

The experiment deliberately computes ungated neural scores for rejected inputs
to measure interception and missed-detection trade-offs. This is an offline
counterfactual evaluation. The live M8 interface still skips neural inference
when its quality gate fails. Recapture can reduce automatic false alerts by
withholding inputs; it does not repair image quality or prove shift invariance.

## Scope and reproducibility

This additional dataset contains one industrial surface category. Its full-frame
resize retains strip-end evidence while changing aspect ratio and reducing fine
defect detail. The source does not provide physical-item/session grouping, so
the audit establishes content-duplicate separation rather than item independence.
Bootstrap intervals resample images and may be optimistic under unobserved
production correlations. No test labels select parameters or thresholds.

The authoritative run is `m9-ksdd2-validation-full-20261013T045835Z`. Model weights, source hashes, resolved
configuration, calibration arrays, memory indexes, per-image predictions, and
condition checkpoints remain in local artifact storage. The original VisA/M7
results remain frozen. Aggregate outputs and their manifest are in [m9/](m9/).

Attribution: Jakob Božič, Domen Tabernik, and Danijel Skočaj,
*Mixed supervision for surface-defect detection: from weakly to fully supervised
learning*, Computers in Industry, 2021.
