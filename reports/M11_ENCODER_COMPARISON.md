# M11 modern encoder comparison

## Outcome and scope

The frozen ConvNeXt-Tiny substitution changes clean VisA macro image AUROC by
-0.0324 relative to matched-budget ResNet-18, with a 95%
category-bootstrap interval [-0.0690,
-0.0006]. Its visible-defect IoU difference is
-0.0551; conformal normal FPR changes by
+1.35 percentage points. On KSDD2 the clean image AUROC difference is
-0.1175, with paired-image interval
[-0.1573, -0.0751].

This is a completed practical representation comparison, not automatic model
promotion. The live M10 inspector, its memories/calibration, and quality gate
remain unchanged. Representation gains must be balanced against false alarms,
localization, severe-shift failures, and measured cost below. No single pooled
score establishes industrial readiness or physical defect diagnosis.

The measured substitution does not support replacing ResNet: clean image ranking and visible-defect overlap are lower on both domains, while the checkpoint and executed encoder are larger. Retain the current ResNet-based inspector; do not cherry-pick per-category test winners.

There is a real trade-off under KSDD2's four selected severe conditions: mean
normal FPR falls from 64.74% to 18.37%,
but confident defect recall also falls from 93.18%
to 56.59%. These benefits and losses are both
retained below; fewer alerts are not automatically better inspection.

This result applies to the common first-three-stage, 64-dimensional projected
memory recipe, initially developed around ResNet. It is not a best-tuned
ConvNeXt result, a test of every modern encoder, or proof of intrinsic
architectural inferiority. Different features/projection budgets would require
a separately declared comparison, not tuning against these test outcomes.

## Matched method

Both frozen ImageNet-1K encoders supply their first three spatial stages,
aligned to a 28x28 patch grid. ResNet yields 448 feature channels and ConvNeXt
672. Both use 224 RGB input, identical geometry and normalization, a seeded
64-dimensional normalized projection, 4,096 candidates, 512 memory entries,
24 patches per training normal split equally over clean and one frozen mild
nuisance view, top-1%-patch image scores, and sigma-4 map smoothing.

At most 256 sample-ID-ranked training normals per category are used for both
encoders. All existing validation normals independently calibrate each encoder;
all 3,166 official test images remain included (2,162 VisA and 1,004 KSDD2).
No test-based layer, threshold, augmentation, or category-winner search is used.
The first three stages are executed, not the classifier or fourth stage.
Capacity and pretraining recipe differ: this is not architecture-only causality.

The study preserves VisA center-crop and KSDD2 full-frame geometry, using a
common 56x56 mask/evidence grid for both encoders. This VisA localization grid
differs from M10's 224 grid; do not compare those IoUs directly. Sampled/cropped
away positive masks are counted explicitly in category tables and excluded only
from undefined foreground metrics, never from image detection. Historical M5/M9
numbers use different training budgets and are not the matched reference here.
On clean inputs, 73 of
1200 VisA anomaly masks and
1 of 110
KSDD2 anomaly masks have no positive pixels after the declared geometry/rasterization.
This is a localization-resolution limitation, not evidence that those products
are normal; both encoders use the same visible subset. Counts for every condition
are in `mask_visibility.csv`.

Thresholds use clean calibration normals only: upper-tail conformal p <= .05
alerts, .05 < p <= .20 requests review, and p > .20 is normal; pixel thresholds
use the 99.5th percentile of smoothed calibration pixels. No capture gate hides
shifted encoder failures; neither encoder uses M10 cleanup or seeded growth.
Clean false-alarm validity needs exchangeability and
does not guarantee shifted control.

See the [fixed protocol](../docs/M11_ENCODER_PROTOCOL.md), including checkpoint
sources, licence boundary, representation choices, and computational limits.

## Clean results

VisA values are equal-category means; KSDD2 is a separate domain.

| Dataset | Encoder | Image AUROC | Image AP | Conformal normal FPR | Confident anomaly recall | AUPRO@30 | Visible-defect IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| visa | resnet18 | 0.8668 | 0.8885 | 3.89% | 50.92% | 0.8185 | 0.1281 |
| visa | convnext_tiny | 0.8344 | 0.8574 | 5.24% | 50.67% | 0.7333 | 0.0730 |
| ksdd2 | resnet18 | 0.9354 | 0.8491 | 3.91% | 80.91% | 0.8600 | 0.2943 |
| ksdd2 | convnext_tiny | 0.8178 | 0.3955 | 4.70% | 31.82% | 0.7721 | 0.2074 |

Foreground overlap never rewards empty normal masks. Aggregate CSVs additionally
include pixel ranking/AP, Dice, precision/recall, normal-pixel FPR, selective
coverage/risk, and legacy all-image IoU under distinct names. A high ranking
score does not imply accurate binary defect boundaries or acceptable miss rates.

## Clean category differences: ConvNeXt minus ResNet

| Dataset/category | Image AUROC delta | 95% paired-image interval | Normal FPR delta (pp) | Defect IoU delta |
|---|---:|---|---:|---:|
| ksdd2/surface | -0.1175 | [-0.1573, -0.0751] | +0.78 | -0.0869 |
| visa/candle | -0.0024 | [-0.0591, +0.0586] | -1.00 | -0.0555 |
| visa/capsules | -0.0862 | [-0.1917, +0.0332] | +8.33 | -0.0771 |
| visa/cashew | -0.0426 | [-0.0952, +0.0058] | +2.00 | -0.0403 |
| visa/chewinggum | +0.0280 | [+0.0054, +0.0570] | +8.00 | -0.0874 |
| visa/fryum | -0.0570 | [-0.1182, +0.0040] | -2.00 | -0.0396 |
| visa/macaroni1 | -0.0138 | [-0.0860, +0.0500] | -2.00 | -0.0586 |
| visa/macaroni2 | -0.1795 | [-0.2670, -0.0942] | -1.00 | -0.0288 |
| visa/pcb1 | -0.0070 | [-0.0417, +0.0272] | -1.00 | -0.0384 |
| visa/pcb2 | +0.0762 | [+0.0074, +0.1443] | -3.00 | -0.0775 |
| visa/pcb3 | -0.0935 | [-0.1594, -0.0220] | +2.97 | -0.0800 |
| visa/pcb4 | -0.0066 | [-0.0394, +0.0253] | +6.93 | -0.0699 |
| visa/pipe_fryum | -0.0042 | [-0.0238, +0.0138] | -2.00 | -0.0075 |

Intervals use 1,000 stratified paired image resamples; VisA macro intervals
resample category differences equally. Shifted category intervals use 100
descriptive resamples. One seed and reused public benchmarks limit confirmation;
nominal intervals are not simultaneous multiple-comparison guarantees. Paired
fit/calibration IDs, test ordering, and labels are checked before aggregation.
Intervals condition on the fitted models and fixed normal calibration; they do
not include refitting uncertainty or establish independent capture-session validity.

## Severe capture shifts

| Dataset | Condition | ResNet AUROC | ConvNeXt AUROC | ResNet normal FPR | ConvNeXt normal FPR | ResNet defect IoU | ConvNeXt defect IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| visa | jpeg_severe | 0.8022 | 0.7085 | 53.14% | 72.43% | 0.0655 | 0.0458 |
| visa | blur_severe | 0.7123 | 0.6643 | 92.11% | 81.42% | 0.0399 | 0.0403 |
| visa | noise_severe | 0.7756 | 0.7250 | 73.22% | 63.51% | 0.0394 | 0.0469 |
| visa | geometric_severe | 0.8241 | 0.7725 | 23.18% | 21.86% | 0.0975 | 0.0623 |
| ksdd2 | jpeg_severe | 0.8917 | 0.7968 | 61.19% | 22.82% | 0.1738 | 0.1997 |
| ksdd2 | blur_severe | 0.9406 | 0.7447 | 74.61% | 34.68% | 0.1267 | 0.1981 |
| ksdd2 | noise_severe | 0.8670 | 0.8562 | 100.00% | 4.47% | 0.0938 | 0.2255 |
| ksdd2 | geometric_severe | 0.8724 | 0.7459 | 23.15% | 11.52% | 0.2470 | 0.1867 |

These four severe conditions are a fixed diagnostic subset of M4, not all
severities or a representative factory capture distribution. Clean calibration
and a newer encoder do not remove the need for capture control and human review.
Score-rank stability relative to the paired clean images is in the tables.

The following are equal-weight descriptive means over the four severe conditions,
with equal category weights within VisA. Lower false alarms are not sufficient
if confident defect recall also collapses. Review cases are not automatic errors
or guaranteed normal products; selective risk applies only to automatic decisions.

| Dataset | Encoder | Normal FPR | Confident defect recall | Automatic coverage | Selective risk |
|---|---|---:|---:|---:|---:|
| visa | resnet18 | 60.41% | 82.92% | 82.85% | 32.54% |
| visa | convnext_tiny | 59.81% | 75.31% | 84.44% | 36.07% |
| ksdd2 | resnet18 | 64.74% | 93.18% | 82.37% | 70.10% |
| ksdd2 | convnext_tiny | 18.37% | 56.59% | 69.32% | 27.39% |

## Resources

| Dataset | Encoder | Active parameters (M) | Checkpoint (MB) | All indexes (MB) | Single-image median (ms) | p95 (ms) |
|---|---|---:|---:|---:|---:|---:|
| visa | resnet18 | 2.78 | 46.83 | 2.75 | 33.9 | 41.0 |
| visa | convnext_tiny | 12.35 | 114.42 | 3.39 | 54.1 | 61.6 |
| ksdd2 | resnet18 | 2.78 | 46.83 | 0.23 | 26.9 | 33.7 |
| ksdd2 | convnext_tiny | 12.35 | 114.42 | 0.28 | 50.8 | 60.9 |

Single-image measurements use the same five normal and five anomalous samples
per category, a warm-up, four PyTorch CPU threads, and one-image calls. They
include image/mask decoding, feature extraction, memory scoring, and map
interpolation/smoothing, but exclude browser rendering, quality gating,
publication, calibration, and metric calculation. Thus they are not the M6/M10
end-to-end browser benchmark. The saved environment identifies CPU/CUDA mode.
Batch costs are separate fields and should not be reported as single-image latency.

Checkpoint bytes include the original full classifier checkpoint; active
parameters count only the executed first three stages. Projected memory is
512x64 float32 for both encoders; different feature dimensions change projection
and compressed-index bytes. Process working set includes shared libraries,
metric buffers, and allocator/order history, so it is descriptive rather than
isolated encoder RAM. No claim of equal total compute or byte budget is made.

## Reproducibility and publication

Authoritative run: `m11-encoder-comparison-full-20261015T005134Z`. It contains 130 category/encoder/condition
metric rows, 65 paired comparisons, and 26 matched single-image resource jobs.
Config, source code, environment, manifest/audit, weight, fitted-bank, calibration,
condition, and benchmark identities are recorded. Every committed output is
hash-verified before resume/republication. Interrupted runs retain completed units.

Public [aggregate tables and manifest](m11/) exclude images, per-image
predictions, weights, and feature banks. Follow the
[manual verification guide](../docs/M11_MANUAL_TESTING.md) for local checks and
the exact report rebuild command. The existing application is intentionally not
silently replaced by this offline research comparison.
