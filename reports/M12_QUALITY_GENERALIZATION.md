# Capture-quality gate generalization

## Outcome and scope

This completed comparison evaluates the original v1 gate and one fixed v2 gate
on both VisA and KolektorSDD2. The detector, anomaly calibration, official splits,
all twelve capture shifts, and sample seeds are unchanged. Only gate measurements
and their normal-only fitted references change; no new backbone training occurs.

| Dataset | Gate | Clean normal recapture | Clean anomaly recapture | Shifted normal recapture | Shifted false-alert interception |
|---|---|---:|---:|---:|---:|
| visa | v1 | 1.98% | 17.08% | 89.13% | 93.22% |
| visa | v2 | 2.08% | 17.50% | 91.67% | 94.80% |
| ksdd2 | v1 | 1.45% | 5.45% | 12.33% | 21.90% |
| ksdd2 | v2 | 1.45% | 5.45% | 12.62% | 22.50% |

KSDD2 shifted false-alert interception changes from
21.90% to
22.50%; clean-normal
recapture changes from 1.45% to
1.45%. VisA interception changes from
93.22% to
94.80%, and clean-defect
deferral changes from 17.08% to
17.50%. These domain-specific trade-offs
are why v2 remains an explicit comparison, not a silent universal upgrade.

The main cross-domain problem remains unresolved: KSDD2 severe-blur normal
recapture changes only from 0.34% to 3.80%, and severe-noise
recapture is 11.07% versus 11.07%. Most such inputs still pass.
The small aggregate gain is not evidence that these proxies generalize reliably.
Retain the historical default; v2 is an auditable research comparison, not a
recommended production replacement. Additional operator action and slightly
greater clean-defect deferral must be considered alongside interception.

Interception counts original review OR anomalous calls on normal images that
are now withheld. It does not mean the products are classified correctly or that
operator workload fell. All shifted summaries are descriptive sample-weighted
totals across repeated conditions, not independent trials. VisA summaries here
are sample-weighted, not equal-category means. All category/condition/label
results, including non-target shifts and regressions, remain in the public CSV.

## Fixed intervention

V2 retains the original eight quality measurements and adds log fine/coarse
energy for contrast-normalized focus, plus a contrast-normalized robust noise
estimate from low-structure pixels. Both gates use median/MAD-IQR normal
references, maximum directional deviation, and clean-normal split-conformal
calibration at alpha .02. No anomalies or shifted images fit the gate.

Each domain/category receives its own original training/calibration normals.
VisA retains centered 224x224 crops; KSDD2 retains full-frame bilinear 224x224
geometry. This is local normal adaptation, not a VisA-to-unseen-product gate.
Historical v1 dispositions were recomputed and required to match saved results
for every evaluated image and condition. See the [fixed protocol](../docs/M12_QUALITY_PROTOCOL.md).

## Normal-input trade-offs

All rates use ALL normal inputs as denominator, including recaptures. The last
column is recapture OR review OR anomalous: unresolved required operator action,
not just model review. A low remaining automatic false-positive rate can coexist
with a very high recapture burden. The full CSV also includes exact binomial
rejection intervals by category/condition, conditional on image independence.

| Dataset | Condition | Gate | Recapture | Remaining automatic anomalous | Total operator action |
|---|---|---|---:|---:|---:|
| visa | clean | v1 | 1.98% | 4.05% | 21.31% |
| visa | clean | v2 | 2.08% | 4.05% | 21.31% |
| visa | jpeg_moderate | v1 | 61.02% | 1.98% | 72.14% |
| visa | jpeg_moderate | v2 | 61.02% | 1.87% | 72.14% |
| visa | jpeg_severe | v1 | 92.00% | 0.94% | 97.51% |
| visa | jpeg_severe | v2 | 92.00% | 0.94% | 97.51% |
| visa | resize_moderate | v1 | 100.00% | 0.00% | 100.00% |
| visa | resize_moderate | v2 | 100.00% | 0.00% | 100.00% |
| visa | resize_severe | v1 | 100.00% | 0.00% | 100.00% |
| visa | resize_severe | v2 | 100.00% | 0.00% | 100.00% |
| visa | blur_moderate | v1 | 100.00% | 0.00% | 100.00% |
| visa | blur_moderate | v2 | 100.00% | 0.00% | 100.00% |
| visa | blur_severe | v1 | 100.00% | 0.00% | 100.00% |
| visa | blur_severe | v2 | 100.00% | 0.00% | 100.00% |
| visa | noise_moderate | v1 | 73.70% | 0.52% | 79.31% |
| visa | noise_moderate | v2 | 76.09% | 0.52% | 81.29% |
| visa | noise_severe | v1 | 100.00% | 0.00% | 100.00% |
| visa | noise_severe | v2 | 100.00% | 0.00% | 100.00% |
| visa | photometric_moderate | v1 | 85.86% | 0.00% | 86.90% |
| visa | photometric_moderate | v2 | 85.55% | 0.00% | 86.69% |
| visa | photometric_severe | v1 | 100.00% | 0.00% | 100.00% |
| visa | photometric_severe | v2 | 100.00% | 0.00% | 100.00% |
| visa | geometric_moderate | v1 | 75.26% | 1.25% | 80.04% |
| visa | geometric_moderate | v2 | 92.93% | 0.31% | 94.80% |
| visa | geometric_severe | v1 | 81.70% | 5.61% | 92.00% |
| visa | geometric_severe | v2 | 92.41% | 2.08% | 96.99% |
| ksdd2 | clean | v1 | 1.45% | 2.35% | 17.67% |
| ksdd2 | clean | v2 | 1.45% | 2.35% | 17.67% |
| ksdd2 | jpeg_moderate | v1 | 29.98% | 4.92% | 53.24% |
| ksdd2 | jpeg_moderate | v2 | 29.98% | 4.92% | 53.24% |
| ksdd2 | jpeg_severe | v1 | 100.00% | 0.00% | 100.00% |
| ksdd2 | jpeg_severe | v2 | 100.00% | 0.00% | 100.00% |
| ksdd2 | resize_moderate | v1 | 0.89% | 5.37% | 25.62% |
| ksdd2 | resize_moderate | v2 | 0.89% | 5.37% | 25.62% |
| ksdd2 | resize_severe | v1 | 0.67% | 12.75% | 48.32% |
| ksdd2 | resize_severe | v2 | 0.67% | 12.75% | 48.32% |
| ksdd2 | blur_moderate | v1 | 0.45% | 14.65% | 52.24% |
| ksdd2 | blur_moderate | v2 | 0.45% | 14.65% | 52.24% |
| ksdd2 | blur_severe | v1 | 0.34% | 74.05% | 99.44% |
| ksdd2 | blur_severe | v2 | 3.80% | 70.81% | 99.44% |
| ksdd2 | noise_moderate | v1 | 2.24% | 4.70% | 29.31% |
| ksdd2 | noise_moderate | v2 | 2.24% | 4.70% | 29.31% |
| ksdd2 | noise_severe | v1 | 11.07% | 88.81% | 100.00% |
| ksdd2 | noise_severe | v2 | 11.07% | 88.81% | 100.00% |
| ksdd2 | photometric_moderate | v1 | 0.56% | 1.45% | 8.17% |
| ksdd2 | photometric_moderate | v2 | 0.56% | 1.45% | 8.17% |
| ksdd2 | photometric_severe | v1 | 0.00% | 0.67% | 3.80% |
| ksdd2 | photometric_severe | v2 | 0.00% | 0.67% | 3.80% |
| ksdd2 | geometric_moderate | v1 | 0.89% | 7.05% | 29.19% |
| ksdd2 | geometric_moderate | v2 | 0.89% | 7.05% | 29.19% |
| ksdd2 | geometric_severe | v1 | 0.89% | 20.47% | 46.64% |
| ksdd2 | geometric_severe | v2 | 0.89% | 20.47% | 46.64% |

## Defect deferral is not successful detection

The following clean-anomaly rates separately expose loss of immediate model
decisions and increased unresolved work. Recapture returns no anomaly score or
heatmap. It must not be counted as confident defect detection or as normal.

| Dataset | Gate | Original confident defect recall | Remaining confident defect recall | Recapture | Total operator action |
|---|---|---:|---:|---:|---:|
| visa | v1 | 51.58% | 42.33% | 17.08% | 79.50% |
| visa | v2 | 51.58% | 42.08% | 17.50% | 79.58% |
| ksdd2 | v1 | 78.18% | 73.64% | 5.45% | 86.36% |
| ksdd2 | v2 | 78.18% | 73.64% | 5.45% | 86.36% |

## Resources and deployment

| Dataset | Gate | Mean gate-only CPU ms |
|---|---|---:|
| visa | v1 | 1.92 |
| visa | v2 | 3.37 |
| ksdd2 | v1 | 2.55 |
| ksdd2 | v2 | 4.68 |

These paired offline timings cover all test-condition gate calls, not image
decoding, transforms, anomaly inference, browser rendering, or a fresh end-to-end
latency benchmark. Fixed v1-then-v2 execution order and host load limit precision.

Historical M6/M8/M10 configurations remain reproducible. The separate M12
configuration enables v2 with the same ResNet detector and cleaned contours;
it does not silently replace the user's running server. Integrity checks bind
the v2 artifact to the same M5 calibration and trusted VisA manifest.
KSDD2 remains a separate research evaluation, not a browser product category.

## Limits and reproducibility

This follow-up was motivated by observed M9 failures on reused benchmarks.
It is exploratory development evidence, not untouched external confirmation,
proof of camera failure, or industrial safety certification. Texture, defects,
noise and blur can still confound the measurements; combined corruptions and
physically grouped camera/session captures remain unvalidated. Clean conformal
validity requires exchangeability and does not guarantee shifted false alarms.
Repeated shifts and unknown item/session dependence prevent treating pooled
counts as independent significance evidence. No post-test feature/threshold sweep
or category-specific winner selection is performed.

Authoritative run: `m12-quality-generalization-full-20261016T062055Z`. All 3,166 official test images are evaluated in
13 conditions with both gates (82,316 gate decisions), producing 676 aggregate
category/condition/label/gate rows. Normal fit/calibration membership, source
metadata, input/code/config fingerprints, and condition checkpoints are recorded.
Interrupted runs resume only from matching, verified inputs. Public files contain
aggregates only: no images, per-image predictions, weights, or normal references.
See the [manual verification guide](../docs/M12_MANUAL_TESTING.md).
