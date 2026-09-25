# Robust Normal-Only Industrial Anomaly Detection Under Capture Shift

## Abstract

Normal-only visual anomaly detection is attractive when defective examples are
rare, but benchmark accuracy can conceal brittle operating behavior. We develop
and evaluate an auditable nearest-neighbour system on VisA that couples frozen
ResNet-18 multi-scale patch features, bounded k-center memory, normal-only
augmentation, category-conditional conformal calibration, selective human
review, and spatial evidence. The official one-class test partition is
preserved; calibration uses only grouped training-normal images, and exact or
confirmed perceptual duplicates may not cross splits. Across 12 categories, the
primary augmented 512-prototype system obtains macro image AUROC 0.8674, pixel
AUROC 0.9557, and AUPRO@30 0.7762. An equal-budget augmentation intervention
improves shifted image AUROC by 0.0077 (95% CI [0.0030, 0.0126]). Clean
category-conformal normal false-positive rate is 0.0430 with 0.8042 automatic
coverage; however, severe blur, noise, and JPEG compression raise false-positive
rates to 0.8686, 0.6820, and 0.4696. Zero-shot category-transfer AUROC is only
0.5783. These failures show that clean calibration is not a distribution-shift
guarantee. The accompanying local workstation exposes scores, spatial evidence,
policy regions, latency, and limitations while keeping images in memory. The
study prioritizes reproducibility, false positives, calibration, explanation
stability, and negative results over a single pooled benchmark score.

## 1. Introduction

Industrial inspection often begins with many acceptable examples and few
representative defects. Normal-only methods therefore model acceptable visual
variation and flag deviations. Their deployment risk is asymmetric: nuisance
changes can create expensive false alarms, while illumination or geometry can
hide defects or destabilize spatial evidence. An image AUROC measured on a
familiar test distribution does not describe these failure modes.

This work asks whether a simple, auditable patch-memory system can remain useful
under controlled capture shifts and whether normal-only calibration can define
a safer operating policy. The contributions are:

1. a leakage-aware VisA manifest preserving the provider test split;
2. a measured progression from interpretable and global baselines to bounded
   multi-scale k-center memory;
3. a deterministic six-family robustness study with paired detection and
   explanation-stability analysis;
4. category-conditional normal-only calibration and selective review; and
5. a privacy-preserving local workstation plus a hash-gated public package.

The results support the augmentation, explanation-stability, category
calibration, and selective-review hypotheses. They do not support a general
multi-scale robustness advantage, and they reject practical zero-shot category
transfer for the evaluated design.

## 2. Related work

VisA provides 10,821 images across 12 categories with image and pixel anomaly
annotations and an official one-class split. PatchCore established a strong
industrial anomaly-detection pattern: frozen pretrained patch features, a
coreset memory, and nearest-neighbour scoring. Our system follows this simple
family because its representation, storage, and scoring decisions remain
inspectable. ResNet-18 supplies the frozen encoder. Unlike benchmark-only work,
the present evaluation makes operating thresholds, shifted false positives,
selective risk, category transfer, and heatmap stability primary outcomes.

## 3. Data governance and protocol

The audited VisA release contains 7,358 training-normal, 1,301
validation-normal, and 2,162 official test images. The test set contains 1,200
anomaly and 962 normal images. Validation is derived deterministically from
official training-normal samples using grouped splitting. The data gate verifies
archive identity, image/mask consistency, category and label rules, exact
duplicates, and confirmed perceptual duplicate groups. The canonical manifest
is hashed and must match every reportable run.

Model choice, memory size, feature layers, augmentation, and thresholds are
frozen before final-test reporting. Detection uses macro per-category image
AUROC, average precision, and normal FPR. Localization uses pixel AUROC, pixel
average precision, mean IoU at a validation-normal-only threshold, and AUPRO to
30% false-positive area. Bootstrap or paired intervals accompany reportable
comparisons.

## 4. Method

### 4.1 Baselines

The sanity baseline concatenates colour, HSV-histogram, local binary pattern,
and gradient descriptors and uses nearest-normal distance. A global baseline
uses frozen ImageNet ResNet-18 embeddings. The spatial reference retains 2,048
mid-scale projected patch prototypes per category and produces a smoothed patch
distance map.

### 4.2 Multi-scale bounded memory

The proposed representation composes frozen ResNet-18 layer-1, layer-2, and
layer-3 features at a common spatial grid. Candidate sampling is bounded and
deterministic. Greedy k-center selection compresses each category memory to 512
prototypes. An image score aggregates the upper tail of the anomaly map rather
than only its single maximum. A unified alternative uses one balanced 1,536-entry
memory across categories.

### 4.3 Robustness intervention

Normal training patches are exposed to the same six nuisance families used in
evaluation while retaining equal patch and prototype budgets. This isolates the
effect of nuisance coverage from additional capacity. Two deterministic
severities are evaluated for JPEG, resize, blur, noise, photometric, and
geometric change. Geometric masks receive the identical transform, and shifted
heatmaps are inverse aligned before correlation and top-support overlap.

### 4.4 Calibration and selective review

Category and pooled empirical-quantile and split-conformal policies are fitted
only to validation-normal scores. The primary category-conformal policy uses
plus-one, tie-conservative p-values. Scores with `p <= 0.05` are anomalous,
`0.05 < p <= 0.20` enter review, and `p > 0.20` are normal. Coverage is the
fraction decided automatically; selective risk is the error rate among those
automatic decisions.

## 5. Clean performance and ablation

| System | Image AUROC | Normal FPR | Pixel AUROC | Pixel AP | Mean IoU | AUPRO@30 |
|---|---:|---:|---:|---:|---:|---:|
| Colour/texture NN | 0.8145 | 0.0629 | — | — | — | — |
| Global ResNet-18 NN | 0.8227 | 0.0657 | — | — | — | — |
| Mid-scale patch, 2,048/category | 0.8283 | 0.0477 | 0.9488 | 0.3038 | **0.2507** | 0.7156 |
| Mid-scale random, 512/category | 0.7615 | 0.0433 | 0.9372 | 0.2562 | 0.1605 | 0.7005 |
| Multi-scale random, 512/category | 0.7465 | 0.0460 | 0.9427 | 0.2173 | 0.1081 | 0.7226 |
| Multi-scale bounded random, 512/category | 0.7506 | **0.0411** | 0.9434 | 0.2363 | 0.0893 | 0.7156 |
| Multi-scale k-center, 512/category | **0.8679** | 0.0471 | **0.9555** | **0.3210** | 0.1665 | **0.7692** |
| Unified balanced k-center, 1,536 total | 0.7936 | 0.0566 | 0.9257 | 0.2227 | 0.1087 | 0.6942 |

Reducing memory without representative selection substantially hurts image
ranking. Multi-scale features become effective only when paired with k-center
selection. The coreset improves image AUROC by 0.0396 and AUPRO by 0.0536 over
the 2,048-entry reference while cutting total category-index storage by roughly
60%; the reference retains better thresholded IoU. The unified index is compact
but loses 0.0743 image AUROC relative to category-specific memory.

Normal augmentation changes clean image AUROC from 0.8679 to 0.8674 and lifts
AUPRO from 0.7692 to 0.7762, so its robustness benefit is not purchased with a
material clean-ranking loss.

## 6. Robustness and generalization

Blur is the largest mean degradation: image AUROC changes by -0.0763 and normal
FPR by +0.4730. Noise follows at -0.0572 and +0.3709. JPEG changes image AUROC by
-0.0306 while increasing normal FPR by +0.2720. Photometric change has little
mean image-AUROC effect but can increase anomaly misses, illustrating why a
single aggregate delta is insufficient.

H2 is inconclusive: the multi-scale robustness advantage is -0.0072 with 95% CI
[-0.0235, 0.0083]. The equal-budget augmentation intervention is supported at
+0.0077 shifted image AUROC, 95% CI [0.0030, 0.0126]. Retained-correct decisions
have heatmaps 0.2153 more correlated with clean evidence than failures, 95% CI
[0.2066, 0.2253], supporting H5. Geometric heatmap correlation averages only
0.4416 after inverse alignment.

Leave-one-category-out transfer obtains image AUROC 0.5783, 0.2896 below the
category-specific system. A known category and verified category memory are
therefore operating requirements, not optional metadata.

## 7. Calibration and selective review

On clean data, the primary category-conformal policy obtains normal FPR 0.0430,
review rate 0.1958, automatic coverage 0.8042, and selective risk 0.1940.
Category calibration improves absolute 5% false-alarm alignment over pooled
calibration by 0.0681, 95% CI [0.0050, 0.1809], supporting H3. The review band
reduces automatic risk by 0.0713, 95% CI [0.0606, 0.0812], over 156
category-condition units while mean coverage remains 0.7790, supporting H4.

Clean calibration does not survive arbitrary shift. Severe blur, noise, JPEG,
and geometry produce normal FPR 0.8686, 0.6820, 0.4696, and 0.2179. This is an
exchangeability failure: p-values calibrated on clean normal scores cannot
guarantee false-alarm control when the capture distribution changes.

## 8. System and reproducibility

The local CPU workstation loads one frozen backbone and verified category
indexes, accepts up to 24 images, and displays the centered model input,
continuous heatmap, frozen support contour, score, category-conformal p-value,
policy decision, thresholds, and timings. It binds to loopback, decodes images
in memory, and exports metadata-only reports. Median end-to-end latency is
77.3 ms, p95 is 108.8 ms, throughput is 12.55 sequential images/s, peak process
working set is 548.7 MB, and runtime artifacts total 49.97 MB.

Every run records resolved configuration, seed, environment, Git state,
manifest, model identity, predictions, metrics, and output hashes. M7 pins five
completed run directories and verifies their metadata and required-file hashes
before regenerating public tables and figures. The public package excludes data,
private images, weights, and feature banks.

## 9. Limitations and responsible use

VisA is a research benchmark and does not represent every production sensor,
line speed, material, defect cost, or prevalence. Controlled corruptions are
proxies, not a validated plant distribution. Heatmaps identify feature-space
difference, not a physical cause. Category transfer is weak. Thresholded
localization remains sensitive to the frozen operating point. CPU benchmarks
come from one workstation, and no prospective factory trial is claimed.

The system should be used only with a known category, verified memory and
calibration artifacts, controlled acquisition, visible human review, and shift
monitoring. A detected capture change should pause automatic use until the
capture path is corrected or calibration is revalidated with approved normal
data.

## 10. Conclusion

A bounded multi-scale k-center memory is a strong and practical clean baseline,
and equal-budget normal augmentation produces a small but statistically
supported robustness gain. Category-conditional calibration and selective
review improve clean operating behavior. The dominant result, however, is the
boundary: capture shift can overwhelm clean false-positive control, spatial
evidence can destabilize, and unseen-category transfer is unreliable. Reporting
those boundaries alongside accuracy makes the system more useful as rigorous
decision-support research and less likely to be mistaken for a deployment
guarantee.

## References

1. Zou, Y. et al. “SPot-the-Difference Self-Supervised Pre-training for Anomaly
   Detection and Segmentation.” ECCV, 2022. VisA project and dataset:
   <https://github.com/amazon-science/spot-diff>.
2. Roth, K. et al. “Towards Total Recall in Industrial Anomaly Detection.” CVPR,
   2022. <https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html>.
3. He, K. et al. “Deep Residual Learning for Image Recognition.” CVPR, 2016.
   <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>.

