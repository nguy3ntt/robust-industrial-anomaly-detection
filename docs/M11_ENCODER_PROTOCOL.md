# M11 modern encoder comparison — fixed protocol

Compare frozen ResNet-18 and ConvNeXt-Tiny ImageNet-1K V1 weights with the same
normal-only augmented patch-memory recipe. ConvNeXt is a modern convolutional
encoder supported by the already locked torchvision dependency. This is not a
claim to benchmark the newest encoder family or a DINO/self-supervised model.

## Representation and matched budget

- Fixed 224x224 RGB input and ImageNet mean/std for both encoders.
- First three stages at strides 4/8/16; pool the first to the second-stage grid,
  interpolate the third, concatenate and L2-normalize. Both yield 28x28 patches.
- ResNet uses 448 channels; ConvNeXt uses 672. Each gets a seeded 64-dimensional
  random projection, L2-normalized projected features, a 4,096-candidate bound,
  and a 512-entry greedy k-center bank. Channel-dependent projection storage is
  reported, not hidden by claiming equal total bytes.
- Select at most 256 training normals per category with a seeded sample-ID hash,
  independent of labels beyond the normal-only training requirement. Use exactly
  the same IDs for both encoders; save the selected IDs. This bounds CPU fitting
  cost. It is not a full-training reproduction of historical M4/M5/M9 metrics.
- Each normal contributes 12 clean and 12 mild-augmentation patches, reusing
  the frozen M4 nuisance pool, seed, and sample-key assignment. Candidate sampling,
  bank size, input geometry, top-1%-patch image aggregation, and smoothing match.
- No encoder fine-tuning, layer search, anomaly training, threshold sweep,
  category-specific winner selection, or test-based augmentation changes.

The comparison isolates a practical pretrained representation substitution, not
architecture alone: parameter capacity and the original pretraining recipe also
differ. It intentionally uses common industrial input geometry, not either
checkpoint's classification resize recipe (ConvNeXt's native classification
recipe resizes to 236 before a 224 crop).

## Data and operating policy

Use all 12 audited VisA categories and independently audited KolektorSDD2.
Keep VisA's centered crop and KSDD2's full-frame resize. Use all existing
training-derived validation normals for separate encoder/category calibration.
Preserve every official test image. KSDD2's labelled training defects remain
excluded. Fit and calibration partitions stay disjoint.

Evaluate clean plus the frozen severe JPEG, blur, noise, and geometric shifts.
Smooth at input resolution with sigma 4 then evaluate both encoders on a common
56x56 grid. Resize masks with nearest-neighbour after the identical input
geometry and shifted-mask transform. Count anomalous masks that disappear.
Do not compare these VisA localization numbers directly with M10's 224 grid.

Image scores use category upper-tail split-conformal calibration: p <= .05
anomalous, .05 < p <= .20 review, otherwise normal. Pixel boundaries use the
99.5th percentile of smoothed calibration-normal pixels. Both are normal-only.
Use plain thresholded support for both encoders, without M10 component cleanup
or seeded growth, so a localization-policy change cannot confound the encoder.
No capture gate is applied: it would obscure representation failure on rejected
inputs. This research run does not change the live M10 inspector or deploy a
new encoder automatically.

## Outcomes and uncertainty

Report image AUROC/AP/FPR@95, empirical and conformal FPR, anomaly recall,
selective coverage/risk, pixel AUROC/AP, AUPRO@30, visible-defect IoU/Dice/
precision/recall, normal pixel FPR, and empty-mask counts per category/condition.
Never substitute empty-normal all-image IoU for visible-defect overlap.

Use 1,000 stratified paired image bootstraps for clean AUROC/AP differences,
and paired normal/anomaly image resampling for operating deltas.
Clean comparisons use 1,000 resamples; shifted per-category comparisons use
100 descriptive resamples to bound reporting cost. For VisA macro
comparisons use equal category weight and category-bootstrap intervals;
KSDD2 is a separate domain, not a thirteenth exchangeable VisA category.
Shift summaries are descriptive over four selected severe conditions, not a
population distribution of real factory captures. Existing benchmark reuse and
one random seed limit confirmatory/generalization claims.

Measure batch extraction/scoring costs and a matched single-image latency
benchmark per category (five normals, five anomalies), including decoding,
feature extraction, memory scoring, and map rendering, but excluding disk
publication and metric computation. No browser or gate cost is included. Record
active encoder parameters, checkpoint bytes, bank/projection bytes, and process
working set; process memory includes shared libraries and allocator history.

## Integrity and resume

Weights are downloaded only from official PyTorch HTTPS URLs, checked against
the official filename hash prefix and pinned observed full SHA-256. Load local
numeric weights with `weights_only=True`; never execute remote model code.
Source images, masks, weights, banks, and per-image predictions stay outside Git.
The source/config/manifest/audit/code/environment/weight identities gate resume.
Checkpoint every fitted category/encoder and every evaluated condition; verify
all committed output hashes before reuse. Publish only completed all-category
aggregate tables and a source/output hash manifest.

## Primary sources and licence boundary

- [Official torchvision ConvNeXt-Tiny weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)
- [ConvNeXt paper](https://arxiv.org/abs/2201.03545)
- [Original ConvNeXt MIT code licence](https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE)
- [Torchvision BSD code licence](https://github.com/pytorch/vision/blob/main/LICENSE)

Code licences do not erase ImageNet training-data terms or certify unrestricted
commercial weight rights. Use remains the project's academic/noncommercial
portfolio setting; do not redistribute either checkpoint. This review adds no
dataset acquisition, accounts, uploads, or production certification.
