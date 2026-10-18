# Small-defect resolution and overlapping-tile protocol

## Fixed question

Separate measurement resolution from actual feature detail. Compare three
readouts on identical official test IDs and a common field of view:

1. `global56`: one 224 RGB input, 56x56 score readout and independently
   normal-calibrated pixel threshold; lift binary support to the common grid.
2. `global336`: the SAME global encoder/memory/image scores, with 336x336
   readout and its own normal-only pixel threshold. This changes measurement,
   not encoder information; do not call an interpolation gain better features.
3. `tiled336`: four overlapping 224x224 crops of a 336x336 input, separate
   matched-budget normal memory, shared encoder/projection recipe, and weighted
   blending into the common 336x336 grid. This exposes 1.5x finer linear detail
   while reducing each tile's context. It costs four encoder views, not one.

No quality gate hides failures in this study. Existing M10/M12 inspectors,
capture references, memories, and operating thresholds remain unchanged.
This follow-up uses previously examined public benchmarks: its evidence is
exploratory development evidence, not independent confirmation.

## Geometry and ground truth

Preserve VisA's centered square field of view and KSDD2's complete narrow frame.
Use a 336 RGB canvas with bilinear sampling. Create aligned ground-truth masks
by BOX area sampling in floating point and retain any positive coverage, rather
than nearest-neighbour sampling that can erase small positive regions. This
conservatively expands boundaries at subpixel scales; report that convention,
do not imply native-mask precision, and retain source/crop positive counts.
Cropping can still remove defects; higher resolution cannot recover them.

All primary IoU/Dice/recall measurements use this SAME fine ground truth for all
readouts, including coarse support lifted with nearest-neighbour interpolation.
Never drop an anomaly just because its coarse mask is empty. Count coarse-mask
losses separately. Empty fine masks remain in image detection, with foreground
overlap undefined and explicitly counted. Also report the historical-style
coarse-grid overlap separately; never confuse it with common-grid improvement.

Tile origins are (0,0), (0,112), (112,0), (112,112). Each native 28x28 patch
score map is interpolated to its 224-pixel tile. Blend with separable raised
cosine weights, axis floor .05, and normalize by accumulated weights. Edges
remain covered. Apply sigma-6 smoothing AFTER stitching for both models,
equivalent to sigma 4 at 224 relative to the original field of view. Do not
sharpen, choose a tile winner, or change thresholds based on defects.

## Normal-only fit and calibration

Use the existing frozen, local, SHA-256-pinned ResNet-18 first three stages.
Select at most 128 normal training IDs per category by the existing seeded hash
rank; use ALL original normal calibration IDs and ALL official test IDs. This
bounded-fit study is not a numerical reproduction of historical M5/M9/M10/M11.
Both models receive identical fit/calibration IDs, a 512-entry memory,
64-dimensional projection, 4,096-candidate ceiling and 24 patches per original
training image: 12 clean plus 12 from one deterministic mild M4 augmentation.
For tiling the 12 patches are shared across ALL tiles, not multiplied by four.
Projection seed/shape, augmentation identity, and model weights are shared.

Apply shifts to the canonical canvas before extracting either view. Scale blur
sigma and geometric translation from 224 to 336 units; intensity/noise and JPEG
parameters are unchanged. Training uses the already frozen mild augmentation
assignment. Evaluate clean plus fixed severe blur and noise from M4, not every
capture threat. Their physical meaning is relative to the same field of view.

Image scores average the highest 1% of pixels on the unsmoothed COMMON fine map.
This is a new matched image-score recipe, not the historical native-patch score.
Each model calibrates image scores with the original clean normals, p <= .05
anomalous, .05 < p <= .20 review. Pixel thresholds independently use the .995
`higher` quantile of each readout's clean calibration pixels. No area cleanup or
growth removes tiny predictions in this comparison. Thresholded false positives,
normal contour incidence, and operator review are therefore essential outcomes.

## Small-defect outcomes

Define common fine-grid 8-connected ground-truth regions once for every image.
Tiny regions occupy <= .01% of the canvas; small regions occupy > .01% and
<= .1%; large regions occupy > .1%. Additionally report the inclusive <= .1%
small-region cohort. Region hit means >= 10% of its pixels are supported, not
an image-level true positive or acceptable defect boundary. Report region
pixel recall as well as hit rate; broad false contours must not look successful
without the accompanying normal/background pixel FPR and image-level IoU.

Report foreground IoU/Dice/precision/recall without rewarding empty normal masks,
small-image overlap (total fine defect area <= .1%), region-size tables, image
AUROC/AP, conformal false positives/review/coverage, normal pixel FPR, contour
incidence, mask visibility, and measured latency/index/working-set cost. These
are exact thresholded measurements; do not claim new full-resolution pixel
AUROC or AUPRO from a subsample or compare them to old coarse-grid values.
Measure single-image latency separately from batched study throughput.

Use paired image resampling for IoU/FPR differences, conditioning on each fitted
model/calibration. Region-rate intervals cluster by image, not individual
regions. VisA macro intervals resample category differences. Intervals are
exploratory, not simultaneous guarantees or independent capture-session evidence.
Do not promote per-category winners, sweep tile sizes, or retune after results.

Publication guard: do not present a bootstrap interval as informative when fewer
than two independent eligible images/categories exist. Preserve raw frozen-run
conditional estimates locally and show point results, cohort counts and explicit
low-count notes publicly. This guard changes reporting, not model/calibration
selection or the registered size cohorts.

## Reproducibility and manual evidence

Run focused geometry/coverage/region tests and a real-data smoke before the full
study. Deliberately stop after a committed condition and verify `--resume`.
Hash configurations, code, environment, data/audits, weights, fit/calibration
membership, indexes, calibration values, and each condition's predictions and
metrics. Recheck consumed image/mask hashes and all committed outputs before
reuse. Checkpoint publication is atomic; incomplete outputs may be overwritten,
but no source data or completed historical run is changed.

Publish aggregate tables and exact source/output pins only. Images, per-image
results, score maps, and memories stay on configured local artifact storage.
Provide local side-by-side panels for fixed normal and small-defect examples,
with ground truth and measured outputs, including failures. A positive study is
not sufficient by itself to replace the current calibrated live inspector.
