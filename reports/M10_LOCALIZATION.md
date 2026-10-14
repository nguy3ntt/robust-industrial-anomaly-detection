# M10 thresholded localization

## Outcome

Removal of small isolated regions at the existing normal-calibrated threshold changes VisA
macro visible-defect IoU from 0.1194 to
0.1196, a paired category-mean difference of
+0.0002 (95% category-bootstrap interval
[-0.0001, +0.0004]). Clean normal pixel
false positives change from 0.483% to
0.478%. The macro fraction of clean normal images with
any outlined region drops from 78.17% to
73.43%; this is contour incidence, not the
image-level anomaly false-alarm rate. Foreground recall changes from
0.6294 to 0.6289, including
small category-specific losses. On KolektorSDD2, visible-defect IoU changes from
0.2868 to 0.2868, with paired
image-bootstrap difference interval [+0.0000, +0.0000].

These are measured binary contour results, separate from image-level detection
and continuous-map ranking. Image anomaly scores, M5 decisions, memory banks,
backbone weights, and the M8 capture gate remain unchanged.

## Fixed method and calibration

Up to 256 deterministic training normals estimate per-location score median and
MAD variation. A robust scale floor prevents unstable division at quiet pixels.
Separate validation normals establish 99th-percentile weak and 99.5th-percentile
strong boundaries. A monotone search can raise the strong threshold until the
region-growth calibration respects a 0.5% mean normal-pixel support budget.
The primary contour keeps the original score map and calibrated threshold and removes
regions smaller than 0.03% of the map. Spatial normalization and weak-evidence growth are separate
comparisons; growth is limited to half one feature-lattice cell (8 pixels at 224x224,
2 at 56x56). Numerical calibration never uses defect masks.

The initial engineering smoke prototype allowed unbounded growth. A weak bridge
could spread its region far from strong evidence, so the full protocol adds an
architecture-derived radius bound. The primary policy retains the existing evidence boundary;
spatial normalization and growth remain measured comparisons. The full results are comparative
evidence after development, not a fresh blind industrial acceptance test.

See the [frozen protocol](../docs/M10_LOCALIZATION_PROTOCOL.md) and
`configs/experiments/m10-localization.yaml` for the exact choices.

## Clean ablation

| Dataset | Policy | Defect IoU | Dice | Precision | Recall | Normal pixel FPR | All-image IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| visa | Existing global threshold | 0.1194 | 0.1895 | 0.1514 | 0.6294 | 0.48% | 0.1674 |
| visa | Existing threshold + cleanup | 0.1196 | 0.1896 | 0.1514 | 0.6289 | 0.48% | 0.1871 |
| visa | Raw seeded growth | 0.1089 | 0.1717 | 0.1291 | 0.6248 | 0.49% | 0.2915 |
| visa | Spatial seed + cleanup | 0.1079 | 0.1733 | 0.1412 | 0.5526 | 0.46% | 0.2000 |
| visa | Spatial seeded growth | 0.0988 | 0.1571 | 0.1204 | 0.5435 | 0.47% | 0.2917 |
| ksdd2 | Existing global threshold | 0.2868 | 0.4174 | 0.3477 | 0.7617 | 0.41% | 0.2672 |
| ksdd2 | Existing threshold + cleanup | 0.2868 | 0.4174 | 0.3477 | 0.7617 | 0.41% | 0.2672 |
| ksdd2 | Raw seeded growth | 0.2699 | 0.3961 | 0.3097 | 0.7742 | 0.38% | 0.5562 |
| ksdd2 | Spatial seed + cleanup | 0.2756 | 0.4056 | 0.3424 | 0.7622 | 0.37% | 0.3287 |
| ksdd2 | Spatial seeded growth | 0.2624 | 0.3886 | 0.3073 | 0.7841 | 0.35% | 0.5982 |

Defect IoU/Dice/precision/recall are means over anomalous images with visible
positive masks. Empty normal masks never earn credit in these metrics. The
legacy all-image IoU remains separate so normal-empty success cannot conceal
defect-localization failures. VisA uses 224x224 centered geometry; KolektorSDD2
retains M9's full-frame input and 56x56 localization grid. Compare policies
within a dataset, not absolute localization values across these grids.

## Category behavior

| Category | Existing defect IoU | Cleaned defect IoU | Paired delta | 95% paired interval |
|---|---:|---:|---:|---|
| visa/candle | 0.0940 | 0.0948 | +0.0007 | [-0.0003, +0.0025] |
| visa/capsules | 0.0988 | 0.0978 | -0.0009 | [-0.0034, +0.0001] |
| visa/cashew | 0.0898 | 0.0900 | +0.0002 | [+0.0001, +0.0004] |
| visa/chewinggum | 0.2199 | 0.2205 | +0.0006 | [+0.0004, +0.0009] |
| visa/fryum | 0.1368 | 0.1369 | +0.0001 | [-0.0000, +0.0003] |
| visa/macaroni1 | 0.0728 | 0.0728 | -0.0000 | [-0.0003, +0.0002] |
| visa/macaroni2 | 0.0391 | 0.0387 | -0.0004 | [-0.0020, +0.0005] |
| visa/pcb1 | 0.1232 | 0.1236 | +0.0004 | [+0.0003, +0.0006] |
| visa/pcb2 | 0.1288 | 0.1292 | +0.0004 | [-0.0003, +0.0011] |
| visa/pcb3 | 0.1548 | 0.1552 | +0.0004 | [-0.0008, +0.0015] |
| visa/pcb4 | 0.1592 | 0.1594 | +0.0002 | [+0.0000, +0.0003] |
| visa/pipe_fryum | 0.1158 | 0.1158 | +0.0000 | [+0.0000, +0.0001] |
| ksdd2/surface | 0.2868 | 0.2868 | +0.0000 | [+0.0000, +0.0000] |

Category gains and losses remain visible. Small defects, object displacement,
or spatially variable normal backgrounds can change precision and recall in
different directions. A contour is unusual spatial evidence, not a physical
defect boundary or root-cause diagnosis.

## Severe capture shifts

| Dataset | Condition | Existing defect IoU | Cleaned defect IoU | Existing normal pixel FPR | Cleaned normal pixel FPR |
|---|---|---:|---:|---:|---:|
| ksdd2 | blur_severe | 0.1410 | 0.1410 | 8.13% | 8.13% |
| ksdd2 | clean | 0.2868 | 0.2868 | 0.41% | 0.41% |
| ksdd2 | geometric_severe | 0.2351 | 0.2351 | 1.57% | 1.57% |
| ksdd2 | jpeg_severe | 0.1494 | 0.1494 | 5.92% | 5.92% |
| ksdd2 | noise_severe | 0.0936 | 0.0936 | 17.73% | 17.73% |
| visa | blur_severe | 0.0348 | 0.0348 | 26.30% | 26.30% |
| visa | clean | 0.1194 | 0.1196 | 0.48% | 0.48% |
| visa | geometric_severe | 0.0928 | 0.0930 | 1.56% | 1.55% |
| visa | jpeg_severe | 0.0615 | 0.0616 | 10.56% | 10.54% |
| visa | noise_severe | 0.0400 | 0.0399 | 27.86% | 27.84% |

Clean calibration does not guarantee shifted control. Spatial references assume
stable capture geometry and may degrade under displacement or image corruption.
Offline support is measured for every official test image to reveal this
failure boundary, including captures the live gate would reject.

## Runtime and reproducibility

The M10 configuration displays contours from the exact evaluated support
function and includes memory-distance threshold units, region count, policy,
and flagged area in metadata-only exports. Startup verifies all spatial
artifact hashes, original memory identity, trusted manifest, and input geometry.
The M6/M8 configurations retain their original pixel policy for reproduction.

The authoritative run is `m10-localization-full-20261014T113141Z`. It covers 13 dataset/category units,
five conditions, and five policies, with 325 aggregate rows. All source images,
per-image support measurements, model weights, spatial numeric artifacts, and
feature memories remain in local artifact storage. Public aggregate tables and
hashes are under [m10/](m10/). Checkpointed training references and conditions
permit verified resume. Provider splits are unchanged.

The spatial alternatives were retained for comparison after their development
results showed worse visible-defect overlap in several categories. The deployment
uses component cleanup at the original threshold. This is a measured engineering
refinement rather than a claim that spatial normalization is a superior detector.
The minimum component area rounds to one pixel on KSDD2's 56x56 grid, so cleanup
does not change its baseline support. Removing more would require a different,
independently justified small-defect policy. The VisA change is modest, not a
substantial new localization accuracy gain. Intervals after development/policy
selection are exploratory, not independent confirmation.
