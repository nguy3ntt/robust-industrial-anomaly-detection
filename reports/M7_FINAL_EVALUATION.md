# M7 final evaluation

The final package reconstructs the public evidence from five hash-pinned,
completed reportable runs. All required metadata, source-table, manifest, and
output hashes passed. No dataset image, private image, model weight, or feature
bank is included.

Package SHA-256:
`51f8c4151b82c9701e734ecb93cb2fbd915b8db764c7062b2abd386e0fd2b36f`.

## Main result

The category-specific multi-scale k-center memory is the strongest clean model
for image ranking and spatial ranking. Normal-only augmentation keeps those
clean results essentially unchanged while improving mean shifted AUROC in the
controlled ablation. The calibrated review policy controls clean false alarms,
but severe blur, noise, and JPEG shift invalidate that clean guarantee.

| System | Image AUROC | Image AP | Normal FPR | Pixel AUROC | Pixel AP | Mean IoU | AUPRO@30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mid-scale patch memory | 0.8283 | 0.8527 | 0.0477 | 0.9488 | 0.3038 | **0.2507** | 0.7156 |
| Multi-scale k-center, 512/category | **0.8679** | 0.8876 | 0.0471 | 0.9555 | 0.3210 | 0.1665 | 0.7692 |
| Augmented multi-scale k-center, 512/category | 0.8674 | **0.8907** | 0.0496 | **0.9557** | **0.3226** | 0.1674 | **0.7762** |

Macro means cover all 12 VisA categories. The reference retains the best frozen
thresholded IoU even though the coreset leads threshold-free localization.

## Controlled shift

| Family | Mean delta image AUROC | Mean delta normal FPR | Mean heatmap correlation |
|---|---:|---:|---:|
| JPEG | -0.0306 | +0.2720 | 0.7430 |
| Resize | -0.0140 | +0.0607 | 0.8204 |
| Blur | **-0.0763** | **+0.4730** | 0.6423 |
| Noise | -0.0572 | +0.3709 | 0.6492 |
| Photometric | -0.0032 | -0.0158 | 0.9285 |
| Geometric | -0.0302 | +0.1117 | **0.4421** |

Blur is the largest average detection and false-positive failure. Geometric
change is the largest spatial-stability failure after inverse alignment.

## Calibration and review

The primary category-conformal policy reaches clean normal FPR 0.0430, review
rate 0.1958, automatic coverage 0.8042, and selective risk 0.1940. Severe-shift
normal FPR rises to 0.8686 for blur, 0.6820 for noise, 0.4696 for JPEG, and
0.2179 for geometry. These are failure measurements, not calibrated guarantees.

| Test | Estimate | 95% interval | Outcome |
|---|---:|---:|---|
| H2 multi-scale robustness | -0.0072 | [-0.0235, +0.0083] | Inconclusive |
| Augmentation intervention | +0.0077 | [+0.0030, +0.0126] | Supported |
| H5 explanation stability | +0.2153 | [+0.2066, +0.2253] | Supported |
| H3 category calibration | +0.0681 | [+0.0050, +0.1809] | Supported |
| H4 selective review | +0.0713 | [+0.0606, +0.0812] | Supported |

## Error taxonomy

| Code | Failure | Evidence | Required response |
|---|---|---|---|
| E1 | Capture nuisance false positives | Severe blur/noise/JPEG add 795/647/466 false positives; severe blur FPR 0.869 | Reject degraded capture, monitor drift, and recalibrate only after an approved change |
| E2 | Photometric anomaly misses | Severe photometric shift creates 98 new false negatives and 8 new false positives | Control illumination and retain human review |
| E3 | Spatial evidence instability | Geometric heatmap correlation averages 0.442 | Treat heatmaps as support, not causal diagnosis |
| E4 | Category transfer failure | Leave-one-category-out AUROC 0.578, 0.290 below category-specific memory | Require a known category and its verified normal memory |
| E5 | Exchangeability failure | Severe blur/noise/JPEG normal FPR 0.869/0.682/0.470 | Pause automatic use under detected shift |
| E6 | Localization operating trade-off | Coreset improves AUPRO while frozen mean IoU trails the 2,048-memory reference | Inspect continuous heatmaps and report both metric types |

## Practical prototype

The CPU workstation reaches 77.3 ms median and 108.8 ms p95 end-to-end latency,
12.55 sequential images/s, 548.7 MB peak process working set, and 49.97 MB of
runtime model/calibration artifacts. Reports are metadata-only by default.

## Public files

- `reports/m7/tables/`: clean comparison, robustness, calibration, coverage-risk,
  hypothesis tests, and error taxonomy.
- `reports/m7/figures/`: publication-ready aggregate figures.
- `reports/m7/manifest.json`: input provenance, output hashes, and privacy flags.
- `docs/FINAL_EVALUATION_PROTOCOL.md`: frozen interpretation and rebuild rules.
