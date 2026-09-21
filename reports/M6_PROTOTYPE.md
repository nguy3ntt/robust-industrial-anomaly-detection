# M6 practical prototype report

## Delivered system

M6 provides a self-contained local browser interface for the frozen augmented
multi-scale coreset and category-conditional M5 calibration. It supports one or
multiple images, all 12 VisA categories, in-memory inference, a three-way
decision, anomaly score, conformal p-value, score and pixel thresholds, spatial
heatmap, batch table, timing, visible limitations, and metadata-only export.

The implementation uses the Python standard-library HTTP server and checked-in
HTML/CSS/JavaScript. It adds no runtime dependency and makes no network request.
The server permits loopback binding only.

## Runtime contract

- Source calibration: `m5-calibration-full-20260920T110316Z`.
- Primary model: `multiscale_augmented_coreset_512`.
- Decision regions: anomalous at p ≤ 0.05, review at 0.05 < p ≤ 0.20,
  normal at p > 0.20.
- Preprocessing: EXIF orientation followed by the evaluated centered 224×224 crop.
- Explanation: smoothed nearest-memory patch distance, white contour at the
  frozen clean-validation pixel threshold.
- Persistence: no uploaded or derived image is written by the application.

Startup verifies the M5 summary/calibration hashes, M5→M4 provenance, M4 index
manifest, every category index, and the frozen ResNet-18 checkpoint. The backbone
is loaded once; category memories load on first use and remain cached.

## Privacy-preserving reports

The browser sends completed result objects to `/api/report`. The server rebuilds
the report from an explicit field allow-list. Preview and heatmap data URLs are
discarded even if a modified client includes them. Reports contain policy,
provenance, limitations, filenames, decisions, scores, p-values, thresholds,
dimensions, flagged-area fraction, and timings, but no image pixels.

## Functional and visual verification

A real VisA candle test image was inspected through both the HTTP API and live
browser UI. Both returned `normal`, anomaly score 0.4639, conformal p-value
0.3309, and 0.4% thresholded heatmap support. The metadata report contained one
result, declared `source_images_embedded=false`, and contained no data URL.

Visual QA covered the 319 px in-app viewport. The initial result table exposed a
minimum-width overflow; grid tracks and result cards were constrained and the
retest measured body width 311 px within a 319 px viewport. Decision metrics,
heatmap, all limitation text, and the batch table are now readable without page
horizontal scrolling.

## CPU benchmark

The authoritative run `m6-prototype-benchmark-20260921T055903Z` inspects two
deterministically selected official-test images from each of 12 categories.
Inputs are read in memory and are not copied into the benchmark artifacts.

| Measurement | Result |
|---|---:|
| Images / categories | 24 / 12 |
| Startup | 323.6 ms |
| Mean end-to-end | 73.5 ms/image |
| Median end-to-end | 73.8 ms/image |
| P95 end-to-end | 103.6 ms/image |
| Mean feature extraction | 12.5 ms/image |
| Mean memory scoring | 3.7 ms/image |
| Sequential throughput | 13.46 images/s |
| Peak process working set | 547.8 MB |
| Python traced allocation peak | 16.2 MB |
| Backbone checkpoint | 46.83 MB |
| All 12 category indexes | 2.75 MB |
| Calibration artifacts | 0.39 MB |
| Total runtime artifacts | 49.97 MB |
| Self-contained UI assets | 24.4 KB |

The working set includes the Python interpreter, imported scientific/PyTorch
runtime, resident ResNet-18, current image tensors, and cached category indexes.
Timings are CPU workstation measurements and depend on filesystem cache,
hardware, threading, and image decoding.

## Limitations retained in the interface

- This is decision support, not certified quality control.
- Clean calibration assumes exchangeability and does not guarantee shifted FPR.
- Severe blur, noise, JPEG, and geometry can produce confident false alerts.
- Product category must be known; category-agnostic transfer is inadequate.
- A heatmap is spatial evidence, not physical root-cause diagnosis.

These statements appear next to each result and in every exported report rather
than being confined to project documentation.
