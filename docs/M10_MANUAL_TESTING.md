# Verify thresholded localization

Run these checks from the repository directory using its existing environment.
No AWS account, new dataset, or training run is needed for the live inspector.

## Start and inspect

```powershell
.venv\Scripts\python.exe scripts/run_m10_prototype.py --port 8766
```

Open <http://127.0.0.1:8766/>. Leave the terminal open. If port 8766 is already
running the M10 inspector, use that page without starting a second server.
The default port is 8765 if no port is specified; an older inspector running
there does not automatically change to M10.

1. Select **candle** and upload
   `D:\industrial-anomaly-data\sources\visa-20220922\candle\Data\Images\Normal\0024.JPG`.
   This verified test image passes the gate and produces a heatmap. Confirm the
   input, evidence view, decision, and flagged area appear without errors.
2. Export the inspection JSON. Each evaluated result should include
   `localization_policy: "raw_cleaned"`, `localized_regions`,
   `localization_threshold_units: "memory-distance"`, and flagged area in
   `anomalous_area_fraction`. Runtime provenance should name
   `m10-localization-full-20261014T113141Z`. The export must not include
   `preview_data_url`, `heatmap_data_url`, or image bytes.
3. Upload another image from the same product category. Do not expect every
   dataset-labelled defect to receive an anomalous decision: the measured
   detector has misses, and the contour is unusual evidence, not ground truth.
4. Upload a completely blank image under candle. Expect **recapture**, no
   heatmap, `model_evaluated: false`, and no evaluated region count. Rejection
   still happens before neural inference.

## Compare with the original inspector

```powershell
.venv\Scripts\python.exe scripts/run_m8_prototype.py --port 8767
```

Open <http://127.0.0.1:8767/> and upload the same files with the same category.
For accepted inputs, compare the exported JSON: anomaly score, conformal p,
decision, quality status, and pixel threshold must match M10. M10's flagged
area can only stay equal or shrink; larger components retain their original
boundary. Tiny islands may disappear, but many images will look identical.
This is deliberately a small refinement. An image can have an outlined region
and still receive a normal image-level decision: these are different operating
points and measurements.

## Automated and research checks

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe scripts/build_m10_report.py
.venv\Scripts\python.exe scripts/benchmark_m6_prototype.py --config configs/prototype/m10.yaml
```

The suite currently has 83 passing tests. Report reconstruction should print
package SHA-256 `c095cec6d6fc1faf69766172f7ca0ab9ff3ce5837ec9da981ef61319c8f93537`.
It verifies the complete experiment's saved output hashes before publication.
The report must contain all five policies, 12 VisA categories plus KSDD2, and
negative results, not only the deployed winner. Test images, per-image support
measurements, fitted references, and banks remain outside Git.

The recorded CPU benchmark is `m10-prototype-benchmark-20261014T113701Z`:
73.2 ms median total latency, 106.1 ms p95, and 0.59 ms mean localization stage
over 24 deterministic inputs (20 evaluated, four recaptures). Localization's
stage time is included in rendering/total time, not an extra additive stage.
The profiles add 4,018,719 bytes. Hardware and cache state affect reruns; comparing
separate benchmark dates does not establish a speed improvement.

## Experiment resume

New full studies use `scripts/run_m10_localization.py`; `--quick` is only a
development subset and is not publication evidence. Interrupted runs require
the same configuration, code, sources, filters, and reference-run choice. The
authoritative run reused verified normal profiles from the completed comparison:

```powershell
.venv\Scripts\python.exe scripts/run_m10_localization.py `
  --resume D:\industrial-anomaly-artifacts\runs\m10-localization-full-20261014T113141Z `
  --reference-run D:\industrial-anomaly-artifacts\runs\m10-localization-full-20261014T112455Z
```

It is already complete; you do not need to rerun this for manual UI checks.
