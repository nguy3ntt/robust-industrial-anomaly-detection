# System card — local calibrated inspection prototype

## Purpose and boundary

The M8 prototype extends the completed M6 research system with a pre-inference
capture gate. It accepts one or more images for a known category and returns an
anomaly score, conformal p-value, three-way recommendation, heatmap, threshold
references, timing, and limitations—or a `recapture` disposition without running
the anomaly model when the capture is out of profile.

It is local decision support, not autonomous industrial control. A human remains
responsible for interpreting review/anomaly results and for responding to input
quality or distribution shift.

## Architecture and data flow

The recommended entry point is `start-inspector.ps1`, backed by
`scripts/run_inspector.py`. It verifies local weights and model references,
fails clearly on occupied ports, binds only IPv4 loopback, and never kills an
existing service or chooses a fallback model. The runtime constructs the
backbone from verified local weights without a downloader path; historical
feature values are unchanged. See [startup/recovery](STARTUP.md).

The [portable release](PUBLIC_DEMO_RELEASE.md) provides an alternative local
initialization path using verified relative-path runtime references. It inherits
the same inspection and server behavior rather than implementing a substitute
model. A dependency-free verifier checks the manifest; explicit `--prepare`
may fetch the official hash-pinned checkpoint, which is not distributed inside
the archive. Normal startup never downloads or retrains. No release is uploaded
or exposed as a public hosted service automatically. Its stored VisA-derived
normal references are distinct from uploads, which remain transient in memory.

1. A standard-library HTTP server binds to `127.0.0.1` by default.
2. The browser sends each selected image as a bounded raw request.
3. The server decodes the image in memory, applies EXIF orientation, and uses the
   same centered 224×224 crop as evaluation.
4. The verified M8 gate measures capture properties and calculates a
   category-conditional validation-normal conformal p-value.
5. Failed captures return `recapture`; the remaining inference steps are skipped.
6. For passed captures, one resident frozen ResNet-18 extracts multi-scale features.
7. The requested category's verified 512-entry index is loaded lazily and cached.
8. The image score and spatial score map are computed by nearest-memory distance.
9. Verified M5 validation-normal scores produce a category conformal p-value and
   normal/review/anomalous decision.
10. The browser renders the centered input, quality evidence, heatmap overlay,
   metrics, limitations,
   and batch table.

Inference is serialized behind a model lock because the HTTP layer is threaded.
The backbone remains resident and category memories are cached after first use.

## Privacy and security properties

The optional [fresh-capture pilot](FRESH_CAPTURE_STUDY.md) is a separate
local workflow, not browser upload storage. Its intentionally retained source
photographs, session/object metadata and checkpointed predictions live in an
ignored/private study directory. It locks model artifacts, calibration, code,
installed versions and development membership before final collection, then
records first holdout opening and permits exact verified resume only. Local
hashes/owner attestations are not access control or independent provenance
certification. No fresh captures have been supplied or evaluated yet. The
workflow is for matching deployed product references, not new-category training
or mask-based localization validation, and makes no automatic public export.

- Loopback binding is enforced; non-loopback hosts are rejected.
- Uploaded image bytes and rendered previews are held only in memory.
- The server does not create upload, preview, or history files.
- Export uses a fixed allow-list and removes original and derived image data.
- Filenames are reduced to bounded basenames before display or export.
- Per-image bytes, decoded pixel count, animation, and supported image decoding
  are validated before inference.
- Responses set no-store, content-type, referrer, framing, and restrictive
  content-security headers.
- The UI has no CDN, analytics, font, script, or network dependency.
- Model, calibration, source-run, index, and backbone hashes are verified before
  the server becomes ready.

The server is a research prototype and has no user authentication. It must stay
on loopback and should not be reverse-proxied or exposed to a network.

## Human oversight

- The operator must select the correct category.
- The review state is a required outcome, not an error to suppress.
- The recapture state is also a required outcome; it must not be relabelled as a
  normal or anomalous product decision.
- The UI keeps the exchangeability and shift limitations adjacent to every result.
- A heatmap is supporting evidence, not a causal explanation.
- Severe blur, noise, compression, geometry change, or unexpected score drift
  should trigger capture investigation or recalibration rather than automatic
  threshold adjustment.
- Independent KolektorSDD2 validation finds that the current quality gate misses
  many blur, noise, and framing stress cases despite normal-only recalibration.
  It screens measured departures; operator capture checks remain necessary.

## Failure handling

- Missing or modified artifacts prevent startup.
- Unsupported, empty, oversized, highly decompressive, or animated images are
  rejected with a bounded client error.
- A failed item stops the current browser batch and leaves completed results in
  memory only; the operator can correct the input and retry.
- No fallback silently changes model, category, or thresholds.
- Server logs contain request status and internal exception type, but no image
  payload or full local source path.

## Resource profile

The authoritative 24-image CPU benchmark is
`m6-prototype-benchmark-20260921T065622Z`:

- startup: 282.2 ms on the measured warm-filesystem run;
- median / p95 end-to-end: 77.3 / 108.8 ms per image;
- sequential throughput: 12.55 images/s;
- peak process working set: 548.7 MB;
- total model, index, and calibration artifacts: 49.97 MB;
- all 12 category indexes: 2.75 MB;
- served self-contained HTML/CSS/JavaScript: 35.3 KB.

Measurements are workstation- and cache-dependent. The benchmark saves exact
environment, input IDs, per-stage timings, configuration, provenance, and hashes.

The M8 gated benchmark `m8-prototype-benchmark-20261013T034435Z` measures a
2.46 ms mean quality-gate stage, 81.3 ms median and 130.4 ms p95 total latency,
and 11.88 images/s across the same deterministic 24 inputs. Four captures skip
neural inference, so the aggregate latency is a mixed-path workflow result. The
quality artifact adds 48,424 bytes, bringing total runtime artifacts to
50,014,739 bytes.

## Operational checklist

M11 is an offline encoder comparison and does not change the deployed model,
quality gate, thresholds, browser behavior, or report privacy. The matched VisA
CPU benchmark measures 33.9 ms median for ResNet versus 54.1 ms for ConvNeXt;
it includes dataset image/mask decoding and map smoothing, not browser rendering
or quality gating, so it is not a replacement M10 application-latency figure.
The full ConvNeXt checkpoint is 114,419,221 bytes and its executed three-stage
encoder has 12,348,000 parameters versus ResNet's 2,782,784. The larger/slower
substitution loses clean ranking/localization and is not automatically promoted.
See [M11 results](../reports/M11_ENCODER_COMPARISON.md) and
[manual verification](M11_MANUAL_TESTING.md).

The optional M10 inspector retains these safety/privacy contracts and adds
hash-verified small-region cleanup with region-count/policy/threshold metadata.
Its 24-input CPU benchmark `m10-prototype-benchmark-20261014T113701Z` records
73.2 ms median, 106.1 ms p95, and 0.59 ms mean localization stage time (20 neural
evaluations, four recaptures). Localization time is included in rendering/total
time. Profiles add 4,018,719 bytes; total runtime artifacts are 54,033,458 bytes.
These mixed-path measurements do not establish a speed gain over older runs.
See the [manual verification guide](M10_MANUAL_TESTING.md).

M12 provides a separately configured schema-2 quality-gate comparison on port
8767. The ResNet detector, anomaly calibration, contour policy, loopback/upload
limits, and image-free export remain unchanged. Runtime health and exported
metadata identify `quality_schema_version` and the source quality run. Rejected
v2 captures still skip all neural inference and remain unresolved. Historical
M8/M10 configurations retain schema 1; no existing server is silently restarted
or replaced. KSDD2 remains research-only rather than a browser category.
The [M12 evidence](../reports/M12_QUALITY_GENERALIZATION.md) reports gate-only
offline timings and both gains/regressions; those timings do not replace the
historical end-to-end application benchmarks. Use the
[M12 manual guide](M12_MANUAL_TESTING.md) to compare both profiles locally.

M13 runs a separate, local-only controlled resolution/tiling evaluation. Its
four-view input geometry, common-map image-score aggregation, normal references
and calibration differ from the current browser deployment; generated panels
must not be uploaded as substitutes for source inputs. No existing server is
restarted or automatically promoted. The panel generator verifies its displayed
binary support/IoU against committed measured outputs, with source labels and
model decisions shown separately. Dataset images and per-image evidence remain
in ignored local artifacts; public reports contain aggregates only. Read the
[M13 manual guide](M13_MANUAL_TESTING.md) and [results](../reports/M13_SMALL_DEFECT_LOCALIZATION.md)
before interpreting any fine-looking contour or small-region hit as an accuracy
gain. Offline timings are not a new browser/end-to-end benchmark.

- Keep `.env`, dataset, model artifacts, and reports on approved local storage.
- Start on the default loopback address.
- Confirm the selected category and capture quality before acting on a result.
- Treat review cases as unresolved until a person evaluates them.
- Retain metadata reports only under the applicable data-governance policy.
- Re-run the benchmark and verification suite after dependency, model, or hardware
  changes.
