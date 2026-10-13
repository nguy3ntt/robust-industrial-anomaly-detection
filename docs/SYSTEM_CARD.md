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

- Keep `.env`, dataset, model artifacts, and reports on approved local storage.
- Start on the default loopback address.
- Confirm the selected category and capture quality before acting on a result.
- Treat review cases as unresolved until a person evaluates them.
- Retain metadata reports only under the applicable data-governance policy.
- Re-run the benchmark and verification suite after dependency, model, or hardware
  changes.
