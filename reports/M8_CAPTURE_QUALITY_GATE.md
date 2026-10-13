# M8 pre-inference capture-quality gate

## Outcome

M8 adds a fail-closed, category-conditional capture gate before the frozen M6
anomaly pipeline. On all 962 clean official-test normals, the gate rejects 19
captures (1.98%) at its configured 2% calibration level. Across the twelve
frozen M4 shifts, it rejects 89.13% of shifted normals and intercepts 5,072 of
5,441 M5 false alerts (93.22%) before feature extraction.

The result supports the gate as a practical distribution-shift guardrail. It
does not make the anomaly detector invariant to shift and does not establish
production camera fitness.

## Protocol

- Input: the exact centered 224×224 RGB crop consumed by the anomaly model.
- Fit data: official training-normal images only.
- Calibration: the existing deterministic validation-normal subset derived
  only from training-normal samples.
- Evaluation: the untouched official test partition, clean plus the twelve
  frozen M4 compression, resize, blur, noise, photometric, and geometric shifts.
- Threshold: category split-conformal upper-tail p-value; `p <= 0.02` returns
  `recapture`.
- Runtime rule: rejected inputs do not run ResNet-18, memory scoring, anomaly
  calibration, or heatmap generation.

Eight inexpensive measurements form the gate: mean illumination, contrast,
Laplacian focus, high-frequency residual noise, clipped-pixel fraction, JPEG
blockiness, and horizontal/vertical edge centroids. Per-feature centers and
scales use the training-normal median and robust MAD/IQR scale. The nonconformity
score is the largest direction-aware robust deviation; validation-normal scores
calibrate it without defect labels.

## Shift results on official-test normals

| Condition | Rejected | M5 false alerts | Intercepted | Interception |
|---|---:|---:|---:|---:|
| Clean | 2.0% | 191 | 5 | 2.6% |
| Blur, moderate | 100.0% | 488 | 488 | 100.0% |
| Blur, severe | 100.0% | 962 | 962 | 100.0% |
| Geometry, moderate | 75.0% | 209 | 163 | 78.0% |
| Geometry, severe | 82.0% | 491 | 392 | 79.8% |
| JPEG, moderate | 61.0% | 311 | 204 | 65.6% |
| JPEG, severe | 92.0% | 795 | 742 | 93.3% |
| Noise, moderate | 74.0% | 286 | 232 | 81.1% |
| Noise, severe | 100.0% | 889 | 889 | 100.0% |
| Photometric, moderate | 86.0% | 139 | 129 | 92.8% |
| Photometric, severe | 100.0% | 158 | 158 | 100.0% |
| Resize, moderate | 100.0% | 195 | 195 | 100.0% |
| Resize, severe | 100.0% | 518 | 518 | 100.0% |

Clean false alerts are mostly model uncertainty or anomaly-model error, not
capture shift, so a capture gate should not be expected to intercept them. The
low clean interception rate is consistent with that separation of roles.

## Defect interaction and operating interpretation

The gate rejects 205 of 1,200 clean anomalies (17.08%). This is expected when a
defect changes global texture, contrast, or edge distribution, but it creates an
important semantic constraint: `recapture` means unresolved input, not “bad
photo and normal product.” The clean automatic-anomalous rate falls from 51.58%
to 42.33% because rejected items are withheld. When recapture, review, and
anomalous are all counted as required operator action, clean-anomaly actionable
coverage rises from the source model's 75.83% alert rate to 79.50%.

High clean-anomaly rejection is category-dependent, reaching 59% for `pcb2`,
42% for `cashew`, and 36% for `fryum`, while `candle` is 0%. This confirms that
the gate must remain a workflow guardrail rather than a defect-independent
camera diagnostic.

## Runtime and integrity behavior

The new M8 prototype verifies the quality-model SHA-256, canonical manifest
identity, exact M5 source metadata, 224×224 geometry, and category coverage
before serving. A rejected request returns quality score, p-value, leading
contributors, raw capture measurements, and zero feature/scoring time. Reports
retain these metadata but still exclude uploaded and derived image pixels.

The deterministic 24-image CPU benchmark
`m8-prototype-benchmark-20261013T034435Z` measures a 2.46 ms mean gate cost,
81.3 ms median and 130.4 ms p95 total request latency, and 11.88 sequential
images/s. Four inputs were routed to recapture and therefore skipped neural
inference, so the mixed-path total is not a like-for-like replacement for the
M6 all-inference benchmark. The serialized gate is 48,424 bytes; total runtime
artifacts are 50,014,739 bytes.

The authoritative run is
`m8-quality-gate-full-20261013T034123Z`. Its frozen configuration SHA-256 is
`6c975159774ae97edc33af292b5fde54a54acbc9c6ac0a93b62e23d56ecbc283`, its
quality model SHA-256 is
`e991ff067b31003c226f1737904253c656b7719111f735feb7b3b5f3e44662d5` and
its aggregate metric SHA-256 is
`d776ba0222fd30c578bb7a05b12e89df3356a575ad8b6c72758cd2fecb542e03`.

## Limitations

- Passing the gate is not proof of correct capture or product quality.
- Rejection attribution identifies the largest proxy deviations, not a physical
  root cause.
- Category identity must be known and correct.
- The 2% clean-normal rate is an aggregate observation; finite category samples
  vary, from 0% to 6% in this evaluation.
- New cameras, optics, lighting, preprocessing, or product revisions require a
  new normal-only calibration and evaluation rather than threshold editing.
