# Manual checks: CI repair and quality-gate comparison

Run these from the repository root in PowerShell. No AWS account, new dataset,
dependency installation, or retraining is needed on this configured workstation.

## 1. Check the exact CI scope

```powershell
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\ruff.exe check .
.venv\Scripts\mypy.exe src scripts
.venv\Scripts\pytest.exe -q -p no:cacheprovider
```

All four checks should pass. In particular, type checking must include `scripts`,
not just `src`. M10/M11 report regeneration should retain their published package
hashes in `reports/m10/manifest.json` and `reports/m11/manifest.json`:

```powershell
.venv\Scripts\python.exe scripts/build_m10_report.py
.venv\Scripts\python.exe scripts/build_m11_report.py
```

Expected package SHA-256 values:

- M10: `c095cec6d6fc1faf69766172f7ca0ab9ff3ce5837ec9da981ef61319c8f93537`.
- M11: `9d62d190e57010604609141c43620817bc5b5c19c44d793793c4bf8ce2e3481b`.

## 2. Inspect the complete gate results

Read `reports/M12_QUALITY_GENERALIZATION.md` and the aggregate CSVs in
`reports/m12/`. Compare BOTH datasets, clean normals, clean defects, every shift,
remaining automatic anomalous calls, and total operator action. Interception is
not successful product classification. A recapture is unresolved work, not a
false alarm magically fixed. Some conditions/categories can regress.

Rebuild the hash-pinned evidence with:

```powershell
.venv\Scripts\python.exe scripts/build_m12_report.py
```

Repeated builds should return an identical package hash. Changed/missing source
files, wrong source metadata, smoke runs, or incomplete coverage must be refused.
The completed package hash is
`263b970acd02458ae2495bf0916a3199bc0fc6d8b39e53f4cdf692ffe996c3a9`.

## 3. Try the v2 comparison inspector

```powershell
.venv\Scripts\python.exe scripts/prepare_m12_manual_examples.py
.venv\Scripts\python.exe scripts/run_m12_prototype.py --open-browser
```

The first command prints an ignored local folder containing five PNGs and
`expected.json`. Cases use the first official test normal/anomaly in the candle
manifest, severe blur/noise applied to that same normal, and a blank capture.
They are demonstration examples, not new validation data. Do not commit or
publish these licensed image files. Expected outputs are computed from the
verified engine, not invented pass/fail guarantees for every degraded image.

Open `http://127.0.0.1:8767`, select `candle`, and upload the five PNGs. Compare
the quality status, decision, and `model_evaluated` with `expected.json`.
For recaptures, expect NO anomaly score or heatmap, and zero feature-extraction
and memory-scoring time. A passed capture may still be normal, review, or
anomalous; passing the gate is not a product-quality verdict.

The fixed candle examples deliberately are not filtered for attractive outcomes.
In the verified run, `clean-normal.png` is recaptured and `clean-anomaly.png`
receives a normal model decision despite its original anomalous source label.
The other three cases are recaptured. These expose a clean rejection and an
image-level miss; matching `expected.json` verifies reproducible behavior, NOT
correct product classification. Centered crops may also omit source evidence.

Inspect `http://127.0.0.1:8767/api/health`: expect `quality_schema_version: 2`,
`quality_alpha: 0.02`, and a full `m12-quality-generalization` source quality run.
The anomaly model remains ResNet and the localization policy remains
`raw_cleaned`. KSDD2 is not a selectable browser category.

Export the inspection report. It should identify quality schema 2, preserve
recapture states, and contain no preview/heatmap data URLs or source image pixels.

Historical comparison is still available on a different port:

```powershell
.venv\Scripts\python.exe scripts/run_m10_prototype.py --port 8766
```

That inspector must report quality schema 1. The two inspectors may differ in
capture disposition; where both pass, the same detector score/p-value and cleaned
support should agree. Do not stop an existing unrelated server to free a port;
choose another port with `--port` instead. Stop your own server with Ctrl+C.

## 4. Optional: verify interruption and resume

```powershell
.venv\Scripts\python.exe scripts/run_m12_quality_generalization.py --quick --stop-after-conditions 1
# Copy the exact directory printed by the previous command:
.venv\Scripts\python.exe scripts/run_m12_quality_generalization.py --quick --resume "D:\industrial-anomaly-artifacts\runs\<printed-smoke-run>"
```

The first command reports `running` after one committed condition. The second
reuses that checkpoint and finishes the remaining conditions. Smoke results
must not be published as full evidence. Changed code/config/source artifacts
must cause resume to refuse reuse. The trusted dataset and source-artifact
verification can take longer than the initial gate calculations.
