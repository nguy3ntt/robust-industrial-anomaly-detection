# Industrial anomaly detection — portable local demo

This package runs the evaluated CPU M10 inspector with the original M8 capture
gate and all 12 VisA category references. No dataset, experiment history,
cloud account, credentials, `.env`, GPU or training run is needed.

## Start here

Install **Python 3.12** and **uv 0.12.13**, extract the ZIP, and open a terminal
inside `industrial-anomaly-demo/`. First verify its contents (standard library only):

```powershell
python demo.py --verify
```

For first use, explicitly prepare the pinned CPU dependencies and official
backbone checkpoint, then open the local website:

```powershell
uv run --locked --extra cpu --no-dev python demo.py --prepare --open-browser
```

Initial dependency setup is a substantial download; the separate checkpoint
is about 45 MiB. There are no usage/service fees or account requirements from
this project, but your ordinary network/storage costs still apply. Read
`THIRD_PARTY_NOTICES.md` before preparing the weights. Packages and weights are
not included in the ZIP. `--prepare` is an explicit network opt-in, not training.
Subsequent launches use the same command **without `--prepare`**. `uv` may still
need package downloads if its environment/cache is absent; with setup complete,
run `.venv\Scripts\python.exe demo.py --open-browser` on Windows, or
`.venv/bin/python demo.py --open-browser` on Linux, for a strictly local launch.

Already have the official verified checkpoint? Supply its path instead of
downloading/copying it:

```powershell
uv run --locked --extra cpu --no-dev python demo.py --checkpoint "path/to/resnet18-f37072fd.pth" --check
```

`--check` validates and loads the model but starts no server. A missing/corrupt
checkpoint is an error; no model fallback or automatic repair occurs. The
default cache is `.demo-cache/` in this extracted folder, not your original
research storage. `--prepare` never overwrites a corrupt existing checkpoint.

## Inspect and interpret

The default URL is `http://127.0.0.1:8765`. Keep the terminal running. Ctrl+C
stops this instance. If the port is occupied, explicitly add `--port 8770`;
the application never kills another listener or exposes a public network port.

Choose the **actual matching product category**, then select your own authorized
local JPEG/PNG images. The ZIP deliberately contains no sample photographs.
The UI shows quality evidence, normal/review/anomalous decisions and spatial
support. An unsuitable capture returns **recapture**, skips anomaly inference
and has no anomaly score or heatmap. Review/recapture are unresolved, not proof
that a defect was detected. Small outlines are evidence, not physical diagnosis.

Images/previews are processed in memory, not retained by the server. Exported
JSON omits image pixels. No uploads leave loopback; there is no analytics/CDN.
Do not use private fresh-study holdouts here before their registered evaluation.
The model is product-specific, not a universal detector for arbitrary objects.

## What is included

- MIT-licensed project source, the existing red/dark/light inspection interface,
  fixed runtime settings, and a locked CPU dependency specification.
- VisA-derived numeric normal references: projected memories, calibration scores,
  capture statistics and localization profiles. No per-image IDs or test predictions.
- Aggregate M10 results and checksums, model/system cards and attribution.
- `manifest.json`: every packaged file's SHA-256 and size.

No raw images/masks, dataset archives, pretrained weights, secrets, private
captures, complete research-run directories or machine-specific runtime paths
are included. Derived features are **not** a privacy/anonymization guarantee;
this export is limited to the original public VisA references, not private data.

`runtime/runtime.json` preserves source hashes and run names for provenance.
All runtime paths are relative; moving the extracted folder is supported.
Numeric files use non-pickle loading. Extra/missing/modified runtime or source
files cause verification failure. Use the separately supplied ZIP checksum to
check the download against a trusted published value: the local manifest alone
is not a signature against an attacker replacing the entire package.

Historical research cards/reports may refer to studies or scripts available
only in the full research repository. For this extracted runtime, follow the
commands in this README rather than attempting to rerun those experiments.

## Limits and manual checks

This is academic/portfolio decision support, not certified product inspection.
Capture shift can break calibration; small defects and unfamiliar arrangements
remain difficult. M12's gate revision and M13 tiling are **not** promoted here.
No fresh-capture accuracy study is claimed. Read the report and cards for
measured failures; portable packaging does not improve model accuracy.

1. `python demo.py --verify` should work without model files or ML dependencies.
2. `--check` should refuse missing weights, then pass with a verified checkpoint.
3. After startup, `/api/health` should report `verified_portable_release`,
   quality schema 1, `raw_cleaned`, and image persistence false.
4. A second launch on the same port should refuse without stopping the first.
5. Copy the extracted folder elsewhere and repeat `--check`; no `.env`, data
   directory or old experiment tree should be necessary.

Verified on Windows CPU. The launcher uses cross-platform Python, but other
operating systems require their own locked-wheel and runtime verification;
this package is not a claim of tested support on every machine.

This is a local release candidate until the repository owner publishes it.
No release URL or external hosting is created by the exporter.
