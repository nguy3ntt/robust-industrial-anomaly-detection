# Portable demo release: maintainer and manual verification guide

This completes the portable-bundle part of the simpler public demo setup.
The full research checkout still uses `start-inspector.ps1`. External reviewers
can instead use a small standalone ZIP, without D: paths or historical runs.
There is no cloud deployment and no automatic GitHub publication.

## Build a local candidate

Use the existing verified research environment and completed original M10/M8
artifacts. The fixed exporter rejects other datasets, gate versions, policies,
partial category sets and an altered M10 report package.

```powershell
.venv\Scripts\python.exe scripts/build_demo_release.py
```

Default outputs, intentionally ignored by Git:

- `artifacts/releases/industrial-anomaly-demo-m16/` — extracted runnable tree.
- `artifacts/releases/industrial-anomaly-demo-m16.zip` — release candidate.
- `artifacts/releases/industrial-anomaly-demo-m16.SHA256SUMS` — ZIP/manifest checksums.

Existing destinations are never overwritten. To test a second build, supply
`--output artifacts/releases/industrial-anomaly-demo-m16-rebuild`; identical
inputs must produce the same ZIP/manifest hashes. ZIP and NumPy archive
timestamps/order/permissions are normalized. A failed build may leave its
incomplete new destination for diagnosis; use a different destination, not a
destructive overwrite. No expensive training or full inference rerun is needed.

Runtime files are deliberately repacked: normal calibration sample IDs and old
index metadata are removed; numeric bank/projection/reference values and frozen
operating policies are preserved. Public source identities are names/hashes,
never absolute data paths. Inference remains the shared inspector implementation,
not a second approximate demonstration model.

## Reviewer setup

Follow the generated [bundle README](DEMO_RELEASE_README.md). The universal
first-use command, inside the extracted folder, is:

```powershell
uv run --locked --extra cpu --no-dev python demo.py --prepare --open-browser
```

This explicitly permits normal dependency setup and the official checkpoint
download. The ZIP itself contains no third-party weights. If a verified local
checkpoint is available, use `--checkpoint <path>` instead; do not use
`--prepare`. After initial setup, the environment's Python runs fully locally.
Python 3.12/uv 0.12.13 are prerequisites, not bundled installers.

## Verification checklist

1. Compare the ZIP SHA-256 with the supplied checksum from a trusted release
   source. Extract it to a NEW private folder. Run `python demo.py --verify`.
2. Run `--check --checkpoint <verified-resnet-file>` with research environment
   variables unset or deliberately pointing to missing folders. Expect success:
   no experiment tree, dataset or `.env` should be accessed.
3. Move/copy the extracted tree to another path, including a path containing
   spaces, and repeat verification/readiness. Nothing in `runtime/` is rewritten.
4. Start it on an unused port with `--open-browser`. Health identifies the
   portable release and the original quality/localization policies. Check both
   recapture and accepted input paths using your own permitted, non-holdout images.
5. Compare fixed allowed local test inputs against the research inspector.
   Ignore stage timings; all decisions, scores, thresholds, spatial support,
   previews/heatmaps and report fields should match. This is packaging parity,
   not a new accuracy or fresh-capture evaluation.
6. Modify a file in a DISPOSABLE COPY and repeat `--verify`: it must refuse.
   Extra runtime/source files, missing members, unsafe paths, wrong hashes,
   corrupt numeric arrays and a bad checkpoint are errors, not fallback paths.
7. Test an occupied port: the second launch must fail without stopping the first.
   Ctrl+C stops only the instance you started.

Automated tests use explicitly synthetic fixtures without network or research
datasets. Real-artifact relocation/parity checks are separately recorded in the
local project progress. Do not delete real weights or use private final-capture
holdouts merely to exercise a demo. [Third-party notices](DEMO_THIRD_PARTY_NOTICES.md)
record the permitted-data/weight boundary and primary-source links.

## Publishing is a separate action

No release has been published by this task. The owner may later attach the ZIP
and checksum file to a GitHub release after reviewing the inventory, notices,
intended-use terms and verification results. Do not attach raw datasets, local
sample photos, `.env`, weights, prediction files or the whole research artifact
tree. Do not describe the package as an online hosted inference service.

Checksums detect changes relative to trusted published values; they are not a
cryptographic signature proving the identity of a publisher who supplies both
the archive and its hashes. Review source/version before executing downloaded
code. Preserve historical scientific reports; packaging must not imply a new
model, corrected failure cases, universal calibration or factory certification.
