# M16 — Portable local demo release receipt

Prepared **2026-10-25**. Status: local release candidate verified; **not uploaded,
published as a GitHub release, or deployed as a hosted service**.

## Release identity

| Item | Verified value |
|---|---|
| Archive filename | `industrial-anomaly-demo-m16.zip` |
| Archive bytes | 7,099,862 (about 7.1 MB) |
| Manifest-listed files | 92, plus the manifest itself |
| Archive SHA-256 | `5bea726b2f92ca91b6df7cf2a6c774c6bad5bd6deacaf52efd36d97c26ccd0ac` |
| Manifest SHA-256 | `ca5f4627bace8a88ae84e4a3937c1f3eda26f9950b1ee226e764baebac0b0a70` |
| Runtime | Original CPU M10 inspector, schema-1 M8 gate, 12 VisA references |
| Source M10 metadata SHA-256 | `43fa655f235282152cfa29dae68acb30c608d5106858012b5e9ea8dc1a2c158b` |

Local outputs are under ignored `artifacts/releases/`. The ZIP, extracted
candidate and `industrial-anomaly-demo-m16.SHA256SUMS` are produced by:

```powershell
.venv\Scripts\python.exe scripts/build_demo_release.py
```

Existing output paths are never overwritten. A second build with a distinct
output path produced identical archive and manifest hashes. The checksums above
pin this candidate; source/documentation changes require a new candidate and
receipt, not a claim that an altered archive is the same release.

## Included/excluded boundary

Includes project source/interface, locked CPU setup specification, numeric
normal memories/calibration/quality/localization references, source receipts,
aggregate M10 evidence, cards and attribution. Numeric references are repacked
without sample IDs, original index metadata or machine-specific data paths.

Excludes source images/masks, private captures, test predictions, credentials,
`.env`, historical experiment trees and pretrained checkpoints. No KSDD2 runtime
references ship. VisA attribution and the separate pretrained-model terms are
documented in [third-party notices](../docs/DEMO_THIRD_PARTY_NOTICES.md).
Absence of raw images does not make arbitrary feature memories anonymous or
automatically redistributable; the exporter is restricted to the pinned public
VisA-derived references. No unrestricted commercial rights clearance is asserted.

## Verification performed

- **156 tests pass**, including dependency-free verification with Python site
  packages disabled, safe paths/inventory/hash enforcement, malformed policies/
  arrays, explicit checkpoint preparation and inherited prediction parity.
- Formatting/lint, strict `mypy src scripts` (75 files), and whitespace checks pass.
- All **12 categories** retain exactly equal banks, projections, normal score
  references, thresholds, quality parameters and localization values.
- Five fixed existing local candle fixtures give identical decisions, scores,
  thresholds, previews and heatmaps, excluding stage timing. One reaches the
  anomaly model; four recapture before inference. These cases include known
  failures and do not establish new accuracy or genuine fresh-capture evidence.
- Extracted the actual ZIP into a separate directory containing spaces. A fresh
  interpreter verified all files without ML dependencies and loaded the model
  with research data/artifact/cache paths deliberately pointing to missing
  folders. Only the explicitly supplied existing backbone file was required.
- Live relocated page/health returned 200/ready with portable manifest identity,
  schema-1 quality, cleaned localization, no-store/CSP and persistence false.
  Five live inspections retained pass/recapture semantics; JSON export omitted
  image data. Occupied-port startup exited 2 without stopping the first server.
- Stopped only the owned temporary test instance. No new account, package,
  weight download, dataset upload, commit or unrelated-server shutdown occurred.

The existing M10/M8 model and all historical scientific results remain unchanged.
M14 still awaits genuine photographs. This milestone improves distribution and
setup, not anomaly accuracy, capture generalization or certification.

See [maintainer/manual checks](../docs/PUBLIC_DEMO_RELEASE.md) and the
[reviewer README](../docs/DEMO_RELEASE_README.md). External publication remains
a separate owner decision after reviewing the inventory, rights and receipt.
Checksums are integrity receipts relative to trusted published values, not
cryptographic publisher signatures.
