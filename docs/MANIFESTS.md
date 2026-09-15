# Manifest and leakage-audit specification

## Publication contract

Every dataset builder emits newline-delimited JSON only after its audit passes.
Paths are dataset-root-relative and use forward slashes. Records never contain
private absolute paths, download URLs, names, or appearance-based identity
guesses. A rejected build writes diagnostic audit JSON but no canonical
manifest. A pre-existing canonical manifest is quarantined under a content-hash
suffix so stale data cannot be consumed accidentally.

Downstream code must use the dataset's verified loader. It checks `passed:
true`, manifest SHA-256, schema and dataset names, safe paths, record count, and
field structure before returning records.

## DFDC Preview contract

Core fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | DFDC manifest contract version, currently `1.0` |
| `video_id` | Stable opaque ID derived from release and relative path |
| `relative_path` | Portable path below the configured DFDC root |
| `split` | Derived `train`, `val`, or provider-preserved `test` |
| `provider_split` | Immutable `train` or `test` value from `dataset.json` |
| `label` / `class_name` | `0`/`real` or `1`/`manipulated` |
| `manipulation_method` | `original`, `method_A`, or `method_B` |
| `target_identity_id` | Provider identity in the base video |
| `swapped_identity_id` | Provider identity inserted into a fake, otherwise null |
| `lineage_identity_ids` | All identities that constrain the derived split |
| `augmentations` | Provider-declared test degradation metadata |
| `sha256` | Full-file digest for provenance and duplicate detection |
| `probe` | Codec, dimensions, frame rate/count, and duration from ffprobe |

The audit fingerprints the original `dataset.json`, records the split seed and
policy, reconciles metadata with every local MP4, and reports exact observed
counts. Meta's test identities remain test-only. Target and swapped identities
form connected components; entire provider-training components are assigned to
validation or training deterministically.

```text
provider train identities ---- connected by every face swap
          |                                |
          +---------- component -----------+
                          |
                    seeded assignment
                     /             \
                  train             val

provider test identities ----------------> test (immutable)
```

Run a real build only after replacing the pending access date:

```powershell
uv run --no-sync python scripts/build_dfdc_manifest.py `
  --config configs/data/dfdc-preview.yaml
```

Checksums and probes may be skipped only in synthetic developer tests. Research
manifests must retain `checksum: sha256` and `ffprobe: required`.

## FaceForensics++ retained contract

The FaceForensics++ builder emits newline-delimited JSON only after its audit
passes. Paths are dataset-root-relative and use forward slashes. A record never
contains a private absolute path, download URL, or face identity guess.

Core fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Manifest contract version, currently `1.0` |
| `video_id` | Stable opaque ID derived from dataset release and relative path |
| `relative_path` | Portable path below the configured dataset root |
| `split` | `train`, `val`, or `test`, assigned from complete source lineage |
| `label` / `class_name` | `0`/`real` or `1`/`manipulated` |
| `manipulation_method` | `original` or the official manipulation directory |
| `target_sequence_id` | Original sequence appearing in the target role |
| `donor_sequence_id` | Donor/source sequence for manipulated media, otherwise null |
| `lineage_sequence_ids` | All source sequences that constrain the split |
| `identity_ids` | Verified identity IDs only when an approved mapping is supplied |
| `sha256` | Full-file digest used for provenance and duplicate detection |
| `probe` | Codec, dimensions, frame rate/count, and duration from ffprobe |

The accompanying audit contains aggregate counts, official split-file hashes,
the exact manifest SHA-256, limitations, warnings, and errors. An invalid
candidate produces an audit but no manifest, preventing accidental downstream
use. If an older canonical manifest exists when a new audit fails, it is moved
to a content-hash-suffixed `.stale-*` filename so it remains recoverable but
cannot be mistaken for the current approved manifest.

Downstream FaceForensics++ code must use its `load_verified_manifest`, which refuses rejected
audits, tampered manifests, unsupported schema versions, absolute paths, and
record-count disagreement. Merely finding a JSONL file is never sufficient
authorization to train.

### FaceForensics++ leakage invariants

```text
official sequence pair
        |
        v
target/donor lineage ---- optional verified identities
        |                            |
        +-------------+--------------+
                      v
              exactly one split
                      |
                      v
              video -> clips -> frames
```

- All target and donor sequence IDs for a manipulated video must resolve to one
  official split.
- A source sequence can occur in only one split across all originals,
  manipulation methods, and compression levels.
- When an identity map is supplied, an identity can occur in only one split.
- Byte-identical files cannot cross splits.
- Clips and frames inherit the parent `video_id` and split. They are never
  randomly re-split.
- Missing assignments, expected directories, method counts, source coverage,
  required probes, or identity rows reject the manifest.

### FaceForensics++ configuration and execution

The tracked configuration is `configs/data/faceforensicspp.yaml`. Before a real
build, replace `dataset.access_date: PENDING` with the actual approved download
date and ensure `.env` points at the local data root.

```powershell
uv run --no-sync python scripts/build_faceforensics_manifest.py `
  --config configs/data/faceforensicspp.yaml
```

Checksums and probes may be skipped only for synthetic/developer smoke tests.
A real research manifest must retain `checksum: sha256` and `ffprobe: required`.

### FaceForensics++ identity-map extension

FaceForensics++ does not provide a project-verified identity mapping in the
selected metadata. If a mapping is later constructed under the applicable
terms, use an access-controlled CSV:

```csv
sequence_id,identity_id
000,opaque-person-001
001,opaque-person-001
```

Every observed sequence must be mapped. Identity IDs must be opaque and must
not encode names or demographic inferences. Set `split.identity_map` to the
local CSV path and rerun the complete audit.
