# Data setup

Do not commit datasets or derived face/video files to this repository.

For every dataset used, record its official source, version, access date, license or terms, expected checksums, and preprocessing configuration. Each collaborator must obtain restricted datasets independently from their official distributor.

Planned sources:

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Deepfake Detection Challenge](https://ai.meta.com/datasets/dfdc/)

Generated manifests should use opaque identifiers and relative paths. Before training, verify that identities, source videos, and derived frames do not leak across splits.

Project governance and schema references:

- [FaceForensics++ dataset card](../docs/datasets/FACEFORENSICS_PLUS_PLUS.md)
- [DFDC dataset card](../docs/datasets/DFDC.md)
- [Dataset governance gate](../docs/datasets/GOVERNANCE.md)
- [Manifest and leakage-audit specification](../docs/MANIFESTS.md)

The tracked FaceForensics++ configuration deliberately contains
`access_date: PENDING`. Replace it only after the authorized researcher obtains
official access. Training code must load manifests through the verified
manifest loader and require a matching audit with `passed: true`.
