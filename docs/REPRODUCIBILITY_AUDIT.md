# M7 reproducibility audit

Audit date: 2026-09-23  
Host: Windows 11 AMD64  
Scope: clean source copy, fresh locked CPU environment, automated verification,
zero-output M7 reconstruction, and Python distribution build.

## Outcome

**Pass with one documented host setup caveat.** The project reproduced in an
isolated copy with a newly created virtual environment. All 54 tests, lint, and
strict typing passed. The M7 package rebuilt from an absent output directory and
produced 11 files with package SHA-256
`51f8c4151b82c9701e734ecb93cb2fbd915b8db764c7062b2abd386e0fd2b36f`.
The lock resolved without changes, and both the source distribution and wheel
built successfully.

This audit verifies the code and publishing pipeline against the existing
licensed local dataset/artifact store. It is not a claim that the VisA archive
or multi-hour M2–M6 inference can be reproduced without downloading the dataset
and pretrained weights.

## Isolation procedure

1. Copied the working source tree to
   `tmp/m7-clean-audit`, excluding `.git`, the existing `.venv`, local tools,
   caches, previous build output, and the audit directory itself.
2. Created a new CPython 3.12 environment from `uv.lock` with the CPU extra and
   development group.
3. Used a dedicated package cache on the project drive.
4. Ran the machine-readable environment check.
5. Ran the full test suite, lint, and strict type checking.
6. Deleted `reports/m7` inside the isolated copy and rebuilt it from zero using
   the pinned, hash-verified M2–M6 source runs.
7. Verified the lock and built a source distribution and wheel.

## Exact checks and results

| Gate | Result |
|---|---|
| Python | 3.12.14 |
| Platform | Windows 11 AMD64 |
| Locked environment | 66 packages resolved; 43 installed in fresh environment |
| PyTorch profile | `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu` |
| Environment checker | `RESULT: READY` |
| Tests | 54 passed |
| Ruff | all checks passed |
| Mypy | no issues in 35 source files |
| M7 zero-output rebuild | 11 outputs; stable package SHA-256 |
| Lock integrity | resolved 66 packages with no lock change |
| Distribution build | `.tar.gz` and `py3-none-any.whl` built successfully |

Representative commands:

```powershell
uv sync --project <audit-copy> --python <python-3.12> `
  --extra cpu --group dev --locked

<audit-copy>\.venv\Scripts\python.exe scripts/check_environment.py `
  --accelerator cpu
<audit-copy>\.venv\Scripts\python.exe -m pytest
<audit-copy>\.venv\Scripts\python.exe -m ruff check src scripts tests
<audit-copy>\.venv\Scripts\python.exe -m mypy src scripts
<audit-copy>\.venv\Scripts\python.exe scripts/build_m7_package.py `
  --config configs/reporting/m7.yaml
uv lock --project <audit-copy> --check
uv build --project <audit-copy> --out-dir <audit-copy>\dist
```

## Host caveat discovered

The first attempt exposed a stale user-level `uv` managed-Python minor-version
link and then exhausted the nearly full system temporary drive. This was not
hidden or counted as a pass. The successful audit explicitly selected the
verified CPython 3.12.14 interpreter and placed the audit copy and package cache
on the project drive. A new machine should either let `uv` install its managed
Python successfully or pass an explicit compatible interpreter.

This caveat affects setup convenience, not the locked dependency solution or
the project tests. The project environment checker reports the interpreter path
and version so the active runtime remains visible.

## Data and artifact prerequisites

The final-public-package reconstruction requires:

- the audited VisA manifest named in `configs/reporting/m7.yaml`;
- the five exact completed M2–M6 run directories pinned in that file;
- every required source file matching its recorded hash; and
- `ANOMALY_ARTIFACT_ROOT` pointing to the licensed local artifact store.

The builder refuses incomplete, substituted, or mutated runs. No raw image,
mask, private inspection sample, model weight, or feature bank is copied into
`reports/m7`.

## Residual limitations

- The audit used one Windows host and the CPU dependency profile.
- CUDA 13.0 remains a configured but unaudited optional profile on this host.
- Re-downloading VisA and repeating all feature extraction would test external
  availability and hardware runtime, but it is not necessary to verify the
  frozen publication aggregation because its inputs are content-addressed.
- The nearly full system drive should be remediated before unrelated software
  installation or large temporary workloads.

