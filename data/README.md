# Data setup

Real data is never committed here. The VisA archive and extracted images belong
under the external path configured by `ANOMALY_DATA_ROOT`.

Primary source:

- [Visual Anomaly (VisA) official project](https://github.com/amazon-science/spot-diff)
- [Direct public archive](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)

The direct link downloads the dataset to the computer; it does not require an
AWS account or cloud environment. Do not manually rearrange the extracted
folders. The M1 preparation command verifies the archive, extracts it safely,
and builds a trusted manifest from the official `split_csv/1cls.csv` table:

```powershell
uv run --extra cu130 python scripts/prepare_visa.py --workers 8
```

On the audited workstation, canonical outputs are stored under
`D:/industrial-anomaly-data/manifests/visa-20220922/` and remain outside Git.

Before any experiment, read the [dataset card](../docs/datasets/VISA.md),
[governance gate](../docs/datasets/GOVERNANCE.md), and
[manifest contract](../docs/MANIFESTS.md).
