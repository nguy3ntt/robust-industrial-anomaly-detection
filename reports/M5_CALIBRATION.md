# M5 calibration and selective-review report

## Scope and controls

M5 calibrates the three frozen M4 models using only clean normal samples from
the deterministic validation partitions derived from official training data.
The official VisA test partition is unchanged and is used only after the policy
is frozen. The augmented 512-entry multi-scale coreset is primary; the plain
coreset and 2,048-entry mid-scale reference remain comparators.

The full run uses 1,301 validation-normal images, producing 3,903 stored
model-score pairs. Category calibration sets contain 68 to 136 samples, so the
smallest attainable conformal p-value ranges from 0.0073 to 0.0145. All source
manifests, M3/M4 indexes, predictions, configurations, and model weights are
verified against their recorded hashes before evaluation.

## Frozen calibration policy

Two normal-only methods are evaluated at both category and pooled scope:

- a conservative empirical 95th-percentile score threshold;
- upper-tail split-conformal p-values with the plus-one correction and
  conservative tie handling.

The primary category-conditional conformal policy is:

| Region | Rule | Meaning |
|---|---|---|
| Anomalous | p ≤ 0.05 | automatic anomaly alert |
| Review | 0.05 < p ≤ 0.20 | defer to a human |
| Normal | p > 0.20 | automatic normal decision |

Exact Clopper-Pearson intervals accompany binomial operating metrics. The
coverage-risk curve freezes review boundaries at 0.05, 0.10, 0.20, 0.30, 0.40,
and 0.50; 0.05 is the no-review conformal baseline.

## Clean operating results

Macro means across 12 categories for the primary augmented coreset are:

| Scope and method | Normal FPR | Review rate | Auto coverage | Selective risk | Confident anomaly recall | Anomaly miss rate |
|---|---:|---:|---:|---:|---:|---:|
| Category empirical quantile | 0.0496 | 0.0000 | 1.0000 | 0.2567 | 0.5683 | 0.4317 |
| Category split conformal | **0.0430** | 0.1958 | 0.8042 | **0.1940** | 0.5158 | **0.2417** |
| Pooled empirical quantile | 0.0650 | 0.0000 | 1.0000 | 0.4445 | 0.2433 | 0.7567 |
| Pooled split conformal | 0.0650 | 0.2589 | 0.7411 | 0.3073 | 0.2433 | 0.4983 |

Category conformal clean normal FPR is 0.0430 with a category-bootstrap 95%
interval of [0.0239, 0.0633]. Individual category estimates and exact binomial
intervals are retained in `category_conditional.csv`; point estimates range
from 0.0000 to 0.1200, illustrating why pooled summaries cannot replace
category reporting.

Selective risk is the error rate among automatic decisions only. A reviewed
sample is neither counted as correct nor incorrect. Risk is explicitly
undefined when a policy makes zero automatic decisions and such units are
excluded, not treated as zero risk.

## Shifted false-positive control

The conformal validity claim applies only when deployment normals are
exchangeable with calibration normals. The controlled M4 shifts deliberately
stress that assumption. Primary category-conformal macro results are:

| Family | Severity | Normal FPR | Review rate | Auto coverage | Selective risk |
|---|---:|---:|---:|---:|---:|
| Clean | 0 | 0.0430 | 0.1958 | 0.8042 | 0.1940 |
| JPEG | 1 | 0.0632 | 0.2576 | 0.7424 | 0.1801 |
| JPEG | 2 | 0.4696 | 0.2579 | 0.7421 | 0.2707 |
| Resize | 1 | 0.0389 | 0.1987 | 0.8013 | 0.1943 |
| Resize | 2 | 0.1226 | 0.3062 | 0.6938 | 0.1911 |
| Blur | 1 | 0.1127 | 0.2896 | 0.7104 | 0.1982 |
| Blur | 2 | **0.8686** | 0.1122 | 0.8878 | **0.3901** |
| Noise | 1 | 0.0663 | 0.2516 | 0.7484 | 0.1912 |
| Noise | 2 | 0.6820 | 0.1832 | 0.8168 | 0.3317 |
| Photometric | 1 | 0.0272 | 0.1845 | 0.8155 | 0.2035 |
| Photometric | 2 | 0.0281 | 0.2038 | 0.7962 | 0.2026 |
| Geometric | 1 | 0.0461 | 0.2007 | 0.7993 | 0.1854 |
| Geometric | 2 | 0.2179 | 0.2318 | 0.7682 | 0.2224 |

Moderate JPEG, resize, noise, photometric, and geometric changes remain near
the clean operating region, while blur already exceeds the nominal target.
Severe blur, noise, JPEG, and geometry invalidate false-positive control. The
review band exposes ambiguity but cannot repair an exchangeability failure when
shifted normals receive confidently extreme scores.

## Confirmatory tests

| Test | Estimate | 95% interval | Coverage gate | Conclusion |
|---|---:|---:|---:|---|
| H3: category calibration improves clean 5% false-alarm alignment over pooled calibration | +0.0681 absolute-error improvement | [+0.00495, +0.18088] | n/a | Supported |
| H4: review lowers primary-model automatic-decision risk across clean and shifted conditions | +0.07125 risk reduction | [+0.06063, +0.08122] | 0.7790 ≥ 0.70 | Supported |

H3 uses the category as the resampling unit. H4 uses all 156 paired
category-condition units and compares the p ≤ 0.05 no-review policy with the
frozen 0.05–0.20 review band; all units have defined automatic risk in the full
run.

## Artifacts and restart behavior

The authoritative full run is `m5-calibration-full-20260920T110316Z`. It
contains:

- 3,903 validation calibration scores and 39 fitted calibration records;
- 337,272 per-sample method/scope decisions;
- 1,872 operating rows and 5,616 coverage-risk rows;
- 156 condition summaries, 156 family rows, and 144 clean category rows;
- two confirmatory-test records plus resolved configuration, environment,
  progress, source provenance, and hashes for every output.

Calibration scores are written atomically after every category. An interrupted
run can be resumed with `--resume <run-directory>` and will reject changed
configuration, data, source predictions, or model indexes before skipping
completed categories.

## Assumptions and limits

- Split-conformal false-alarm validity is marginal, finite-sample, and depends
  on exchangeability between calibration normals and future normal inputs.
- Category-conditional calibration assumes the category is known and stable.
- No false-alarm guarantee is claimed for shifted inputs; those results are
  empirical stress tests and show clear failures.
- Calibration uses normal data only. Anomaly recall and selective risk are
  reported after freezing the policy, not optimized using test defects.
- Human review defers uncertain cases; it is not a safety certification and
  does not identify the physical cause of a defect.
- Operators must monitor input quality and score drift. Severe blur, noise,
  compression, or geometry changes require capture correction or recalibration,
  not blind reliance on the clean policy.
