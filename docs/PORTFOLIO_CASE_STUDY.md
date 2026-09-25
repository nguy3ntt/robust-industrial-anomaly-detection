# Robust Industrial Anomaly Detection

## Designing for the failures hidden by a benchmark score

### The brief

Manufacturing-defect data is sparse and uneven: normal examples are plentiful,
but labelled defects rarely cover the combinations of product, camera, lighting,
and process change encountered later. I built a normal-only inspection research
system that detects unusual images, localizes suspicious regions, and exposes a
human-review policy. The core challenge was not maximizing one leaderboard
number. It was finding where the system stops being trustworthy.

### What I built

The project is an end-to-end, configuration-driven research system:

- secure acquisition and a leakage-aware manifest for 10,821 VisA images;
- interpretable, global-feature, and spatial patch-memory baselines;
- multi-scale frozen ResNet-18 features with bounded k-center memory;
- six controlled capture/post-processing shift families at two severities;
- category-conditional conformal calibration and selective review;
- a local privacy-preserving inspection workstation; and
- a hash-gated academic and portfolio publishing pipeline.

Every reportable run records its resolved configuration, seed, environment,
dataset manifest, model identity, output hashes, predictions, and metrics. Raw
data, private images, weights, and feature banks remain outside the repository.

### Why the design stayed simple

Nearest-neighbour patch memory was chosen deliberately. Its training data is
normal-only; memory capacity is explicit; spatial scores can be projected back
to the input; and model additions can be measured without hiding them inside a
large fine-tuned network. The strongest design retains 512 k-center prototypes
per category from frozen multi-scale features. That is about 60% less index
storage than the 2,048-entry reference while improving image AUROC from 0.8283
to 0.8679 and AUPRO@30 from 0.7156 to 0.7692.

### The result—and the boundary

The final augmented system reaches macro image AUROC 0.8674, pixel AUROC 0.9557,
and AUPRO@30 0.7762 across 12 categories. Normal-only augmentation produces a
small but supported shifted-AUROC gain of 0.0077, 95% CI [0.0030, 0.0126].

On clean data, category-conformal calibration reaches normal FPR 0.0430. A
selective-review band automatically decides 80.4% of samples and reduces the
automatic error rate by 0.0713 relative to no review.

The most important finding is negative: clean calibration is not robust
calibration. Under severe blur, noise, and JPEG compression, normal FPR rises to
0.8686, 0.6820, and 0.4696. Leave-one-category-out AUROC is only 0.5783. A known
product category, controlled capture, drift monitoring, and human review are
therefore requirements—not deployment polish.

### Decisions that improved rigor

1. **Provider test preservation.** Validation comes only from grouped
   training-normal samples; the official test partition is untouched.
2. **Duplicate gates.** Exact and confirmed perceptual groups cannot cross
   splits.
3. **Matched geometry.** An audit exposed an image/mask transform mismatch. I
   corrected the evaluator and regenerated localization results rather than
   retaining more flattering numbers.
4. **Equal-budget intervention.** Robustness augmentation uses the same patch
   and prototype budgets as the comparator.
5. **Negative results stay visible.** The inconclusive multi-scale robustness
   hypothesis and weak category transfer are reported with the headline result.
6. **No threshold sandbox.** The workstation displays frozen boundaries but
   does not let a user tune them against an inspected sample.

### Practical system

The local workstation runs on CPU at 77.3 ms median and 108.8 ms p95 per image,
or 12.55 sequential images/s. It loads 49.97 MB of model/calibration artifacts,
uses 548.7 MB peak process working set in the benchmark, binds only to loopback,
and exports metadata-only reports by default. The interface shows the evaluated
model crop, continuous heatmap, fixed support contour, score, conformal p-value,
decision, thresholds, stage timing, and limitations in one technical view.

### Skills demonstrated

- Experimental design and hypothesis discipline
- Computer vision feature extraction and spatial evaluation
- Calibration, selective prediction, and uncertainty communication
- Robustness testing and error taxonomy design
- Data integrity, provenance, and reproducible pipelines
- Typed Python, automated tests, local web systems, and visual communication
- Responsible-use framing for high-consequence decision support

### What I would do next

A production-oriented continuation would collect approved normal images from the
actual camera path, define measurable capture-quality gates, validate drift
detection prospectively, and run a shadow-mode study with domain experts. Model
complexity should increase only after those data and operating controls exist.

