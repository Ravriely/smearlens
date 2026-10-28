# 10 · How much does the biologist need to review?

Date: 28/10/2026. Script: `scripts/review_coverage.py`.
Numbers: [`10-review-coverage.json`](10-review-coverage.json).

## Question

SmearLens does not replace the biologist, it sorts the work. If the biologist
checks only the cells the classifier is least sure about, how many errors are
left in the cells nobody looks at? This is the impact number: time saved at
a given quality.

## Setup

- Classifier: the stain-augmented Glimmer probe ([08](08-stain-augmented-probe.md)).
- Data: the full PBC test split, 2,560 cells.
- Calibration: one temperature fitted on the val split (minimum negative
  log-likelihood). Confidence = highest class probability.
- Review curve: sort the test cells by confidence, the biologist reviews the
  lowest X % (and corrects them), we measure the error rate in the rest and
  the share of all errors that the review caught.

## Results

The probe is already well calibrated: expected calibration error 0.72 %
before and 0.59 % after temperature scaling (T = 0.87). Mean confidence
97.9 % for 97.6 % accuracy.

| Cells reviewed by the biologist | Errors left in the rest | Errors caught by the review |
|---|---|---|
| 0 % | 2.38 % | 0 % |
| 1 % | 1.93 % | 20 % |
| 2 % | 1.71 % | 30 % |
| 5 % | 0.86 % | 66 % |
| **10 %** | **0.30 %** | **89 %** |
| 15 % | 0.18 % | 93 % |
| 20 % | 0.10 % | 97 % |
| 30 % | 0.06 % | 98 % |

## What it tells us

- **Reviewing 1 cell in 10 removes 9 errors in 10.** The errors left fall from
  2.4 % to 0.3 %, about 1 wrong cell in 330.
- The confidence is honest: when the probe says 98 %, it is right about 98 %
  of the time. This is what makes a review list trustworthy, and it is why
  the review agent ([07](07-review-agent.md)) can sort cells by confidence.

For the pitch: "SmearLens shows the biologist the 10 % of cells it is least
sure about. Checking those leaves 0.3 % errors in the rest."

## Notes and limits

- In-distribution: PBC crops, as in [05](05-glimmer-linear-probe.md). On
  another lab's slides the curve will be worse, and detection errors (missed
  or extra cells, [09](09-detection-probe.md)) come on top of type errors.
- The curve assumes the biologist corrects every reviewed cell.
- A full end-to-end curve needs more typed cells than the 55 of the TXL-PBC
  test images.

## Reproduce

```bash
PYTHONPATH=. uv run python scripts/review_coverage.py
```
