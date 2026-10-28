# 09 · A Glimmer probe for "white cell or not"

Date: 28/10/2026. Script: `scripts/detection_probe.py` (runs on a CPU box,
Glimmer features on Modal). Summary: [`09-detection-probe-summary.json`](09-detection-probe-summary.json).

## Question

Detection is now the weakest step (F1 0.73 in [06](06-pipeline-with-glimmer-probe.md)),
and it costs one Muse Spark call per SAM candidate. Can a second linear probe
on the frozen Muse Glimmer features decide "white cell or not" better, with
no LLM call?

## Setup

1. **Candidates.** SAM 3 on the TXL-PBC train, val and test splits, two
   prompts: "cell with a large purple nucleus" (score ≥ 0.3) and "red blood
   cell" (score ≥ 0.5), images under 512 px upscaled first. Nested boxes are
   merged. Adding the red cell prompt gives candidates on BCDD, where the
   white-cell prompt alone finds nothing.
2. **Labels.** A candidate is positive when its centre falls in a true white
   cell box. Train: 9,188 candidates (669 positive); val: 2,686 (191);
   test: 1,509 (121).
3. **Features.** Crop at PBC scale (red cells as the ruler, as in 06),
   Glimmer perception encoder, mean of the last hidden state.
4. **Probe.** Class-balanced logistic regression; weight decay and decision
   threshold picked on val (best val F1).
5. **Merge.** Kept candidates closer than one red cell diameter, or nested,
   count as one white cell (highest probability wins).

## Results

White cell detection on the 82 test images (87 true white cells, centre-in-box match):

| Filter | Precision | Recall | F1 | Count error / image | Muse Spark calls |
|---|---|---|---|---|---|
| SAM 3 alone | 0.15 | 0.77 | 0.25 | 4.98 | 0 |
| SAM 3 + Muse Spark (04/06) | 0.78 | 0.68 | 0.72 | 0.55 | 455 |
| **SAM 3 + Glimmer detection probe** | **0.95** | **0.68** | **0.79** | **0.38** | **0** |

By source (probe): PBC F1 0.90 (precision 1.00, count error 0.20), BCCD F1
0.77 (count error 0.44), BCDD F1 0.13 (1 of 14 white cells found).

Before the merge step, many white cells produced 2-3 positive candidates
(nucleus lobes, red cell box around a white cell), and BCCD precision was
only 0.26. The merge fixes that.

## What it tells us

- **The same frozen Meta encoder also does detection.** A logistic layer
  beats the Muse Spark filter on precision (0.95 vs 0.78) and count error
  (0.38 vs 0.55), and removes all per-cell LLM calls: the pipeline becomes
  SAM 3 → Glimmer (one pass) → two linear probes. Muse Spark is kept for the
  review agent, where it reasons over the whole slide.
- **BCDD is still out of reach.** Its white cells are pale on a greenish
  background; the probe, trained with only 70 BCDD images among 524, rejects
  them. A stain-augmented version of this probe (as in [08](08-stain-augmented-probe.md))
  is the next step.
- The threshold chosen on val is 0.9995: the probe is very confident and not
  calibrated. Fine for a yes/no decision, but the review list needs
  calibrated probabilities (next experiment).

## Notes and limits

- Recall (0.68) is capped by SAM: about 23 % of the white cells get no
  candidate whose centre falls inside them.
- The first runs of this experiment crunched crops and the logistic
  regression on the Raspberry Pi that hosts the assistant; it is now run on a
  separate CPU container. Numbers are unchanged.

## Reproduce

```bash
for s in train val test; do
  modal run scripts/modal_sam3.py --limit 1000 --split $s --threshold 0.3 --tag detect-$s-up512 \
    --upscale-below 512 --wbc-prompt "cell with a large purple nucleus" --platelet-prompt "small purple dot"
done
PYTHONPATH=.:scripts uv run python scripts/detection_probe.py
```
