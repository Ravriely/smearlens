# 13 · A second SAM 3 prompt to find white cells on BCDD

Date: 29/10/2026. Scripts: `scripts/modal_sam3_prompt_bcdd.py` (prompt sweep,
Modal L40S) and `scripts/detection_probe.py --extra-tags` (CPU box). Data:
[`13-sam3-prompts-by-source.json`](13-sam3-prompts-by-source.json),
[`13-detection-neutrophil-summary.json`](13-detection-neutrophil-summary.json).

## Question

In [09](09-detection-probe.md) the detection probe found 1 of 14 white cells
on BCDD (F1 0.13). Is the probe the problem, or does SAM 3 never propose a
good box for those pale cells? And if a better prompt exists, what does it
cost on the other sources?

## Setup

1. **Prompt sweep.** Eight white-cell prompts on the train and val images of
   each source (PBC, BCCD, BCDD), images under 512 px upscaled, score cutoffs
   0.05 to 0.3. For each: candidate recall (share of true white cells with a
   box whose centre falls inside them) and boxes per image.
2. **Detection probe.** Same probe as 09, with the boxes of the best new
   prompt added to the candidate pool (score ≥ 0.3), on all three splits.
   The threshold grid now goes up to 0.99995, since the best val threshold
   sat at the top of the old grid.

## Results

Candidate recall on BCDD at score ≥ 0.3:

| Prompt | Candidate recall | Boxes / image |
|---|---|---|
| "cell with a large purple nucleus" (09) | 0.02 | 0.08 |
| "white blood cell" | 0.54 | 13.6 |
| "lymphocyte" | 0.98 | 7.0 |
| **"neutrophil"** | **0.92** | **3.1** |
| "nucleated cell" | 1.00 | 37.4 |
| "leukocyte" | 0.00 | 0.0 |

"neutrophil" gives the fewest boxes for a high recall, also on PBC (0.93,
2.2 boxes) and BCCD (0.95, 4.7 boxes). We add it to the pool. Candidates:
train 9,348 (800 positive), val 2,713 (205), test 1,509 → 1,529 (121 → 135).

White cell detection on the 82 test images:

| Candidates | Precision | Recall | F1 | Count error / image |
|---|---|---|---|---|
| 09: purple nucleus + red cell | 0.95 | 0.68 | 0.79 | 0.38 |
| **+ "neutrophil"** (threshold 0.9999) | **0.89** | **0.75** | **0.81** | **0.34** |

By source:

| Source | 09 F1 | 13 F1 | 13 precision / recall | 13 count error |
|---|---|---|---|---|
| PBC (50 images) | 0.90 | 0.91 | 1.00 / 0.84 | 0.18 |
| BCCD (18) | 0.77 | 0.72 | 0.72 / 0.72 | 0.56 |
| BCDD (14) | 0.13 | **0.52** | 0.67 / 0.43 | 0.64 |

With the old threshold grid (best 0.9995) the probe gave F1 0.81 too, with
more recall (0.81) and less precision (0.82; BCCD precision 0.59).

## What it tells us

- **On BCDD the bottleneck was SAM 3, not the probe.** The original prompt
  almost never fires on pale cells; the word "neutrophil" does. With the
  boxes there, the same frozen Glimmer probe finds 6 of 14 white cells
  instead of 1.
- **Text prompts are a cheap lever.** One more SAM 3 prompt per image, no
  training, and overall recall goes from 0.68 to 0.75.
- **There is a trade-off on BCCD.** More candidates means more false
  positives there (precision 0.81 → 0.72). Overall F1 and count error still
  improve, so this is the new default.

## Notes and limits

- BCDD has only 14 test images and 14 white cells: its numbers move a lot
  with one cell.
- The probe threshold (0.9999) is extreme: the probe is not calibrated for
  this yes/no decision.
- **Stain augmentation does not help here.** Training the detection probe
  with 2 stain-augmented copies of each train crop (as in
  [08](08-stain-augmented-probe.md), `--augment-copies 2`) gives val F1 0.77
  (vs 0.81) and test F1 0.75: BCCD precision falls to 0.42 and BCDD F1 to
  0.48 ([`13-detection-neutrophil-aug-summary.json`](13-detection-neutrophil-aug-summary.json)).
  Augmented background crops start to look like pale white cells. The
  default stays without augmentation.
- The sweep measures recall on train and val only; the test split is used
  only for the final table above.

## Reproduce

```bash
modal run scripts/modal_sam3_prompt_bcdd.py
for s in train val test; do
  modal run scripts/modal_sam3.py --limit 1000 --split $s --threshold 0.3 --tag detect-$s-neutrophil \
    --upscale-below 512 --wbc-prompt "neutrophil" --platelet-prompt "small purple dot"
done
PYTHONPATH=.:scripts uv run python scripts/detection_probe.py --extra-tags neutrophil --out runs/detection-probe-neutrophil
```
