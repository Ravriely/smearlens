# 03 · SAM 3 prompt and threshold sweep on the TXL-PBC test split

Date: 28/10/2026. Scripts: `scripts/sam3_validate.py` (`--wbc-prompt`,
`--platelet-prompt`, `--threshold`, `--version`), run on Modal with
`scripts/modal_sam3.py --tag <run>`. All numbers: [`03-sam3-sweep.csv`](03-sam3-sweep.csv).

## Question

[02](02-sam3-segmentation.md) showed that the text prompt "white blood cell"
detects almost every cell. Can a better prompt or threshold make SAM 3 find
white cells and platelets, and does SAM 3.1 beat SAM 3?

## Setup

| | |
|---|---|
| Data | Full TXL-PBC **test** split after the license filter: 82 images, 87 WBC, 1,241 RBC, 47 platelets |
| Hardware | Modal, 1× L40S per run, 12-19 s of compute per run |
| Match | Predicted box matched to a true box at IoU ≥ 0.5 |
| RBC prompt | "red blood cell" in every run |

## Results

White blood cells (87 true cells):

| Run | Model | Threshold | WBC prompt | Predicted | Precision | Recall | F1 | Count error / image |
|---|---|---|---|---|---|---|---|---|
| baseline | 3.1 | 0.5 | white blood cell | 1,466 | 0.02 | 0.31 | 0.04 | 17.1 |
| | 3.1 | 0.3 | white blood cell | 1,857 | 0.03 | 0.66 | 0.06 | 21.8 |
| | 3.1 | 0.5 | leukocyte | 0 | 0 | 0 | 0 | 1.1 |
| | 3.1 | 0.5 | purple nucleus | 934 | 0.07 | 0.70 | 0.12 | 10.7 |
| **best** | **3.1** | **0.5** | **cell with a large purple nucleus** | **199** | **0.23** | **0.53** | **0.32** | **2.2** |
| | 3 | 0.5 | cell with a large purple nucleus | 207 | 0.23 | 0.55 | 0.33 | 2.3 |
| | 3.1 | 0.3 | cell with a large purple nucleus | 591 | 0.11 | 0.77 | 0.20 | 6.6 |

Red blood cells (1,241 true cells): F1 **0.72** at threshold 0.5 (precision 0.66,
recall 0.81), 0.68 at 0.3 (more recall, less precision). SAM 3 and 3.1 are
within 0.01.

Platelets (47 true): **0 detections** with the prompt "platelet", at 0.5 and at
0.3. With "small purple dot" at 0.3: 153 detections, 16 matched, F1 0.16.

## What it tells us

- **Describe what the cell looks like, not what it is called.** "white blood
  cell" and "leukocyte" fail (the first finds every cell, the second finds
  nothing). "cell with a large purple nucleus" cuts the count error from 17 to
  2.2 per image and multiplies the F1 by 9.
- Same lesson for platelets: "platelet" finds nothing, "small purple dot" finds
  the first ones.
- **SAM 3.1 and SAM 3 give the same text-prompt results** on still images
  (3.1 mainly improves video tracking). But SAM 3.1 breaks box prompts, see
  below, so **SAM 3 is now the default**.
- A lower threshold buys recall at a high precision cost. For SmearLens this is
  acceptable only if the classifier removes the false positives afterwards.
- IoU ≥ 0.5 is too strict for SAM's boxes. With a **centre-in-box match**
  (the predicted box centre falls inside a true box, one match per box):
  WBC F1 0.32 → **0.37** (recall 0.61) with the best prompt, and platelets
  ("small purple dot", 0.3) F1 0.16 → **0.41, recall 0.87**. SAM finds most
  platelets; its boxes just do not overlap the ground truth enough.

For the pitch: one sentence of prompt engineering takes SAM from "useless for
white cells" to "a count within about 2 cells per image", with no training.

## Notes and limits

- **SAM 3.1 box prompts are random.** The "box mode" fill ratio of
  [02](02-sam3-segmentation.md) moved from 0.12 to 0.95 between runs, even
  for two identical runs, and even when measured before any text prompt.
  `sam3.1_multiplex.pt` loads into the image model with missing keys
  ("found missing and/or unexpected keys"): the interactive head is randomly
  initialised at each run. SAM 3 loads with no warning and gives **0.634 on
  two identical runs** (a circle inscribed in its box fills 0.785, so the mask
  covers most of the cell). Text prompts were deterministic in every run, so
  the text results above stand.
- Launching 5 Modal runs at the same time hit "App create rate limit
  exceeded". Runs are now launched one after the other.
- One prompt per run, no ensembling, no per-class threshold.

## Next

1. Done: box mode measured in a separate pass before text prompts, centre-in-box
   match added (`smearlens.metrics.match_centers`), SAM 3 as default.
2. Pipeline: SAM ("cell with a large purple nucleus", 0.3 for recall) gives the
   candidates, the classifier from [01](01-zero-shot-baseline.md) removes the
   false positives and labels the real WBCs.

## Reproduce

```bash
modal run scripts/modal_sam3.py --limit 200 --version sam3.1 --tag wbc-large-purple-nucleus \
  --wbc-prompt "cell with a large purple nucleus"
modal run scripts/modal_sam3.py --limit 200 --tag plt-small-purple-dot --threshold 0.3 \
  --wbc-prompt "cell with a large purple nucleus" --platelet-prompt "small purple dot"
```
