# 14 · A size filter on red cell and platelet counts

Date: 29/10/2026. Script: `scripts/count_filter_eval.py` (CPU, no new SAM 3
or Glimmer run). Data: [`14-count-filter.json`](14-count-filter.json).

## Question

The demo backend (`scripts/modal_app.py`) counts too many platelets and,
on BCCD and BCDD fields, several red cells too many per image. SAM 3 boxes
for "small purple dot" and "red blood cell" come in all sizes. Can a size
rule, scaled on the red cells of the same image, fix the counts with no
model and no LLM call?

## Setup

- **Scale.** Each image already has `rbc_px`, the median longest side of
  its red cell boxes with score ≥ 0.5 (as in [06](06-pipeline-with-glimmer-probe.md)).
  A box's relative size is its longest side divided by `rbc_px`.
- **Rule** (`count_cells` in `smearlens/analyze.py`):
  1. keep a box only if its relative size is in a range (one range for red
     cells, one for platelets);
  2. remove nested boxes (the existing `dedupe`);
  3. drop a box whose centre is inside a kept white cell, or that contains
     the centre of a kept white cell (new: a red cell box drawn over a
     white cell).
- **Red cell score floor.** The count used score ≥ 0.5. With the size rule
  in place, 0.3 (the SAM 3 cutoff) is also tried. The white cell candidates
  and `rbc_px` still use 0.5.
- **Data.** The cached SAM 3 detections of [13](13-sam3-prompts-bcdd-detection.md)
  (`runs/detect-{val,test}-up512` + `neutrophil`), the cached Glimmer
  features and the shipped detection probe, so the kept white cells are
  exactly the ones the backend keeps. Ground truth: TXL-PBC boxes for red
  cells and platelets.
- **Protocol.** Grid search on the 154 val images (red cells: score floor
  0.3 / 0.4 / 0.5, lower bound 0 to 0.9, upper bound 1.4 to none;
  platelets: lower bound 0 to 0.28, upper bound 0.4 to none). The best val
  setting is then run once on the 82 test images.

Check: on one PBC and one BCCD test image, `modal run` of the backend with
the new code gives the same counts as this offline replay.

## Results

Settings kept (best val error): red cells score ≥ 0.3 and size 0.85 to
1.6 × `rbc_px`; platelets size 0.22 to 0.6 × `rbc_px`.

Mean absolute count error per image, 82 test images:

| Count | Truth / image | Before | After | Bias before → after |
|---|---|---|---|---|
| Platelets | 0.6 | 1.33 | **0.21** | +1.16 → −0.04 |
| Red cells | 15.1 | 6.51 | **3.41** | +0.66 → −1.49 |
| White cells | 1.1 | 0.34 | 0.34 | −0.17 → −0.17 |

By source (test):

| Source | Red cells before → after | Red cell bias after | Platelets before → after |
|---|---|---|---|
| PBC (50 images) | 6.94 → 4.42 | −3.06 | 1.58 → 0.16 |
| BCCD (18) | 5.11 → 2.00 | +0.89 | 1.67 → 0.50 |
| BCDD (14) | 6.79 → 1.64 | +1.07 | 0 → 0 |

Val gives the same picture: platelets 1.32 → 0.25, red cells 7.31 → 3.75,
white cells 0.32 unchanged.

## What it tells us

- **Most platelet false positives are tiny.** On val, the "small purple
  dot" boxes that do not fall on a true platelet have a median size of
  0.11 × `rbc_px`; the ones on a true platelet, 0.29. A lower bound of
  0.22 removes almost all of them. The upper bound does not change the val
  error (0.6 and none are equal); it stays as a guard against a whole cell
  counted as a dot.
- **The red cell overcount on BCCD and BCDD came from boxes of the wrong
  size.** On val, a quarter of the red cell boxes that do not fall on a
  true red cell are under 0.55 × `rbc_px` and a tenth are over 1.4 (for
  boxes on a true red cell: 0.93 and 1.24 at the same percentiles). What
  they show (fragments, clusters, pallor) was not checked by eye. With the
  size range, BCDD goes from +6.5 to +1.1 red cells per image.
- **On PBC the problem is the other way.** PBC red cells were already
  undercounted (bias −2.5) and the filter makes it worse (−3.1), even if the
  absolute error goes down. On val, SAM 3 misses 356 of 1,231 PBC red cells
  completely (106 of them touch the image border), so no filter on
  existing boxes can fix that.
- **White cells do not move.** The filter only acts on the red cell and
  platelet counts, after the white cells are chosen.

## Notes and limits

- **The red cell lower bound is sensitive.** Val error is 4.01 at 0.80,
  3.75 at 0.85 and 4.03 at 0.90. 0.85 is the val optimum, not a round
  rule; a lab with more variable red cell sizes (anisocytosis) would lose
  real cells with it. Platelets are less sensitive (0.30 at 0.20, 0.25 at
  0.22, 0.33 at 0.25).
- **The range assumes normal cells.** Giant platelets (> 0.6 × a red cell)
  and microcytes (< 0.85) are not counted. For a review tool this is a
  limit to state, not a hidden error: the counts stay a draft.
- If an image has no red cell box with score ≥ 0.5, `rbc_px` falls back to
  the PBC value (74.6 px) and the size ranges use that guess.
- Platelets are rare in this data (0.6 per test image, none annotated on
  BCDD), so the platelet error is computed on few objects.
- Only TXL-PBC sources (PBC, BCCD, BCDD). Not checked on the Munich images
  of [11](11-external-lab-munich.md), which have no red cell or platelet
  boxes.

## Reproduce

```bash
PYTHONPATH=. uv run python scripts/count_filter_eval.py
```

It needs `runs/detect-{val,test}-{up512,neutrophil}/detections.json` and
`runs/detection-probe-neutrophil/features_{val,test}.npz` from
[13](13-sam3-prompts-bcdd-detection.md).
