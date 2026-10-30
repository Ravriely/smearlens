# 16 · Rescue the best candidate of a field with no white cell

Date: 30/10/2026. Code: `smearlens/analyze.py` (`white_cells`, `RESCUE_FLOOR`),
`scripts/detection_rescue.py`. Scores: [`16-empty-field-rescue.json`](16-empty-field-rescue.json).

## Question

On the demo, some TXL-PBC fields came back with no white cell at all, while
the cell is plain to see. Three of the 50 PBC test fields do this
(`MO_70630`, `LY_74824`, `EO_24835`). Why, and can the backend keep these
cells without adding false ones?

## Setup

- **Diagnosis.** `/api/analyze` now returns `candidate_boxes`: every SAM 3
  candidate with its probability from the detection probe of
  [09](09-detection-probe.md). On the three fields, the white cell is a
  candidate with p = 0.997, 0.995 and 0.999. The threshold chosen in 09 is
  0.9995, so the probe drops it by a hair. The threshold was chosen on val
  for precision: PBC precision is 1.00.
- **Two rules**, each with a floor below the threshold:
  - `every_field`: keep every candidate with p ≥ floor (a plain lower
    threshold). The cells between the floor and 0.9995 go to review.
  - `empty_field_only`: only when no candidate passes 0.9995, keep the best
    one if p ≥ floor, marked `needs_review` and `uncertain_detection`.
- **Data.** The deployed backend (SAM 3, two prompts, detection and type
  probes) on all TXL-PBC val (154) and test (82) fields of the three
  sources. Matching: a found cell is right when its centre is in a truth
  box ([03](03-sam3-prompt-sweep.md)). Floors from 0.9 to 0.9995. The floor
  is chosen on val, test is read once.

## Results

`every_field` loses on every floor. On val, each white cell it finds comes
with 3 to 6 false ones: F1 0.795 → 0.788 at 0.999, → 0.722 at 0.99.

`empty_field_only`, F1 on val: 0.795 (no rescue), 0.802 (0.999), 0.807
(0.99), 0.811 (0.98), **0.829 (0.95)**, 0.829 (0.9). The floor is **0.95**
(the higher of the two best).

| Split / source | Fields | F1 | Recall | Precision | Count error per field | Fields with white cells, none found | Rescued: right / wrong |
|---|---|---|---|---|---|---|---|
| val, all | 154 | 0.795 → **0.829** | 0.798 → 0.873 | 0.793 → 0.789 | 0.34 → **0.24** | 22 → 6 | 12 / 4 |
| test, all | 82 | 0.798 → **0.815** | 0.793 → 0.862 | 0.802 → 0.773 | 0.40 → **0.27** | 12 → 1 | 6 / 5 |
| test, PBC | 50 | 0.922 → 0.933 | 0.855 → 0.891 | 1.000 → 0.980 | 0.16 → 0.10 | 3 → 0 | 2 / 1 |
| test, BCCD | 18 | 0.696 → 0.750 | 0.889 → 1.000 | 0.571 → 0.600 | 0.78 → 0.67 | 2 → 0 | 2 / 0 |
| test, BCDD | 14 | 0.480 → 0.516 | 0.429 → 0.571 | 0.545 → 0.471 | 0.79 → 0.36 | 7 → 1 | 2 / 4 |

On the demo, `LY_74824.jpg` (0 white cell before) now shows one
lymphocyte, typed right, in the review queue with "uncertain detection".

## What it tells us

- The probe threshold of 09 is right for a sure cell, but a hard cut throws
  away cells the probe is almost sure of. Lowering it everywhere costs more
  than it gains.
- Rescuing only on an empty field keeps the gain and the review queue
  absorbs the risk: a rescued cell never enters the count as a sure cell, it
  is marked for review with the reason. The agent is told to say that it may
  not be a white cell, and to flag it.
- The count error falls by a third (0.40 → 0.27 per test field), and only
  one test field with a white cell is still empty.

## Notes and limits

- **TXL-PBC has a white cell in almost every field**, by construction. On a
  real slide many fields have none: there, the rule can add one false cell
  per empty field whose best candidate reaches 0.95. It goes to review, but
  the biologist then sees it. To measure on real slides with empty fields.
- BCDD gains recall but loses precision (4 of 6 rescued cells are wrong):
  its pale white cells are the weak spot of the detector since
  [13](13-sam3-prompts-bcdd-detection.md).
- The rescue does not check the type: a rescued cell is typed by the same
  probe, with its own confidence. Types are only known on PBC.
- Val was already used to choose the threshold of 09 and other settings
  (see [15](15-review-agent-on-backend.md)); the floor is one more setting
  chosen on it.

## Reproduce

```bash
uv run python scripts/detection_rescue.py fields --out runs/rescue/truth.json
modal run scripts/modal_app.py $(uv run python scripts/detection_rescue.py args runs/rescue/truth.json) \
    --out runs/rescue/reports.json
uv run python scripts/detection_rescue.py score runs/rescue/truth.json runs/rescue/reports.json \
    --out docs/experiments/16-empty-field-rescue.json
```

The scoring replays both rules from `candidate_boxes`, so any backend
version that returns them gives the same result, with or without the rescue.
