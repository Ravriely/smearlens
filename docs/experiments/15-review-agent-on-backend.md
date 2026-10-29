# 15 · Review agent on the demo backend, one slide of 132 white cells

Date: 29/10/2026. Code: `smearlens/agent.py` (`review_slide`), `scripts/review_slide.py`,
`scripts/pbc_slide.py`. Scores: [`15-slide-score.json`](15-slide-score.json).
Reports and full traces: [`15-assets/`](15-assets/).

## Question

The review agent of [07](07-review-agent.md) read the output of the old
offline pipeline and only saw slides of fewer than 100 white cells. Does it
still work when it reads what the demo backend (`scripts/modal_app.py`)
returns, on a slide with enough white cells for a real differential? Is the
report right, careful and useful for the biologist who validates the slide?

## Setup

- **Input.** `review_slide(reports, images)` takes one backend `report` per
  field (white cells with type, confidence and `needs_review`, red cell and
  platelet counts) and the field images, for the crops of the second
  opinions. It does not depend on Modal. The LLM client and the vision
  call can be replaced (the unit tests use fakes, no network).
- **Tools.** Same 6 tools as in 07, adapted: `get_slide_summary` now gives
  the backend review queue (`needs_review`, cutoff of
  [11](11-external-lab-munich.md)), red cells, platelets, fields without a
  white cell and a `known_limits` list; `list_cells` sorts least confident
  first and can keep only the queued cells; at most 12 second opinions per
  slide. The guardrail now checks the reasons given to `flag_cell` as well
  as the report.
- **Slide.** No TXL-PBC source has 100 white cells in the test split
  (PBC 55, BCCD 18, BCDD 14). The slide is all **PBC** fields of **test (50)
  and val (97)**: 147 fields, 156 white cell boxes. One source, so one lab
  (Hospital Clinic of Barcelona), one stain and one camera, but not one
  patient: PBC fields are single-cell images sampled per class (5 classes,
  no ig and no erythroblast in TXL-PBC). The differential is therefore far
  from a real blood (about 20 % basophils).
- **Val is not clean.** Every val image was used to choose a setting:
  the detection probe threshold ([09](09-detection-probe.md),
  [13](13-sam3-prompts-bcdd-detection.md)), the temperature of the type
  probe ([10](10-review-coverage.md), PBC val cells) and the size filter of
  [14](14-red-cell-platelet-size-filter.md). No weight was trained on val.
  Scores are given for test and val apart.
- **Truth.** TXL-PBC boxes for counts; the PBC class of the original file
  name for the central white cell of each field.
- **Run.** One `modal run scripts/modal_app.py` with the 147 images (one
  L40S container), then the agent on Baseten (`muse-spark-open-1026`).

## Backend on the slide

| | test (50 fields) | val (97 fields) | all (147 fields) |
|---|---|---|---|
| White cells in truth / found | 55 / 47 | 101 / 85 | 156 / 132 |
| Found cells not on a truth box | 0 | 0 | 0 |
| Labelled cells missed (field with no white cell) | 3 | 12 | 15 |
| Right type (found labelled cells) | 45 / 47 (95.7 %) | 82 / 85 (96.5 %) | 127 / 132 (96.2 %) |
| Wrong type sent to review | 2 / 2 | 2 / 3 | 4 / 5 |
| Cells marked for review | 7 | 8 | 15 (11.4 %) |
| Count error per field, red cells / platelets | 4.42 / 0.16 | 4.87 / 0.21 | 4.71 / 0.19 |

Differential, found cells (132) vs truth (147 labelled cells):

| | neutrophil | lymphocyte | monocyte | eosinophil | basophil | ig |
|---|---|---|---|---|---|---|
| SmearLens | 19.7 | 18.2 | 25.0 | 15.2 | 20.5 | 1.5 |
| Truth | 19.7 | 18.4 | 24.5 | 17.7 | 19.7 | 0.0 |

The largest gap is 2.5 points (eosinophils), mostly from the 15 missed
cells (6 eosinophils). The two "ig" cells are errors (a monocyte and a
neutrophil). Red cells: 1,341 counted for 1,866 in truth (−28 %, the PBC
undercount of 14). Platelets: 71 for 91.

Cost: 147 images in 2 min 56 s of wall time, cold start included (model
load 33 s, about 1.2 s per field), about 0.10 $ of L40S.

## Agent runs

| Run | Prompt | Tool calls | Second opinions | Flags | Wrong cells among flags | Time | Agent tokens (in + out) | Vision tokens |
|---|---|---|---|---|---|---|---|---|
| 1 | 07 adapted | 14 | 6 | 4 | 2 / 5 | 64 s | 12,915 + 2,809 | 10,077 |
| 2 | revised | 25 | 12 | 9 | 4 / 5 | 122 s | 16,644 + 6,177 | 21,949 |
| 3 | revised, same as 2 | 22 | 12 | 6 | 3 / 5 | 130 s | 16,087 + 6,464 | 25,654 |

No report was refused by the guardrail and no flag reason was refused.

**Run 1 was not good enough.** Every number was right, but the report
listed only the 4 flagged cells and never said that 11 more cells were in
the review queue; it looked at only 6 of the 15 queued cells; it gave high
priority to two cells only because the second opinion disagreed, and
wrote "verify exclusion from count" as if the second opinion were right
(both cells were correct monocytes). It said nothing about the very
unusual pattern (6 of 7 types out of range).

**Revised prompt and tools** (2 changes): the summary carries
`known_limits` (the second opinion model is 65 % right on PBC cells, the
classifier 97.6 %; red cells are undercounted; ranges are indicative), and
the prompt asks for second opinions on every rare type then the other
queued cells least confident first, says a disagreement is a reason to
look, never a correction, fixes four report parts (adequacy, differential,
review list with the queue size, limits) and asks to say when most types
are out of range that slide quality and field selection should be checked.

Excerpt of run 2 ([full report](15-assets/report-pbc-147.md)):

> 132 white cells across 147 fields (15 fields without white cell) — enough
> for differential (≥100). […] Overall pattern is unusual: 6/7 types
> outside indicative ranges — check slide quality and field selection
> before using differential. […] 15 cells marked for review (3 with
> confidence <0.70). Flagged for opening first (9): High priority: IG 84
> (0.77) and 42 (0.90) — rare type + second-opinion disagreement […]
> Counts unchanged; disagreements are for attention only.

### Reading the reports

- **Right.** All numbers in runs 2 and 3 match the tool results (counts,
  percentages, ranges, the 3 cells below 0.70). One slip in run 3: it
  lists erythroblasts 0.0 % [0-0] under the heading "types outside
  ranges", although 0 is inside the range.
- **Careful.** No disease name and no advice. The only reading of the
  pattern is "unusual, check slide quality and field selection", which is
  the right call here: the slide is a class-balanced set, not a patient.
- **Useful.** Both ig cells, the only rare type, are flagged high in every
  run, and both are errors of the classifier. Runs 2 and 3 give the queue
  size and the cells to open first, and the limits in one line.
- **The second opinion is a weak signal.** In run 2, it agreed with the
  classifier on 4 cells, all right, and disagreed on 8, of which 4 were
  wrong classifier calls. It is right on 6 of the 12 cells. It finds
  errors by disagreeing, not by giving the right type: on cell 84 (a
  monocyte the classifier calls ig) it says erythroblast.
- **Runs vary.** The vision model changed its answer on 3 of the 12 cells
  between runs 2 and 3 (one unparsed), so cell 6 (a monocyte called
  lymphocyte) was flagged in run 2 and not in run 3. The flags are a
  ranking help on top of the backend queue, not a replacement for it.

## What it tells us

- The agent now runs on what the demo returns, on a slide with enough
  white cells, in about 2 minutes and for about 0.15 $ of Baseten tokens.
- On this slide the backend alone is strong (96.2 % right type, 4 of 5
  errors in a queue of 11 % of cells). The agent adds a readable report,
  the queue size, an order (rare types and disagreements first) and the
  limits, without touching the counts.
- The first version of the prompt hid most of the queue. A report for a
  biologist must always say how many cells are left to check.

## Notes and limits

- **Not a patient slide.** Single-cell PBC fields sampled per class; no ig
  and no erythroblast in truth. The agent's behaviour on a real
  differential (mostly neutrophils and lymphocytes, a few rare cells) is not
  measured.
- **Val was used to choose settings** (see Setup). On test alone the
  backend scores the same (95.7 % vs 96.5 %), with only 47 cells.
- **Three agent runs only**, all on the same slide.
- **Token cost** uses the public Baseten price of Muse Spark Open
  (1.25 $ per million input tokens, 4.25 $ per million output tokens). The
  vision calls report only total tokens; they are counted at the output
  price, an upper bound. Runs 1 to 3: about 0.07, 0.14 and 0.16 $.
- Fields without a white cell are all detection misses here (each PBC field
  has one labelled white cell). On a real slide such fields are normal; the
  report does not tell the two apart.
- The cell crops are not stored in the review JSON: the UI needs the field
  images (field name and box are in each flag and second opinion).

## Reproduce

```bash
uv run python scripts/pbc_slide.py fields --out runs/slide-pbc/truth.json
uv run modal run scripts/modal_app.py $(uv run python scripts/pbc_slide.py args runs/slide-pbc/truth.json) \
    --out runs/slide-pbc/reports.json
uv run python scripts/pbc_slide.py score runs/slide-pbc/truth.json runs/slide-pbc/reports.json
export BASETEN_API_KEY=...
uv run python scripts/review_slide.py --run runs/slide-pbc/reports.json --out results/review/slide-pbc-147
```
