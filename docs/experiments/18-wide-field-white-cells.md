# 18 · Wide fields: a detection probe trained on wide fields

Date: 31/10/2026. Code: `smearlens/analyze.py` (`detection_probe`, `white_cells(..., wide=)`),
`scripts/modal_app.py` (`Analyzer.analyze`), `scripts/wide_wbc_eval.py`, probe
`models/detection_probe_wide.npz`. Scores: [`18-wide-field-scores.json`](18-wide-field-scores.json),
[`18-fusion-val.json`](18-fusion-val.json), [`18-wide-probe.json`](18-wide-probe.json),
[`18-padding-single-fields.json`](18-padding-single-fields.json).

## Question

On wide fields (2 × 2 mosaics of test fields, [17](17-wide-field-tiling.md)) white cell detection
falls to F1 0.725, against 0.863 for the same fields one by one. Precision is the problem: 38 false
white cells for 66 right ones. Can it be fixed without changing single fields?

## Setup

- **Same protocol as 17.** 2 × 2 mosaics of one source, white cell boxes on a seam (± 3 px) left
  out on both sides, matching by centre. `mosaic_eval.py build` now takes `--split`: 130 train,
  38 val and 19 test mosaics. Every choice below is made on val (or train), test is read once.
- **Fusion rules** (`wide_wbc_eval.py`). One GPU pass keeps every white cell candidate of the
  whole image and of four window grids (2 × 2 and 3 × 3, margin 20 % or 50 % of a tile), with the
  logit of the detection probe of [09](09-detection-probe.md). Logits, because on mosaics many
  candidates saturate at p = 1.0. The rules are then replayed offline, 346 of them:
  - `windows`: candidates from the windows only (in the core, or all with a merge by centre);
  - `and`: a whole-image cell is kept only if a window agrees (centre in the other box, or closer
    than half a red cell);
  - `or`: whole-image cells plus window cells that no whole-image candidate covers;
  - each with the threshold searched on the windows, crops at the tile or whole-image scale;
  - `whole` with another threshold;
  - `disc`: the crop blanked outside a disc of 1 or 1.5 red cells around the candidate.
- **Wide probe.** The same logistic probe as 09 (Glimmer features of the same crops), trained on
  the whole-image candidates of the 130 train mosaics (2795 candidates, 646 on a white cell).
  Weight decay, steps and threshold chosen on the 38 val mosaics. The backend uses it only when the
  field is tiled (`tiles_per_side > 1`, rule of 17); a single field keeps the probe of 09.

## Why the probe of 09 fails on wide fields

The crop of a candidate is a square of 4.8 red cells around it. On a small field, a candidate far
from the centre gets a padded crop (filled with the background). On a wide field no crop is padded.
On the single fields of 16 (val and test), the share of non white cell candidates that the probe
keeps grows with the part of the crop that is image:

| Source | Crop < 60 % image | 60-80 % | 80-95 % | ≥ 95 % |
|---|---|---|---|---|
| BCCD | 1.5 % (907) | 1.5 % (336) | 3.8 % (132) | **14 %** (35) |
| BCDD | 0.8 % (533) | 1.6 % (126) | **32 %** (25) | 1 of 1 |
| PBC | 0 % (1463) | 0 % (341) | 0 % (3) | — |

The probe learnt in part "no padding means a centred cell". On val mosaics it keeps 38 % of the
BCDD red cells and fragments, 9 % on BCCD. Windows do not change that: the crop is the same, only
the candidates change.

## Results

Val mosaics, white cells off the seams, best of each rule (all settings searched on val):

| Rule | F1 | Precision | Recall |
|---|---|---|---|
| Whole image, probe of 09 (before) | 0.769 | 0.69 | 0.87 |
| Whole image, other threshold | 0.775 | 0.72 | 0.84 |
| `windows` (2 × 2, margin 50 %, merge by centre) | 0.797 | 0.74 | 0.86 |
| `and` whole image + windows | 0.786 | 0.74 | 0.84 |
| `or` whole image + windows | 0.781 | 0.70 | 0.88 |
| `disc` 1.5 red cells | 0.871 | 0.91 | 0.83 |
| **Wide probe** | **0.902** | **0.98** | 0.84 |

The wide probe is kept. Test, through the backend (`modal run scripts/modal_app.py`, same 95 images
as 17):

| Images | White cell F1 | Precision | Recall | Found / false / true | Red cells found / true | Platelets found / true |
|---|---|---|---|---|---|---|
| Mosaics, before (17) | 0.725 | 0.63 | 0.85 | 66 / 38 / 78 | 1013 / 1124 | 33 / 45 |
| **Mosaics, wide probe** | **0.811** | **0.86** | 0.77 | 60 / 10 / 78 | 1059 / 1124 | 36 / 45 |
| Same 76 fields one by one, before | 0.863 | 0.87 | 0.85 | 69 / 10 / 81 | 1005 / 1124 | 40 / 45 |
| Same 76 fields one by one, after | 0.863 | 0.87 | 0.85 | 69 / 10 / 81 | 1005 / 1124 | 40 / 45 |

By source (mosaics, found / false of true): PBC 45 / 2 → 46 / 1 of 51 (F1 0.918 → 0.939), BCCD
12 / 13 → 12 / 8 of 15 (0.600 → 0.686), BCDD 9 / 23 → 2 / 1 of 12 (0.409 → 0.267).

Paired bootstrap on the 19 test mosaics (5000 draws): F1 gain +0.086, 95 % interval +0.039 to
+0.122, never ≤ 0. Single fields are never tiled, so they keep the probe of 09: same cells, same
scores. Red cells counted rise a little (1013 → 1059) because fewer false white cells hide them.

## What it tells us

- The drop on wide fields was not a scale or a merge problem: it came from a shortcut of the
  detection probe, learnt from the padding of small fields. Fusion rules on windows only win 0.01
  to 0.03 of F1 on val.
- A probe trained on wide fields removes almost all the false cells (val precision 0.69 → 0.98,
  test 0.63 → 0.86) and closes most of the gap with single fields (0.811 against 0.863).
- Recall drops (0.85 → 0.77 on test), almost all on BCDD: the wide probe keeps 2 of its 12
  white cells. BCDD white cells are pale, the weak spot since [13](13-sam3-prompts-bcdd-detection.md);
  on BCDD mosaics the backend now misses cells instead of adding false ones.

## Notes and limits

- **The mosaics are the train data too.** The wide probe has seen 2 × 2 mosaics with seams, one
  stain and one camera per mosaic. A real wide field has no seams; the probe may rely on them in
  part. To check on a real wide field with boxes when one exists.
- Few mosaics: 38 val, 19 test (4 BCCD, 3 BCDD). The BCCD and BCDD numbers carry a large error.
- The 2 × 2 windows fit the four fields of a mosaic exactly, which favours the `windows` rules;
  they still lose.
- The `disc` crop (0.871 on val, no training) is a second option, not kept: the wide probe is
  better on val, and it gives the probe crops it never saw.
- The rescue of [16](16-empty-field-rescue.md) is unchanged; on a tiled field it uses the wide
  probe probability with the same floor of 0.95.
- Not tried: training one probe on single and wide fields together, which could replace both.

## Reproduce

```bash
for s in train val test; do uv run python scripts/mosaic_eval.py build --split $s --out runs/wide/$s; done
for s in train val test; do
  modal run scripts/wide_wbc_eval.py --truth runs/wide/$s/truth.json --out runs/wide/$s/features.npz --features
done
uv run python scripts/wide_wbc_eval.py train-wide runs/wide --probe models/detection_probe_wide.npz \
    --out docs/experiments/18-wide-probe.json
modal run scripts/wide_wbc_eval.py --truth runs/wide/val/truth.json --out runs/wide/val/raw.json
uv run python scripts/wide_wbc_eval.py tune runs/wide/val/truth.json runs/wide/val/raw.json --out runs/wide/val/tune.json
uv run python scripts/wide_wbc_eval.py padding runs/rescue/truth.json runs/rescue/reports.json \
    --out docs/experiments/18-padding-single-fields.json
modal run scripts/modal_app.py $(uv run python scripts/mosaic_eval.py args runs/wide/test/truth.json) \
    --out runs/wide/test/reports-wide.json
uv run python scripts/mosaic_eval.py score runs/wide/test/truth.json runs/wide/test/reports-wide.json --ignore-seams
```

`runs/rescue/` comes from [16](16-empty-field-rescue.md). The best rows of the `tune` replay are in
`18-fusion-val.json`.
