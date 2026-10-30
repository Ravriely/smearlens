# 17 · Wide fields: count red cells on tiles

Date: 30/10/2026. Code: `smearlens/analyze.py` (`tiles_per_side`, `plan_windows`, `in_core`),
`scripts/modal_app.py` (`Analyzer.detect_tiled`), `scripts/mosaic_eval.py`.
Scores: [`17-wide-field-tiling.json`](17-wide-field-tiling.json).

## Question

All the fields used so far are small: 360 px (PBC), 640 × 480 (BCCD), 256 px
(BCDD). A biologist who tries the demo with a photo from their own
microscope sends a wider field, with more and smaller cells. Does the
backend still work, and if not, what breaks?

## Setup

- **Wide fields with a truth.** No dataset with boxes has wide fields, so
  `scripts/mosaic_eval.py` tiles 4 test fields of one source into one 2 × 2
  image and shifts their boxes: 12 PBC (720 × 726), 4 BCCD (1280 × 960) and
  3 BCDD (512 × 512) mosaics, from 76 test fields. The same 76 fields alone
  are the reference.
- **Seams.** At the joins, cells cut by the edge of the original fields
  give fragments that a real slide does not have. White cell boxes, found
  or true, that touch a seam (± 3 px) are left out of the white cell score,
  on both sides. Red cell and platelet counts are compared as totals per
  image.
- **Why SAM 3 fails.** It resizes every image to 1008 px. On the single
  fields it finds red cells that span 150 to 200 px of that input (median:
  PBC 207, BCDD 204, BCCD 149). In the mosaics they span 104, 107 and 59 px.
- **Tiling rule** (`tiles_per_side`). A first pass on the whole image gives
  the median red cell (score ≥ 0.5, at least 3). If it spans less than
  130 px of the SAM input, the image is cut into n × n windows, n = 2 to 4,
  so that it spans about 180 px. If the first pass finds no red cell and the
  short side is at least 500 px, n = 2. Windows overlap by 20 % of a tile on
  each side. A detection is kept only by the window whose core (the n × n
  grid cell) holds its centre, so each cell is counted once.
- **Three variants**, on the same mosaics:
  - whole image (before);
  - every detection from the tiles;
  - final: white cells from the whole image, red cells, platelets and their
    size scale from the tiles.

## Results

Mosaics, white cells off the seams:

| Variant | White cell F1 | Precision | Recall | Red cells found / true | Red cell error per image | Platelets found / true |
|---|---|---|---|---|---|---|
| Whole image | 0.725 | 0.63 | 0.85 | 431 / 1124 | 36.5 | 39 / 45 |
| Every detection from tiles | 0.619 | 0.48 | 0.88 | 956 / 1124 | — | 36 / 45 |
| **Final** | **0.725** (same cells) | 0.63 | 0.85 | **1013 / 1124** | **7.9** | 33 / 45 |
| Same 76 fields one by one | 0.863 | 0.87 | 0.85 | 1005 / 1124 | 3.3 per field | 40 / 45 |

By source (final): PBC red cells 0 / 601 → 529 / 601 (error per image 50.1
→ 8.2); BCCD 285 → 330 / 332 (11.8 → 4.0); BCDD 146 → 154 / 191 (15.0 →
12.3).

The 76 single fields are never tiled (n = 1 for all of them), so their
results do not change. SAM time per mosaic: 0.26 s → 1.5 s.

## What it tells us

- On a wide field, the red cell count was the first thing to break: on PBC
  mosaics it fell to 0. With the tiles it is back at the level of single
  fields.
- Taking white cells from the tiles is worse: the detection probe of
  [09](09-detection-probe.md) accepts red cells and fragments seen at the
  tile scale (PBC precision 0.96 → 0.59 off the seams). The white cells
  stay on the whole image pass, which is the scale the probe was trained on.
- White cells on wide fields are still the weak spot: F1 0.725 against
  0.863 for the same fields one by one, mostly precision on BCCD and BCDD.
  Part of it is the mosaic itself (fragments near the seams that are not
  quite on them), part is real. A fix would need training the detection
  probe on tiles, or tiling only for the white cell candidates and
  scoring them on the whole-image crop.

## Notes and limits

- Mosaics are a stand-in for wide fields: four fields of one source, one
  stain and camera, with seams. A real wide field from another lab is
  still to test.
- 19 mosaics only (4 BCCD, 3 BCDD). The PBC numbers carry the result.
- Platelets lose a little (39 → 33 of 45); they are few, and the size
  filter of [14](14-red-cell-platelet-size-filter.md) now uses the tile
  scale.
- The thresholds (130 px, 180 px, 500 px, 20 % overlap) come from the
  single-field medians above, not from a search. Nothing was tuned on test.

## Reproduce

```bash
uv run python scripts/mosaic_eval.py build --out runs/mosaic
modal run scripts/modal_app.py --no-tiling $(uv run python scripts/mosaic_eval.py args runs/mosaic/truth.json) \
    --out runs/mosaic/reports.json
modal run scripts/modal_app.py $(uv run python scripts/mosaic_eval.py args runs/mosaic/truth.json) \
    --out runs/mosaic/reports-final.json
uv run python scripts/mosaic_eval.py score runs/mosaic/truth.json runs/mosaic/reports-final.json --ignore-seams
```

The "every detection from tiles" variant is not a switch of the backend:
it was the first version of `Analyzer.analyze` in this experiment.
