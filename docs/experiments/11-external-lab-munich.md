# 11 · Another lab: Munich (AML-Cytomorphology_LMU)

Date: 28/10/2026. Scripts: `scripts/lmu_eval.py`, `scripts/lmu_adapt.py`
(run on Modal with `scripts/modal_lmu_adapt.py`), `scripts/lmu_zoom_eval.py`.
Numbers: [`11-lmu-external.json`](11-lmu-external.json),
[`11-lmu-adaptation.json`](11-lmu-adaptation.json),
[`11-lmu-zoom.json`](11-lmu-zoom.json).

## Question

All the numbers so far come from PBC (Barcelona). A real lab has its own
microscope, camera and stain. What happens when the type probe sees cells
from a lab it never saw? Does it know when it is wrong? How many labelled
cells does a new lab need to give?

## Data

AML-Cytomorphology_LMU (Matek et al., Munich University Hospital, TCIA,
CC BY 3.0, DOI [10.7937/tcia.2019.36f5o9ld](https://doi.org/10.7937/tcia.2019.36f5o9ld)):
18,365 single-cell images (400 × 400 px) from 100 AML patients and
100 controls, labelled by experts.

- 15,045 cells of types the probe knows. Mapping: segmented and band
  neutrophils → neutrophil; metamyelocytes, myelocytes, promyelocytes →
  immature granulocyte; basophil, eosinophil, erythroblast, lymphocyte,
  monocyte as is.
- 3,320 cells of types the probe never saw: myeloblasts (3,268),
  monoblasts (26), atypical lymphocytes (11), smudge cells (15).

The probe is the stain-augmented Glimmer probe of
[08](08-stain-augmented-probe.md), with the temperature of
[10](10-review-coverage.md). Nothing was tuned on Munich for the first test.

## 1 · Zero-shot external test

| Measure | PBC test (in-lab) | Munich (other lab) |
|---|---|---|
| Accuracy | 97.6 % | **83.7 %** |
| Balanced accuracy | 97.5 % | 72.3 % |
| Calibration error (ECE) | 0.6 % | 9.2 % |

Per type on Munich:

| Type | Cells | Recall | Most common mistake |
|---|---|---|---|
| Neutrophil | 8,593 | 96.8 % | monocyte (83) |
| Erythroblast | 78 | 96.2 % | — |
| Monocyte | 1,789 | 91.6 % | immature granulocyte (54) |
| Immature granulocyte | 145 | 75.9 % | monocyte (29) |
| Lymphocyte | 3,937 | **56.9 %** | erythroblast (988) |
| Basophil | 79 | 48.1 % | neutrophil (18) |
| Eosinophil | 424 | 40.6 % | neutrophil (154) |

## 2 · Does the review list still catch the errors?

We keep the review rule fixed from PBC: a cell goes to the biologist when
its confidence is below the PBC 10 % cutoff (0.960). On Munich this rule
sends **28 %** of the cells to review (more doubt, as it should) and the
errors left in the rest fall from **16.3 % to 5.4 %**.

| Cells reviewed | Errors left | Errors caught |
|---|---|---|
| 0 % | 16.3 % | 0 % |
| 10 % | 12.0 % | 34 % |
| 30 % | 4.7 % | 80 % |
| 50 % | 1.3 % | 96 % |

## 3 · Blasts: types the probe has never seen

There is no "blast" class in PBC, so the probe can only give a wrong name.
What matters is that it hesitates:

| Unseen type | Cells | Sent to review by the fixed rule | Mean confidence |
|---|---|---|---|
| Myeloblast | 3,268 | **71 %** | 0.81 |
| Monoblast | 26 | 38 % | 0.91 |
| Atypical lymphocyte | 11 | 45 % | 0.89 |
| Smudge cell | 15 | 60 % | 0.88 |

Known types are sent to review 28 % of the time, myeloblasts 71 % of the
time. A slide full of blasts gives a long review list, and that is
the signal the biologist needs. The agent ([07](07-review-agent.md)) adds
its own flag when many cells are uncertain. 29 % of myeloblasts still pass
with high confidence (mostly called monocytes), so SmearLens is not a blast
detector and must not be presented as one.

## 4 · Adapting to a new lab with a few labelled cells

A lab labels N cells per type; we retrain the linear head, warm-started
from the PBC probe, on PBC train + those cells (the lab cells weighted to
the same total as PBC). The encoder stays frozen. Test on all the other
Munich cells of known types. 3 random draws per N.

| Labelled per type | Total labelled | Accuracy | Balanced accuracy | Lymphocyte | Eosinophil | Basophil |
|---|---|---|---|---|---|---|
| 0 | 0 | 83.7 % | 72.3 % | 57 % | 41 % | 48 % |
| 5 | 35 | 84.8 % ± 1.3 | 77.5 % | 66 % | 82 % | 72 % |
| **10** | **70** | **89.4 % ± 1.0** | **84.7 %** | **83 %** | **87 %** | **78 %** |
| 25 | 175 | 89.7 % ± 2.2 | 87.3 % | 81 % | 91 % | 77 % |
| 50 | 350 | 91.8 % ± 0.7 | 88.9 % | 90 % | 91 % | 85 % |
| 100 | 700 | 92.6 % ± 0.3 | 88.5 % | 92 % | 91 % | 79 % |

Basophils have only 79 cells, so at 50 and 100 shots the test keeps very
few of them and the number is noisy.

## 5 · Is it the magnification?

Hypothesis: Munich cells are photographed at another scale, and lymphocytes
look like erythroblasts (round, dark nucleus) at the wrong size. Test: 300
cells per type, rescaled by a factor from 0.5 to 1.2 (LANCZOS) and centred
in the 360 px PBC field, padded with the background colour.

| Zoom | 0.5 | 0.6 | 0.75 | 0.9 | 1.0 | 1.2 |
|---|---|---|---|---|---|---|
| Balanced accuracy | 65.7 % | 69.3 % | 69.4 % | **71.5 %** | 70.1 % | 67.1 % |
| Lymphocyte recall | 57 % | 59 % | 52 % | 54 % | 47 % | 31 % |

No zoom fixes the lymphocytes. The best factor (0.9, the native 400 px image
resized to 360) is the same as the untouched image. **The hypothesis is
rejected**: the gap is in how the cells look (stain, camera, focus), not in
their size. A frozen encoder plus a linear head cannot fully close it,
which is why we try a light fine-tune of the encoder (LoRA) next.

## What it tells us

- **Out of the box on another lab: 84 % right, and it knows when it doubts.**
  The fixed review rule catches two thirds of the errors (16.3 % → 5.4 %).
- **Blasts are flagged, not named.** 71 % of myeloblasts go to the review
  list.
- **A new lab needs about 70 labelled cells** (10 per type, a few minutes of
  a biologist's time) to go from 84 % to 89 %, with lymphocytes from 57 % to
  83 %. 350 cells give 92 %.
- Calibration does not transfer: ECE goes from 0.6 % to 9.2 %. The probe is
  over-confident on the new lab, so the review cutoff should be re-fitted on
  the lab's own labelled cells.

For the pitch: "On a hospital it has never seen, SmearLens gets 84 % right
and sends the doubtful cells, including 71 % of leukaemia blasts, to the
biologist. Ten labelled cells per type bring it to 89 %."

## Notes and limits

- File names carry no patient ID in a usable form, so adaptation splits are
  by image, not by patient. Cells from the same patient can be in both the
  labelled and test sets, which may flatter the adaptation numbers a little.
- Munich images are single centred cells, not whole slides: this tests the
  type probe only, not detection.
- The zoom test uses a class-balanced subsample, so its accuracy is not
  comparable with the full-set accuracy above. Compare balanced accuracy.
- Glimmer's processor falls back from LANCZOS to BICUBIC on GPU (same for all
  our features, so no bias between experiments).

## Reproduce

```bash
modal run scripts/modal_glimmer_probe.py::lmu                # features, ~1.5 h on L40S
PYTHONPATH=.:scripts uv run python scripts/lmu_eval.py
modal run scripts/modal_lmu_adapt.py                         # 16 CPU, a few minutes
modal run --detach scripts/modal_glimmer_probe.py::lmu_zoom
PYTHONPATH=.:scripts uv run python scripts/lmu_zoom_eval.py
```
