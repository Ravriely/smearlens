# 19 · A "suspected blast" class learnt from Munich

Date: 01/11/2026. Script: `scripts/modal_blasts.py` (Modal: one L40S for the features, one L4 for
the probes). Numbers: [`19-blasts-metrics.json`](19-blasts-metrics.json),
[`19-blasts.csv`](19-blasts.csv), [`19-blast-threshold.csv`](19-blast-threshold.csv).

## Question

The type probe knows the 8 PBC types. A blast is not one of them, so the best it could do so far was
to hesitate: 71 % of Munich myeloblasts went to the review queue
([11](11-external-lab-munich.md)), drowned among 28 % of the normal cells. Can a ninth class,
"blast", put the blasts at the top of the review list, without losing anything on PBC?

## Data and setup

- PBC (Barcelona): the same train/val/test split as [05](05-glimmer-linear-probe.md), plus the two
  stain-augmented copies of the train cells and the simulated "other lab" test of
  [08](08-stain-augmented-probe.md), recomputed with the same seeds.
- Munich (AML-Cytomorphology_LMU, CC BY 3.0, DOI
  [10.7937/tcia.2019.36f5o9ld](https://doi.org/10.7937/tcia.2019.36f5o9ld)): 18,365 single cells.
  **Blast = myeloblast (MYO, 3,268) + monoblast (MOB, 26).** Both are blasts for the biologist, and
  26 monoblasts are too few for a class of their own. Atypical lymphocytes and smudge cells (26 in
  all) are never trained on. The other types are mapped to the PBC types as in 11.
- Munich split by image, stratified by type, seed 0: 60 % train, 20 % val, 20 % test (3,667 test
  cells, 659 of them blasts). **The test split is used for the numbers below only**: weight decay and
  temperature are picked on PBC val + Munich val, the blast threshold on Munich val and PBC val.
- Same recipe as the deployed probe: frozen Glimmer vision features, linear head, class-balanced
  loss, weight decay picked on val. The review cutoff keeps the rule of the deployed probe (10 % of
  PBC test cells below it).

Five heads compared:

| Head | Classes | Trained on |
|---|---|---|
| current | 8 | the deployed probe ([08](08-stain-augmented-probe.md)) |
| pbc8_retrained | 8 | PBC train + augmented copies (sanity check: same as current) |
| adapted8 | 8 | + Munich train cells of the 8 known types |
| blast9_blasts_only | 9 | PBC + augmented + Munich train blasts only |
| **blast9** | 9 | PBC + augmented + Munich train cells of all types, blasts as the ninth class |

## Results

### 1 · Nothing lost on PBC

| Head | PBC test (2,560) | PBC cells called blast | Simulated other lab | Called blast there |
|---|---|---|---|---|
| current | 97.62 % | — | 97.38 % | — |
| adapted8 | 97.77 % | — | 97.58 % | — |
| **blast9** | **97.73 %** | **0** | 97.34 % | **0** (1 cell at threshold 0.3) |

The ninth class costs nothing on PBC (+0.1 point, within noise) and never fires on normal PBC
cells, clean or stain-shifted. The review curve on PBC stays the same: reviewing 10 % of cells leaves
0.43 % errors (0.30 % with the current probe), 20 % leaves 0.15 %.

### 2 · Blasts on the Munich test split

"Review list" means the cell is below the review cutoff, the only blast signal of the current probe.

| Head | Blast signal | Recall | Precision | Other cells flagged |
|---|---|---|---|---|
| current | review list | 72.4 % | 35.5 % | 28.7 % |
| adapted8 | review list | 43.1 % | 51.2 % | 9.0 % |
| **blast9** | type = blast (argmax) | **95.6 %** | **90.8 %** | **2.1 %** |
| **blast9** | p(blast) ≥ 0.3 (**deployed**) | **97.4 %** | **88.0 %** | 2.9 % |
| blast9 | blast or review list | 99.4 % | 62.3 % | 13.2 % |

Myeloblasts: 96.0 % recall at argmax. Monoblasts: 2 of 5 test cells called blast, the other 3 go to
the review list (5 of 5 caught by "blast or review"); with 26 monoblasts in total this number is
anecdotal.

Blasts that are missed are called monocyte (18), lymphocyte (9) or immature granulocyte (2).
Cells wrongly called blast (argmax): lymphocytes 27 of 787, monocytes 21 of 358, immature
granulocytes 8 of 29, basophils 2 of 16, neutrophils 5 of 1,719. Promyelocytes and blasts are close
cousins on a smear, so the immature granulocyte confusion is expected.

**Threshold.** A cell is a suspected blast when p(blast) ≥ 0.3, even if another type scores higher.
Chosen on val: the highest threshold that keeps val recall near 98 % while flagging at most 0.05 %
of PBC val cells.

| p(blast) ≥ | Val recall | Val precision | PBC val flagged | Test recall | Test precision | PBC test flagged |
|---|---|---|---|---|---|---|
| 0.1 | 99.5 % | 79.3 % | 0.11 % | 99.2 % | 79.0 % | 0 % |
| **0.3** | **97.9 %** | **87.4 %** | **0.04 %** | **97.4 %** | **88.0 %** | **0 %** |
| 0.5 | 95.0 % | 90.5 % | 0 % | 95.6 % | 91.3 % | 0 % |
| 0.9 | 80.4 % | 95.7 % | 0 % | 81.8 % | 97.5 % | 0 % |

### 3 · Why Munich normal cells must be in the training set

If the ninth class only sees Munich blasts (blast9_blasts_only), it learns "Munich look = blast":
it calls **71 % of Munich normal cells blast** and the known-type accuracy on Munich falls to 29 %.
All blasts come from one lab, so the class can only be told apart from the lab style if normal cells
of the same lab are there too. With them (blast9), the known types on Munich go from 83.6 % to
95.4 % (adaptation, as in [11](11-external-lab-munich.md) §4; adapted8 without blasts: 96.8 %).
The price of the ninth class on Munich normal cells is 1.4 points, mostly immature granulocytes
(76 % → 62 %).

### 4 · On the deployed app

After the deploy, through `POST /api/analyze` ([`19-live-check.json`](19-live-check.json)):

- Example slide (20 PBC test fields): 20 of 20 labelled cells typed right, **0 suspected blast**.
- Munich test images (whole 400 px images, through SAM 3 and the detection probe): 4 myeloblasts →
  3 suspected blasts, 1 with no white cell found (detection miss); 1 monoblast → suspected blast
  (p = 0.48); 1 lymphocyte, 1 neutrophil, 1 monocyte typed right, none called blast.
- `POST /api/review` on a myeloblast image: the cell is flagged high, "suspected blast, to review",
  the report says the second opinion has no blast class and names no disease (15k tokens).

## What it tells us

- **A suspected blast class is a real gain for the review list**: on Munich, 96-97 % of blasts are
  named "suspected blast" and put first, with about 9 in 10 flags right, instead of 72 % hidden in a
  queue where only 1 flag in 3 is a blast.
- **No cost on PBC**: 97.73 % vs 97.62 %, and zero blast flags on 2,560 normal PBC cells, clean or
  stain-shifted.
- Decision: **integrated** as the deployed type probe (`models/type_probe_blast.npz`, threshold 0.3).
  Every suspected blast is in the review queue, flagged high by the agent before it runs, with the
  reason "suspected blast, to review", and an indicative range of 0 %. The agent never names a disease
  (regex guard extended to disease names and acronyms, with tests).

For the pitch: "On a hospital it learnt blasts from, SmearLens puts 97 % of blasts at the top of the
review list, and never raised a blast flag on 2,560 normal cells from another lab." Never: "it detects
leukaemia".

## Notes and limits

- **No patient ID.** Munich file names carry none, so the split is by image: cells of the same
  patient are in train and test. A patient with AML gives many blasts, so the blast recall may be
  flattered, maybe by a lot. This is the main limit.
- **One lab for blasts.** Every training blast comes from Munich. The zero flags on PBC show that the
  class does not fire on another lab's normal cells, but nothing measures yet whether it finds blasts
  from a third lab. A blast from another hospital may land in the review list (low confidence) rather
  than in the blast class.
- Munich images are single centred cells (400 × 400 px), not fields: this tests the type probe only,
  not detection. On a field, a blast that the detection probe drops is never typed.
- The review cutoff (0.916) follows the same 10 %-of-PBC rule as before; Munich still has its own
  calibration gap ([11](11-external-lab-munich.md)).
- Monoblasts (5 test cells), atypical lymphocytes and smudge cells are too few to conclude.
- Same BICUBIC fallback in the Glimmer processor as every other experiment.

## Reproduce

```bash
modal run --detach scripts/modal_blasts.py::features --part lmu       # ~35 min on L40S
modal run --detach scripts/modal_blasts.py::features --part pbc_aug   # ~15 min on L40S
modal run --detach scripts/modal_blasts.py::evaluate                  # ~3 min on L4
modal volume get smearlens-hf-cache glimmer/blasts runs/blasts
```

The deployed probe file is `glimmer/blasts/probe_blast9.npz` with the temperature, review cutoff and
blast threshold of `metrics.json` added as arrays.
