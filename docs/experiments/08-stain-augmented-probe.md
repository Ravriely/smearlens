# 08 · Stain-augmented probe against domain shift

Date: 28/10/2026. Code: `smearlens/augment.py`, `scripts/modal_glimmer_probe.py::augmented`,
`scripts/pipeline_eval.py --probe runs/glimmer-aug/probe_aug.npz`.
Metrics: [`08-augmented-probe-metrics.json`](08-augmented-probe-metrics.json),
pipeline: [`08-summary_glimmer_rbc_aug.json`](08-summary_glimmer_rbc_aug.json).

## Question

The review agent ([07](07-review-agent.md)) showed that the probe, trained on
PBC only, calls most BCCD white cells "monocyte". Does training with stain and
colour augmentation make it robust to another lab's slides?

## Setup

**No public BCCD type labels with a clear license** (the original BCCD
annotations only say WBC/RBC/platelet), so the shift is measured two ways:

1. **Simulated other lab** on the full PBC test split (2,560 cells): each test
   image goes through a strong random perturbation (strength 1.5, its own
   seed): stain jitter in haematoxylin-eosin space (Ruifrok & Johnston
   vectors), brightness/contrast/saturation, ±22 % scale, flips and rotations.
   Optionally, colours are then normalised back to PBC (Reinhard, LAB space).
2. **BCCD in the real pipeline**: agreement with Muse Spark and type
   distribution (weak signal: Muse Spark is only 65 % right on its own).

Probes compared (same encoder, same training recipe as [05](05-glimmer-linear-probe.md)):

| Probe | Training images |
|---|---|
| clean | 11,873 PBC train images |
| **augmented** | the same + 2 augmented copies of each (strength 1.0, ±15 % scale), 35,619 |

## Results

Accuracy on the PBC test split (2,560 cells):

| Probe | Clean test | Simulated other lab | Other lab + Reinhard |
|---|---|---|---|
| clean | 97.6 % | 91.8 % | 56.6 % |
| **augmented** | **97.6 %** | **97.4 %** | 61.3 % |

Recall per class on the simulated other lab:

| Class | clean probe | augmented probe |
|---|---|---|
| **lymphocyte** | **48.9 %** | **95.5 %** |
| basophil | 90.2 % | 97.8 % |
| ig | 91.6 % | 95.2 % |
| monocyte | 92.1 % | 95.3 % |
| erythroblast | 95.9 % | 97.9 % |
| platelet | 96.8 % | 99.1 % |
| neutrophil | 94.4 % | 97.0 % |
| eosinophil | 99.6 % | 99.8 % |

In the pipeline (82 TXL-PBC test images, crops scaled on red cells):

| | clean probe ([06](06-pipeline-with-glimmer-probe.md)) | augmented probe |
|---|---|---|
| Right type, PBC cells (55) | 72.7 % | 72.7 % |
| Lymphocytes right (9) | 2 | **6** |
| BCCD cells called monocyte (26) | 17 (65 %) | 7 (27 %) |
| BCCD cells called neutrophil | 1 | 8 |
| BCCD agreement with Muse Spark | 3/26 | 10/26 |

Cost: one Modal L40S run (about 30,000 embeddings), about $0.75.

## What it tells us

- **Augmentation fixes the simulated shift at no cost on clean data**: 91.8 %
  → 97.4 % under a strong colour, stain and scale shift, still 97.6 % on clean
  images. Lymphocytes were the weak point (49 % → 95 %): they are small and
  their recognition hangs on size and nucleus colour, which is exactly what
  the shift changes.
- **On real BCCD slides the "everything is a monocyte" failure is gone**
  (65 % → 27 % monocytes, neutrophils 1 → 8, agreement with Muse Spark
  3 → 10 of 26). Without labels we cannot give an accuracy, but the
  distribution now looks like a normal smear.
- **Reinhard normalisation hurts** (56-61 %). Matching global LAB statistics
  shifts the cell colours themselves; the probe relies on them. Not used.
- In the PBC pipeline, 4 more lymphocytes are right and 4 other cells moved
  (2 monocytes called lymphocytes, 1 called ig, 1 neutrophil called ig): same
  total, a better balance.

The augmented probe replaces the clean one as default.

## Notes and limits

- The "other lab" is simulated. Its perturbations come from the same family as
  the training augmentation (different strength and seed). A real external
  test set with type labels is still missing.
- **BCDD gives no candidates at all**: SAM 3 finds no "cell with a large purple
  nucleus" on these 256 px images, even at threshold 0.3, so the pipeline
  misses all 14 BCDD white cells. Low resolution is the likely cause (upscale
  before SAM).
- 7 BCCD candidates are now called `platelet` by the probe; they are left out
  of the white cell count.

## Reproduce

```bash
modal volume put smearlens-hf-cache runs/glimmer/pbc_features.npz /glimmer/pbc_features.npz
modal run scripts/modal_glimmer_probe.py::augmented --copies 2
uv run python scripts/pipeline_eval.py --images 200 --typer glimmer \
  --probe runs/glimmer-aug/probe_aug.npz --tag aug
```
