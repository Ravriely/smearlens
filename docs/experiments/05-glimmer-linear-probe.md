# 05 · Linear probe on the Muse Glimmer perception encoder

Date: 28/10/2026. Script: `scripts/modal_glimmer_probe.py` (Modal, 1× L40S).
Metrics: [`05-glimmer-probe-metrics.json`](05-glimmer-probe-metrics.json).

## Question

The zero-shot baseline ([01](01-zero-shot-baseline.md)) and the pipeline
([04](04-end-to-end-pipeline.md)) fail on the rare and look-alike cells. Do
the features of a Meta open-weights vision encoder already separate them, so
that a simple linear layer is enough?

## Setup

| | |
|---|---|
| Encoder | Perception encoder of `meta-models/Muse-Glimmer-30B` (ViT-G/14, 1.8B parameters, 50 layers), **frozen** |
| Weights | Only the vision tower is downloaded: one contiguous 3.7 GB byte range of the first 50 GB shard, fetched with HTTP range requests and rewritten as a small safetensors file |
| Code | `transformers` main at commit `674c0e39` (Muse Glimmer is not in a release yet) |
| Feature | Mean of the encoder's last hidden state over all image tokens, 6,144 dimensions |
| Probe | One linear layer (6,144 → 8), standardised features, class-balanced cross-entropy, AdamW, 300 full-batch steps; weight decay picked on val among 1e-4 … 1e-1 |
| Data | PBC, all 17,092 cells: train 11,873, val 2,659, test 2,560 (splits from `scripts/prepare_pbc.py`, test = same split as 01) |
| Compute | One Modal L40S run, about 15 min, about $0.60 |

## Results

**Test accuracy: 97.6 %** (2,499/2,560). Macro F1: **0.974**. Val accuracy: 97.7 %
(weight decay 0.1).

| Class | Precision | Recall | F1 | Zero-shot recall (01) |
|---|---|---|---|---|
| eosinophil | 1.00 | 1.00 | 1.00 | 0.80 |
| platelet | 1.00 | 1.00 | 1.00 | 0.55 |
| erythroblast | 0.98 | 0.99 | 0.99 | **0.35** |
| neutrophil | 0.98 | 0.96 | 0.97 | 0.90 |
| lymphocyte | 0.97 | 0.96 | 0.97 | 0.80 |
| basophil | 0.95 | 0.97 | 0.96 | 0.70 |
| monocyte | 0.96 | 0.96 | 0.96 | 0.90 |
| ig (immature granulocyte) | 0.95 | 0.95 | 0.95 | **0.20** |

Confusion matrix (rows = true class):

| true \ pred | baso | eosino | erythro | ig | lympho | mono | neutro | plt |
|---|---|---|---|---|---|---|---|---|
| basophil | **179** | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| eosinophil | 0 | **476** | 0 | 0 | 0 | 0 | 0 | 0 |
| erythroblast | 0 | 0 | **239** | 0 | 0 | 1 | 2 | 0 |
| ig | 3 | 0 | 2 | **400** | 1 | 5 | 8 | 0 |
| lymphocyte | 3 | 0 | 1 | 0 | **169** | 2 | 1 | 0 |
| monocyte | 1 | 1 | 0 | 2 | 3 | **206** | 1 | 0 |
| neutrophil | 2 | 0 | 1 | 14 | 0 | 1 | **482** | 0 |
| platelet | 0 | 0 | 0 | 0 | 1 | 0 | 0 | **348** |

## What it tells us

- **The Meta encoder already knows the difference.** A single linear layer on
  frozen Muse Glimmer features goes from 65 % (zero-shot Muse Spark, 01) to
  **97.6 %**. Immature granulocytes go from 20 % to 95 % recall, erythroblasts
  from 35 % to 99 %: exactly the cells that a reviewer must not miss.
- The remaining errors make clinical sense: immature granulocytes versus
  neutrophils (8 + 14), the maturation continuum where human experts also
  disagree.
- No fine-tuning of the 1.8B encoder was needed. Training the probe takes
  seconds; the cost is the one-time feature extraction.

For the pitch: open weights plus one linear layer, trained for about $0.60,
turn a general model into a specialist for blood cells (97.6 % on 8 classes).

## Notes and limits

- **In-distribution result.** Train and test come from the same dataset (one
  hospital, one scanner, centred crops). The real test is on crops cut by SAM
  from other datasets (BCCD, BCDD), which is the next experiment.
- PBC holds only normal cells. Blasts and other pathological cells are not in
  the 8 classes; the review list must still send unknown shapes to a human.
- One hidden macOS file (`.DS_169665.jpg`) in the PBC archive made the first run
  fail; `prepare_pbc.py` now skips hidden files. It was in the val split, so
  the test split of 01 did not change.
- The probe has no "not a white cell" class: in the pipeline, Muse Spark keeps
  the role of filtering false SAM candidates.

## Next

Plug the probe into the pipeline of [04](04-end-to-end-pipeline.md): SAM 3
candidates → Muse Spark filter (white cell or not) → Glimmer probe for the
type, and measure the gain on the 82 TXL-PBC test images.

## Reproduce

```bash
uv run python scripts/prepare_pbc.py
modal run scripts/modal_glimmer_probe.py            # full run
modal run scripts/modal_glimmer_probe.py --per-class 5   # smoke test
```
