# 01 · Zero-shot baseline: Muse Spark on PBC

Date: 28/10/2026. Script: `scripts/baseten_zeroshot.py`. Raw answers:
[`01-zero-shot-baseline.csv`](01-zero-shot-baseline.csv).

## Question

How well does a general-purpose Meta vision-language model classify white
blood cells with no training at all? This number is the bar that SmearLens
must clear.

## Setup

| | |
|---|---|
| Model | `meta-models/muse-spark-open-1026` (hackathon ID on Baseten) |
| Endpoint | Baseten, OpenAI-compatible chat completions |
| Data | PBC dataset (Acevedo et al., 2020), 8 classes, May-Grünwald-Giemsa |
| Sample | 20 random cells per class from the **test** split (`seed=0`), 160 cells |
| Prompt | Fixed list of the 8 classes, one sentence of reasoning, then `LABEL: <class>` |
| Budget | `max_tokens=4000` per call (it is a reasoning model, see notes) |

The test split comes from `scripts/prepare_pbc.py`. It keeps the TXL-PBC
splits for the PBC images that TXL-PBC also uses, so later experiments that
train on TXL-PBC do not leak into this test set.

## Results

**Accuracy: 104/160 = 65.0 %.** Macro F1: 0.64.

| Class | Correct /20 | Precision | Recall | F1 |
|---|---|---|---|---|
| monocyte | 18 | 0.75 | 0.90 | 0.82 |
| neutrophil | 18 | 0.56 | 0.90 | 0.69 |
| eosinophil | 16 | 0.94 | 0.80 | 0.86 |
| lymphocyte | 16 | 0.47 | 0.80 | 0.59 |
| basophil | 14 | 0.56 | 0.70 | 0.62 |
| platelet | 11 | 1.00 | 0.55 | 0.71 |
| erythroblast | 7 | 0.78 | 0.35 | 0.48 |
| ig (immature granulocyte) | 4 | 0.80 | 0.20 | 0.32 |

Confusion matrix (rows = true class, columns = prediction):

| true \ pred | baso | eosino | erythro | ig | lympho | mono | neutro | plt | unparsed |
|---|---|---|---|---|---|---|---|---|---|
| basophil | **14** | 1 | 0 | 0 | 1 | 0 | 4 | 0 | 0 |
| eosinophil | 0 | **16** | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| erythroblast | 0 | 0 | **7** | 0 | 11 | 1 | 1 | 0 | 0 |
| ig | 8 | 0 | 1 | **4** | 0 | 3 | 2 | 0 | 2 |
| lymphocyte | 1 | 0 | 0 | 0 | **16** | 2 | 1 | 0 | 0 |
| monocyte | 0 | 0 | 1 | 0 | 0 | **18** | 1 | 0 | 0 |
| neutrophil | 0 | 0 | 0 | 1 | 0 | 0 | **18** | 0 | 1 |
| platelet | 2 | 0 | 0 | 0 | 6 | 0 | 1 | **11** | 0 |

Cost and speed: 248,749 tokens in total (1,555 per cell on average, most of it
reasoning). Median latency 5.3 s per cell, p90 10.6 s. The full run took
2 min 50 s with 6 parallel requests.

## What it tells us

- The model already knows the common, mature cells (monocytes, neutrophils,
  eosinophils, lymphocytes): recall of 80-90 %.
- It fails on the cells that matter most for a biologist's review:
  **immature granulocytes (20 % recall)**, most often called basophils, and
  **erythroblasts (35 %)**, most often called lymphocytes. A circulating
  erythroblast or immature granulocyte is exactly what a reviewer must not miss.
- "lymphocyte" and "neutrophil" work as fallback answers (precision 0.47 and
  0.56): when the model is unsure, it picks a common class.
- A general VLM is a usable assistant for the easy cells. It is not enough for
  the review step. This is the gap SmearLens targets with a specialised
  classifier (linear probe on the vision encoder, then a LoRA fine-tune).

## Notes and limits

- Muse Spark is a reasoning model. With `max_tokens=200`, all tokens went to
  reasoning and the answer was empty. At 2000, 1 answer out of 8 was still
  empty, so the run uses 4000. Three answers stayed unparsed (counted as wrong).
- 20 cells per class gives a rough estimate: a 95 % interval on one class
  recall is about ±20 points. The global accuracy (160 cells) is about ±7 points.
- PBC crops are centred on one cell, from one hospital and one scanner. Real
  smears are harder: several cells per field, other stains, artefacts.
- No prompt tuning was done. This is a floor, not the best the model can do
  zero-shot.

## Reproduce

```bash
export BASETEN_API_KEY=...
uv run python scripts/prepare_pbc.py
uv run python scripts/baseten_zeroshot.py --per-class 20 --workers 6
```
