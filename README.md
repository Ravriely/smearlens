# SmearLens

A review assistant for peripheral blood smears. SmearLens helps lab biologists
go through slides faster: it finds every cell, proposes a differential count,
and flags the slides and cells that need a closer human look first.

> **SmearLens is a lab workflow tool, not a diagnostic device.** It never gives
> a diagnosis or medical advice. Every count it proposes is a draft that a
> biologist reviews and validates.

Built for the Meta AI Developer Hackathon 2026 (Indie track).

## How it works

1. **Segment.** SAM 3 finds and segments every cell on the smear image.
2. **Classify.** A fine-tuned Meta open-weights model labels each white blood
   cell (neutrophil, lymphocyte, monocyte, eosinophil, basophil, ...).
3. **Review.** An agent drafts the differential count and lists what the
   biologist should check first: low-confidence cells, atypical shapes,
   counts outside the usual range.

## Status

Work in progress. Segmentation uses SAM 3 (`facebook/sam3`); see the experiments below.

## Experiments

| # | What | Result |
|---|---|---|
| [01](docs/experiments/01-zero-shot-baseline.md) | Zero-shot Muse Spark on PBC, 160 cells | 65.0 % accuracy, fails on immature granulocytes (20 %) and erythroblasts (35 %) |
| [02](docs/experiments/02-sam3-segmentation.md) | SAM 3.1 text-prompt segmentation, 10 TXL-PBC images | RBC F1 0.70; the WBC prompt detects every cell (precision 0.02); no platelets |
| [03](docs/experiments/03-sam3-prompt-sweep.md) | SAM 3 prompt/threshold sweep, full test split (82 images) | "cell with a large purple nucleus": WBC F1 0.04 → 0.37, count error 17 → 2.2 per image; platelets recall 0.87; SAM 3.1 box prompts are random, SAM 3 is the default |

## Quick start

```bash
uv sync --extra dev
uv run python scripts/fetch_data.py
uv run pytest
```

Segmentation needs a CUDA GPU and access to the gated
[`facebook/sam3`](https://huggingface.co/facebook/sam3) checkpoint (or `facebook/sam3.1` with `--version sam3.1`):

```bash
uv sync --extra sam
export HF_TOKEN=...
uv run python scripts/sam3_validate.py --limit 10
```

Without a local GPU, run it on [Modal](https://modal.com):

```bash
uv sync --extra modal
uv run modal secret create huggingface HF_TOKEN=...
uv run modal run scripts/modal_sam3.py --limit 10
```

## Data

SmearLens only uses public datasets with a clear license. See
[DATA.md](DATA.md) for sources, licenses and citations.

## License

MIT. Dataset licenses are listed in [DATA.md](DATA.md).
