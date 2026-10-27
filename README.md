# SmearLens

A review assistant for peripheral blood smears. SmearLens helps lab biologists
go through slides faster: it finds every cell, proposes a differential count,
and flags the slides and cells that need a closer human look first.

> **SmearLens is a lab workflow tool, not a diagnostic device.** It never gives
> a diagnosis or medical advice. Every count it proposes is a draft that a
> biologist reviews and validates.

Built for the Meta AI Developer Hackathon 2026 (Indie track).

## How it works

1. **Segment.** SAM 3.1 finds and segments every cell on the smear image.
2. **Classify.** A fine-tuned Meta open-weights model labels each white blood
   cell (neutrophil, lymphocyte, monocyte, eosinophil, basophil, ...).
3. **Review.** An agent drafts the differential count and lists what the
   biologist should check first: low-confidence cells, atypical shapes,
   counts outside the usual range.

## Status

Work in progress. The first step validates cell segmentation with SAM 3.1.

## Quick start

```bash
uv sync --extra dev
uv run python scripts/fetch_data.py
uv run pytest
```

Segmentation needs a CUDA GPU and access to the gated
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) checkpoint:

```bash
uv sync --extra sam
export HF_TOKEN=...
uv run python scripts/sam3_validate.py --limit 10
```

## Data

SmearLens only uses public datasets with a clear license. See
[DATA.md](DATA.md) for sources, licenses and citations.

## License

MIT. Dataset licenses are listed in [DATA.md](DATA.md).
