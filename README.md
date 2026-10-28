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

Work in progress. The pipeline runs end to end with no training (SAM 3 + Muse Spark); see the experiments below.

## Experiments

| # | What | Result |
|---|---|---|
| [01](docs/experiments/01-zero-shot-baseline.md) | Zero-shot Muse Spark on PBC, 160 cells | 65.0 % accuracy, fails on immature granulocytes (20 %) and erythroblasts (35 %) |
| [02](docs/experiments/02-sam3-segmentation.md) | SAM 3.1 text-prompt segmentation, 10 TXL-PBC images | RBC F1 0.70; the WBC prompt detects every cell (precision 0.02); no platelets |
| [03](docs/experiments/03-sam3-prompt-sweep.md) | SAM 3 prompt/threshold sweep, full test split (82 images) | "cell with a large purple nucleus": WBC F1 0.04 → 0.37, count error 17 → 2.2 per image; platelets recall 0.87; SAM 3.1 box prompts are random, SAM 3 is the default |
| [04](docs/experiments/04-end-to-end-pipeline.md) | End to end: SAM 3 candidates + Muse Spark labels, 82 test images | WBC F1 0.25 → 0.72, count error 4.98 → 0.55 per image, right type for 58 % of WBCs (70 % of kept cells) |
| [05](docs/experiments/05-glimmer-linear-probe.md) | Linear probe on frozen Muse Glimmer vision features, PBC | 97.6 % test accuracy (zero-shot: 65 %); immature granulocytes 20 % → 95 %, erythroblasts 35 % → 99 % |
| [06](docs/experiments/06-pipeline-with-glimmer-probe.md) | Pipeline with the Glimmer probe, crops scaled on red cells | Right type end to end 58 % → 72.7 % (87 % of kept cells), same detection (F1 0.73, count error 0.52) |
| [07](docs/experiments/07-review-agent.md) | Review agent: Muse Spark with 6 tools prepares the slide review | Second opinions, review list and report in about 1 min per slide, no diagnostic wording; flagged 21/24 disagreements on BCCD (probe domain shift) |
| [08](docs/experiments/08-stain-augmented-probe.md) | Stain/colour/scale augmentation of the probe | Simulated other lab 91.8 % → 97.4 % (lymphocytes 49 % → 95 %), clean unchanged (97.6 %); BCCD monocyte collapse fixed (65 % → 27 %) |
| [09](docs/experiments/09-detection-probe.md) | Glimmer probe for "white cell or not" over all SAM 3 candidates | WBC F1 0.72 → 0.79, precision 0.78 → 0.95, count error 0.55 → 0.38, zero LLM calls per cell |
| [10](docs/experiments/10-review-coverage.md) | Calibration and review curve of the type probe, 2,560 PBC cells | Calibrated (ECE 0.6 %); reviewing the 10 % least sure cells leaves 0.3 % errors and catches 89 % of them |
| [11](docs/experiments/11-external-lab-munich.md) | External lab: 15,045 Munich cells + 3,320 blasts and other unseen types | 83.7 % zero-shot, 71 % of myeloblasts sent to review; 70 labelled cells → 89.4 %; magnification is not the cause |
| [12](docs/experiments/12-glimmer-lora.md) | LoRA fine-tune of the Glimmer encoder on PBC (0.2 % of weights, 25 min) | PBC 97.7 → 98.1 %, simulated other lab 97.0 → 98.2 %, Munich +1.9 points, myeloblasts sent to review 79 → 89 %; with 5 labelled cells per type 88.6 % vs 84.8 % frozen |

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
