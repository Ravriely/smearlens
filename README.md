# SmearLens

A review assistant for peripheral blood smears. SmearLens helps lab biologists
go through slides faster: it finds every cell, proposes a differential count,
and flags the slides and cells that need a closer human look first.

> **SmearLens is a lab workflow tool, not a diagnostic device.** It never gives
> a diagnosis or medical advice. Every count it proposes is a draft that a
> biologist reviews and validates.

Built for the Meta AI Developer Hackathon 2026 (Indie track).

## How it works

1. **Find the cells.** SAM 3 proposes candidates on the field image with three text prompts ("cell with a large
   purple nucleus", "neutrophil", "red blood cell") and a fourth for platelets ("small purple dot"). On a wide field, red cells and platelets are
   counted again on overlapping tiles ([17](docs/experiments/17-wide-field-tiling.md)).
2. **Keep the white cells and type them.** Each candidate is cropped at the scale of the red cells
   around it and embedded by the frozen vision encoder of **Muse Glimmer**. Two linear probes on these
   features say whether it is a white cell ([09](docs/experiments/09-detection-probe.md)) and which
   type it is ([05](docs/experiments/05-glimmer-linear-probe.md)). No LLM call per cell.
3. **Send the doubtful cells to review.** The type probe is calibrated: cells below a 96 % confidence
   cutoff go to a review queue ([10](docs/experiments/10-review-coverage.md)), as do cells kept with
   an uncertain detection ([16](docs/experiments/16-empty-field-rescue.md)).
4. **Prepare the review.** A **Muse Spark** agent with six tools reads the counts, asks a second
   opinion on the doubtful cells, flags the ones to open first and writes a short slide report, with
   a guardrail against diagnostic wording ([07](docs/experiments/07-review-agent.md),
   [15](docs/experiments/15-review-agent-on-backend.md)).

The biologist validates or corrects every cell. SmearLens proposes, it does not decide.

## Status

Runs end to end and is deployed as a web demo (see [Web demo](#web-demo)): one field of view in, white
cells with type and confidence, red cell and platelet counts, a differential, a review queue and the
agent report out, in a few seconds plus about 40 s for the agent. Main numbers:

- Type of white cells: 97.6 % on PBC test cells, 96.2 % on a 147-field PBC slide through the whole
  backend, with 4 of 5 errors already in the review queue ([05](docs/experiments/05-glimmer-linear-probe.md),
  [15](docs/experiments/15-review-agent-on-backend.md)).
- Review: checking the 10 % least sure cells leaves 0.3 % errors ([10](docs/experiments/10-review-coverage.md)).
- External lab (Munich, never seen): 83.7 % zero-shot, 71 % of myeloblasts, a type the probe never
  saw, sent to review ([11](docs/experiments/11-external-lab-munich.md)).
- Known weak spots: white cell detection on pale BCDD images and on wide fields, red cells
  undercounted on crowded PBC fields. Each experiment page lists its limits.

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
| [13](docs/experiments/13-sam3-prompts-bcdd-detection.md) | Second SAM 3 prompt ("neutrophil") for pale BCDD white cells | BCDD candidate recall 0.02 → 0.92; BCDD detection F1 0.13 → 0.52; overall WBC F1 0.79 → 0.81, recall 0.68 → 0.75, count error 0.38 → 0.34 (BCCD precision 0.81 → 0.72) |
| [14](docs/experiments/14-red-cell-platelet-size-filter.md) | Size filter on red cell and platelet boxes, scaled on the median red cell of the image | Count error per image: platelets 1.33 → 0.21, red cells 6.51 → 3.41 (BCDD 6.79 → 1.64), white cells unchanged (0.34); PBC red cells still undercounted |
| [15](docs/experiments/15-review-agent-on-backend.md) | Review agent on the demo backend output, one PBC slide of 147 fields (132 white cells) | Backend: 96.2 % right type, 4 of 5 errors in a review queue of 11 %; agent: report with queue size, rare types and disagreements first, no diagnostic wording, about 2 min and 0.15 $ per slide |
| [16](docs/experiments/16-empty-field-rescue.md) | Keep the best candidate of a field with no sure white cell, as a cell to review | WBC F1 0.798 → 0.815, count error 0.40 → 0.27 per test field, empty fields 12 → 1; lowering the threshold everywhere loses |
| [17](docs/experiments/17-wide-field-tiling.md) | Wide fields (2 × 2 mosaics of test fields): count red cells and platelets on tiles | Red cells found 431 → 1013 of 1124 (PBC 0 → 529 of 601), error 36.5 → 7.9 per image; white cells unchanged; single fields never tiled |

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

## Demo backend (Modal)

`scripts/modal_app.py` serves the whole analysis (SAM 3, Glimmer, both probes) on one L40S GPU:

```bash
uv run modal run scripts/modal_app.py --image field.png   # one-off, the app stops after the run
```

The deployed app scales to zero: the first request after a pause waits for a cold start (about 40 s),
then a container stays up 15 min after the last request. For filming or the jury, keep one container
warm, and turn it off right after:

```bash
uv run python scripts/modal_warm.py on       # one L40S always up
uv run python scripts/modal_warm.py status
uv run python scripts/modal_warm.py off      # back to zero; the container stops after 15 min idle
```

A warm L40S costs about 2 $/h, idle or not (about 50 $ per day if left on). `off` leaves at most
15 min of idle time (about 0.50 $); `modal container list` then `modal container stop <id>` stops it at
once. A new `modal deploy` resets the override; `SMEARLENS_WARM=1 modal deploy scripts/modal_app.py`
deploys with one container warm from the start.

## Web demo

The deployed app serves the review page and its API from one URL:

**https://ravriely--smearlens-app-web.us-east.modal.run**

The page opens on an offline demo, no GPU and no LLM call: one PBC field of the TXL-PBC test split
(`BNE_59680.jpg`, a neutrophil at 95.8 %, so it sits in the review queue), the backend output on it
(`web/sample.json`) and a recorded run of the review agent on that output (`web/sample-review.json`).
To analyse your own image, click **Access token**, paste the token you were given, then drop a JPG,
PNG or WebP field of view (12 MB and 4096×4096 px at most). The token stays in the browser
(localStorage). Once the counts are shown, the page asks the review agent for its report on the same
image: the panel shows "Muse Spark, live", then the report and the cells to open first, after about
30 to 60 s (more after a pause, while the agent container starts).

**Slide mode.** Drop 2 to 20 fields of the same slide at once (or click **Slide · 12 fields**, then
**Run it live**). Each field goes to `/api/analyze` on its own, one after the other: SAM 3 does not
handle stitched mosaics, so the fields are never assembled. The page then adds them up into one slide
report: the differential over every white cell of every field, the red cell and platelet counts per
field on average, and one review list across the fields (a thumbnail strip, or the arrow keys, opens
each field). The agent reviews the whole slide through `POST /api/review-slide`, with at most 14 steps
and 6 second opinions. The ready-to-click example is 12 PBC fields of the TXL-PBC test split
(`web/slide/`, 230 kB, picked and recorded by `scripts/slide_demo.py`): 12 white cells typed right out
of 12, analysed in about 65 s from cold (1 s per field once the GPU is up), reviewed in about 40 s for
20k Baseten tokens.

Cell numbers are the same everywhere: the badges on the image, the review list and the agent report
count white cells from 1, in the order of the backend output (field after field on a slide),
platelets skipped.

The review agent gets at most 10 steps and 3 second opinions per image. On one field it costs about
15k to 22k Baseten tokens. Below 20 white cells it does not compare the differential with the
reference ranges: a percentage over a handful of cells is not worth it.

The API is `POST /api/analyze` with the image as the raw body and the token in the
`x-smearlens-token` header; it returns the JSON of `smearlens/analyze.py:report` plus
`review_cutoff`, `round_trip_s` and `candidate_boxes` (every SAM candidate with its detection
probability, to see why a cell was kept or dropped):

```bash
curl -X POST --data-binary @field.jpg -H "content-type: image/jpeg" \
  -H "x-smearlens-token: $SMEARLENS_WEB_TOKEN" \
  https://ravriely--smearlens-app-web.us-east.modal.run/api/analyze
```

`POST /api/review` takes `{"analysis": <output of /api/analyze>, "image": <the same image in
base64>, "name": "..."}` with the same token header, and returns the output of
`smearlens/agent.py:review_slide` (report, flags, second opinions, tool calls, tokens). It runs in its
own CPU function, `review`, the only one that holds the Baseten key (Modal secret `baseten`, key
`BASETEN_API_KEY`). To regenerate the recorded sample run after a change to the agent:

```bash
cd web && uv run python -c 'import json; from pathlib import Path; from smearlens.agent import review_slide; \
a = json.loads(Path("sample.json").read_text())[0]; \
r = review_slide([a], [Path("sample.jpg").read_bytes()], names=["sample.jpg"], max_steps=10, max_second_opinions=3); \
Path("sample-review.json").write_text(json.dumps(r, indent=2) + "\n")'
```

`POST /api/review-slide` takes `{"fields": [{"name": "...", "analysis": ..., "image": ...}, ...]}`,
1 to 20 fields and 64 MB at most, and returns the same review for the whole slide. Its agent flags
carry the field name. To regenerate the example slide: `SMEARLENS_WEB_TOKEN=... uv run python scripts/slide_demo.py record`.

Cost guards, since the URL is public:

- A small CPU container checks the token, the size and the pixel count of the image before it
  calls the GPU class. A refused request (401, 413, 415) never starts a GPU.
- The token lives in the Modal secret `smearlens-web` (key `SMEARLENS_WEB_TOKEN`), never in the
  repo. To change it: `modal secret create --force smearlens-web SMEARLENS_WEB_TOKEN=...`, then
  `modal deploy scripts/modal_app.py`.
- One GPU container at most (`max_containers=1`), none at rest (`min_containers=0`).
- The agent only runs behind the same token, on a body of at most 16 MB (85 MB for a slide of up to 20 fields), and not at all when the
  analysis found no white cell. Its step and second opinion budgets cap the cost of one call.

At rest the demo costs nothing. A request after a pause waits for a cold start (about 45 s,
shown in the page as "Waking up the GPU"), then the L40S stays up 15 min after the last
request, about 2 $/h while it is up (about 0.50 $ per idle window). For filming or the jury,
`scripts/modal_warm.py on` keeps it warm and `off` lets it scale to zero again (see above).

To run the page from a local copy, serve `web/` (`python3 -m http.server -d web`); it calls the
deployed URL, or the one given with `?api=https://...`. Add `?lang=fr` for French.

## Data

SmearLens only uses public datasets with a clear license. See
[DATA.md](DATA.md) for sources, licenses and citations.

## License

MIT. Dataset licenses are listed in [DATA.md](DATA.md).
