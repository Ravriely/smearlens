# 07 · Review agent: Muse Spark prepares the slide review

Date: 28/10/2026. Code: `smearlens/agent.py`, `scripts/review_slide.py`.
Reports and full traces: [`07-assets/`](07-assets/).

## Question

Can Muse Spark, as a tool-using agent, turn the pipeline output into what a
lab biologist needs before validating a slide (counts, differential, what is
unusual, which cells to look at first), without ever giving a diagnosis?

## Setup

The agent gets one slide (a set of fields) already analysed by the pipeline of
[06](06-pipeline-with-glimmer-probe.md) (SAM 3 → Muse Spark filter → Glimmer
probe). It talks to `meta-models/muse-spark-open-1026` on Baseten with native
tool calling and has 6 tools:

| Tool | What it returns or does |
|---|---|
| `get_slide_summary` | Fields, white cells by type, differential in %, types outside the reference ranges, low-confidence cells, classifier/detector disagreements |
| `get_reference_ranges` | Indicative adult ranges per 100 white cells ("the lab's own ranges prevail") |
| `list_cells` | Cells filtered by type or by maximum classifier confidence |
| `ask_second_opinion` | Muse Spark looks at the crop of one cell again and returns a label and its reasoning |
| `flag_cell` | Puts a cell on the review list, priority high or normal, with a reason |
| `submit_report` | Submits the report. **Guardrail in code:** a report containing diagnostic or treatment wording (diagnosis, leukemia, infection, treatment, …) is refused and the agent must rewrite it |

The system prompt sets the role (prepare the review, never interpret), asks
that every number come from a tool, and asks to say when fewer than 100 white
cells were counted. At most 16 steps.

Two slides from the TXL-PBC test split:

- **all**: the 64 fields where the pipeline kept at least one white cell
  (74 white cells, mixed sources);
- **BCCD**: only the 16 BCCD fields (24 white cells), a different lab, stain
  and camera from the PBC training data of the probe.

## Results

| Slide | Tool calls | Second opinions | Flags | Rejected reports | Time | Tokens (agent + second opinions) |
|---|---|---|---|---|---|---|
| all (64 fields, 74 cells) | 15 | 6 | 5 | 0 | 69 s | 15,389 + 13,990 |
| BCCD (16 fields, 24 cells) | 13 | 4 | 4 | 0 | 42 s | 18,389 + 7,155 |

On both slides the agent followed the same sensible path without being told
the order: summary → reference ranges → list the uncertain cells → second
opinion on each → flag → report. Both reports state that fewer than 100 cells
were counted and that the differential is only indicative. No diagnostic
wording; the guardrail did not have to fire.

Excerpt of the "all" report ([full report](07-assets/report-all-64.md)):

> **Sample adequacy:** 64 fields analysed, 74 white cells counted. Fewer than
> 100 white cells were counted: the differential is only indicative and
> requires human validation. […]
> - Cell 22 (monocyte 0.62 vs erythroblast) – second opinion: erythroblast pattern

Excerpt of the BCCD report ([full report](07-assets/report-bccd-16.md)):

> **Quality flags:** 4 low-confidence cells; 21/24 cells with
> classifier/detector disagreement – high risk of misclassification. Manual
> review required.

## What it tells us

- **The agent does the reviewer's triage.** It reads the counts, compares them
  to the ranges, digs into the uncertain cells with a second look, and writes
  a short, actionable report. That is the "agentic" part of SmearLens.
- **It caught a real problem.** On the BCCD slide the probe calls 17 of 24
  white cells monocytes, while Muse Spark sees mostly neutrophils (BCCD is
  mostly neutrophils). The agent did not validate this: it reported 21/24
  disagreements and asked for a manual review. This is the expected
  behaviour of a lab tool, and it exposes a **domain shift**: the probe was
  trained only on PBC (one lab, one stain, one camera).
- The no-diagnosis rule holds by design (prompt) and by construction (the
  guardrail refuses the report).

For the pitch: the agent never says what the patient has. It tells the
biologist what to look at, and it raises its hand when the machine
disagrees with itself.

## Notes and limits

- The test slides are made of dataset fields, not one patient's slide. PBC
  fields were sampled per class, so the differential itself (44.6 % monocytes)
  is not realistic. What matters here is the review behaviour.
- The forbidden-word list is a simple safety net, not a full clinical-language
  filter.
- Two runs only; the agent's path varies between runs (the second "all" run
  flagged different cells).

## Next

1. Fix the domain shift: train the probe with colour and stain augmentation,
   and check BCCD again.
2. Use the agent output in the demo (report + review list with cell crops).

## Reproduce

```bash
export BASETEN_API_KEY=...
uv run python scripts/review_slide.py                  # all fields
uv run python scripts/review_slide.py --source BCCD    # one source
```

Needs `results/pipeline/cells_glimmer_rbc.json` from `scripts/pipeline_eval.py --typer glimmer`.
