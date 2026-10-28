# 12 · Fine-tuning the Glimmer encoder (LoRA)

Date: 28/10/2026. Script: `scripts/modal_glimmer_lora.py` (Modal, L40S, 25 min).
Numbers: [`12-lora-metrics.json`](12-lora-metrics.json).

## Question

The linear probe sits on a frozen encoder. [11](11-external-lab-munich.md)
showed that the gap on another lab is not about scale. Can a light
fine-tune of the encoder itself make the features hold better on a lab it
has never seen?

## Setup

- LoRA (rank 16, alpha 32, dropout 0.05) on the attention projections
  (q, k, v, out) of the last 16 of the 50 vision blocks. This trains 3.1 M parameters, 0.2 % of
  the encoder's weights. Earlier blocks stay frozen and need no gradient.
- Head: linear, warm-started from the augmented probe of
  [08](08-stain-augmented-probe.md) (its standardisation folded into the
  weights).
- Training: PBC train only, 2 epochs, batch 16, AdamW (LoRA lr 1e-4, head
  3e-4), 50-step warm-up then cosine, class-balanced loss. A fresh
  stain/colour/scale augmentation ([08](08-stain-augmented-probe.md)) on
  every image at every step.
- Before/after on the same cells: the same run evaluates the frozen encoder
  with the starting head (adapter disabled) and the LoRA encoder with the
  trained head.
  - PBC test: 2,560 cells.
  - PBC test with a strong simulated stain shift (strength 1.5).
  - Munich: up to 300 cells per Munich class, 1,611 cells of known types plus
    300 myeloblasts. **Never used for training or model choice.**

## Results

| Test set | Frozen + probe | LoRA | Change |
|---|---|---|---|
| PBC test, accuracy | 97.7 % | **98.1 %** | +0.4 |
| PBC simulated other lab, accuracy | 97.0 % | **98.2 %** | +1.2 |
| Munich, balanced accuracy | 72.1 % | **74.0 %** | +1.9 |
| Munich, accuracy | 73.1 % | 75.3 % | +2.2 |
| Munich known cells sent to review (PBC 10 % cutoff) | 41 % | 50 % | |
| Myeloblasts sent to review | 79 % | **89 %** | +10 |

Munich recall per type:

| Type | Frozen | LoRA |
|---|---|---|
| Neutrophil | 89.2 % | 91.7 % |
| Erythroblast | 94.9 % | 97.4 % |
| Immature granulocyte | 83.4 % | 87.6 % |
| Monocyte | 85.7 % | **74.3 %** |
| Lymphocyte | 57.0 % | **66.0 %** |
| Eosinophil | 51.7 % | 60.7 % |
| Basophil | 43.0 % | 40.5 % |

The Munich numbers here use a class-balanced subsample and no temperature,
so they differ from the full-set numbers of [11](11-external-lab-munich.md).
Compare the two columns with each other, not with fiche 11.

## LoRA + a few labelled Munich cells

Same protocol as part 4 of [11](11-external-lab-munich.md) (head retrained on
PBC train + N Munich cells per type, 3 draws, full Munich set of known types),
on features from the LoRA encoder. Numbers:
[`12-lora-adaptation.json`](12-lora-adaptation.json).

| Labelled per type | Frozen: accuracy | LoRA: accuracy | Frozen: balanced | LoRA: balanced |
|---|---|---|---|---|
| 0 | 83.7 % | **84.7 %** | 72.3 % | **75.0 %** |
| 5 | 84.8 % | **88.6 %** | 77.5 % | **82.5 %** |
| 10 | 89.4 % | **90.0 %** | 84.7 % | **87.1 %** |
| 25 | 89.7 % | 89.2 % | 87.3 % | **88.1 %** |
| 50 | 91.8 % | 91.6 % | 88.9 % | **89.6 %** |
| 100 | 92.6 % | **93.1 %** | 88.5 % | 88.4 % |

The two levers add up where labels are scarce: with 5 cells per type the
LoRA encoder gives +4 points of accuracy and +5 points of balanced accuracy.
With 25 cells or more per type the two encoders are level, because the head
then learns the new lab on its own.

For the pitch: "With the fine-tuned encoder, 35 labelled cells (5 per type)
take a new hospital from 84.7 % to 88.6 %."

## What it tells us

- **The fine-tune helps a little everywhere it can see**: PBC and
  simulated stain shifts go up to 98 %.
- **On a real other lab the gain is small (+2 points)** and uneven:
  lymphocytes and eosinophils improve, monocytes get worse. The encoder
  learns what PBC looks like. Augmentation alone does not teach it what
  Munich looks like.
- **It is safer on blasts**: 89 % of myeloblasts go to review instead of
  79 %. Part of this comes from the LoRA model being very sure on PBC (its PBC
  10 % cutoff is 0.9985 against 0.937), so the fixed rule sends more cells
  to review on any new lab. This is the right direction for a triage tool.
- The big lever for a new lab is still a few labelled cells (+6 points with
  70 cells, [11](11-external-lab-munich.md)), not a fine-tune on the source
  lab.

For the pitch: "We fine-tuned Meta's Glimmer encoder (LoRA, 0.2 % of the
weights, 25 minutes on one GPU). It is more robust to stain changes (98 %)
and sends 89 % of leukaemia blasts to the biologist."

## Notes and limits

- One run, one seed, no hyper-parameter search (rank, blocks, epochs). The
  training loss reaches ~0 early, so fewer steps or stronger regularisation
  may transfer better.
- The adapter (`lora/adapter`) and head (`lora/head.pt`) are saved in the
  `smearlens-hf-cache` Modal volume. They will be used for the demo and published as
  open weights.
- The processor's LANCZOS → BICUBIC fallback applies to both columns.

## Reproduce

```bash
modal run scripts/modal_glimmer_lora.py --max-steps 20 --eval-per-class 20   # smoke test, ~5 min
modal run --detach scripts/modal_glimmer_lora.py
modal run --detach scripts/modal_glimmer_lora.py::features   # LoRA features of all PBC and Munich cells
modal run scripts/modal_lmu_adapt.py --encoder lora
```
