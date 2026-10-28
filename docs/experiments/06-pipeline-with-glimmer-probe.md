# 06 · Pipeline with the Glimmer probe for the cell type

Date: 28/10/2026. Script: `scripts/pipeline_eval.py --typer glimmer`, crop
logic in `smearlens/pipeline.py` (`crop_at_scale`), Glimmer features from
`scripts/modal_glimmer_probe.py` (`embed_crops`). Raw summaries:
[`tight`](06-summary_glimmer_tight.json), [`rbc`](06-summary_glimmer_rbc.json),
[`rbc + platelet recheck`](06-summary_glimmer_rbc_plt.json).

## Question

The probe reaches 97.6 % on PBC crops ([05](05-glimmer-linear-probe.md)). Does
it improve the cell type in the full pipeline ([04](04-end-to-end-pipeline.md)),
on cells found by SAM 3 in whole smear images?

## Setup

Same 82 TXL-PBC test images, same SAM 3 candidates and same Muse Spark labels
as 04 (cached, no new Baseten calls). Muse Spark still decides "white cell or
not". For every candidate that Muse Spark keeps as a nucleated cell, the probe
replaces Muse Spark's type. Three ways to cut the image given to the probe:

| Variant | Crop |
|---|---|
| tight | Square, 1.8× the SAM box (the crop used for Muse Spark) |
| **rbc** | **Same physical scale as PBC: side = 360 px × (median red cell size in this image / 74.6 px), centred on the candidate, padded with the background colour** |
| rbc + platelet recheck | As rbc, and the candidates that Muse Spark called `platelet` also go through the probe |

The median red cell size comes from SAM 3's own "red blood cell" detections
(score ≥ 0.5): 74.6 px on PBC, 94 px on BCCD, 52 px on BCDD. Red cells are
the ruler: they have nearly the same diameter in every smear.

## Results

| Variant | Right type, end to end (55 PBC cells) | Right type, kept cells | WBC precision | WBC recall | WBC F1 | Count error / image |
|---|---|---|---|---|---|---|
| Muse Spark only (04) | 58.2 % | 70 % (32/46) | 0.78 | 0.68 | 0.72 | 0.55 |
| tight | 60.0 % | 72 % (33/46) | 0.80 | 0.67 | 0.73 | 0.54 |
| **rbc** | **72.7 %** | **87 % (40/46)** | **0.80** | **0.68** | **0.73** | **0.52** |
| rbc + platelet recheck | 81.8 % | 85 % (45/53) | 0.60 | 0.74 | 0.66 | 0.79 |

Per type with **rbc** (end to end): neutrophil 10/10, monocyte 12/13,
eosinophil 11/11, basophil 5/12 (7 never reach the probe, see below),
lymphocyte 2/9 (4 called monocyte, 2 never detected by SAM).

Count error per image with **rbc**: 0.34 on PBC, 0.67 on BCCD, 1.00 on BCDD.

## What it tells us

- **Scale matters more than the model.** With a tight crop the probe barely
  helps (58 → 60 %): the cell fills the crop, looks twice as big as in
  training, and lymphocytes turn into monocytes. Cutting at the PBC scale,
  using red cells as a ruler, gives **72.7 % end to end and 87 % on kept
  cells**, with the same detection quality.
- **The remaining errors are upstream.** 7 of the 12 basophils are called
  `platelet` by Muse Spark and never reach the probe. Sending those
  "platelets" to the probe recovers 5 basophils (81.8 % right type) but turns
  real platelets into white cells: precision falls from 0.80 to 0.60. The
  probe was trained on white cells and platelets centred in a PBC field, it is
  not a reliable "is this a white cell" filter. This variant is off by default.
- Lymphocytes stay hard (2/9). They are the smallest white cells, so any error
  in the scale estimate hits them first.

For the pitch: in the full pipeline, the Meta open-weights probe lifts the
right cell type from 58 % to 73 %, 87 % on the cells kept, with one
domain-knowledge trick: red cells give the scale.

## Notes and limits

- A first version clamped the crop inside the image. On PBC images (360 px)
  every crop became the full image, so the probe always saw the main white
  cell. That inflated the platelet-recheck variant. The version above centres
  the crop and pads with the background colour; all numbers are from it.
- 55 typed cells: about ±12 points on the end-to-end accuracy.
- Cell types are only known for PBC images. On BCCD and BCDD only the count is
  scored.
- The 74.6 px reference is the median SAM red cell size on the PBC test
  images, not a physical calibration.

## Next

1. Basophils: a dedicated "white cell or not" decision that does not confuse
   dark granules with platelets (prompt change, or a small binary probe on
   Glimmer features with SAM false positives as negatives).
2. Lymphocytes: check the scale estimate on BCCD/BCDD by eye.
3. Move on to the review agent and the demo.

## Reproduce

```bash
export BASETEN_API_KEY=...
uv run python scripts/pipeline_eval.py --images 200                       # 04, Muse only
uv run python scripts/pipeline_eval.py --images 200 --typer glimmer       # rbc (default)
uv run python scripts/pipeline_eval.py --images 200 --typer glimmer --glimmer-crop tight
uv run python scripts/pipeline_eval.py --images 200 --typer glimmer --recheck-platelets
```

Run from the repo root with `PYTHONPATH=.:scripts`. Glimmer features are
cached in `results/pipeline/glimmer_features_<crop>.npz`.
