# Data

SmearLens only uses public datasets whose license allows reuse. No private or
hospital data is used, at any step.

| Dataset | Content | License | Use in SmearLens |
|---|---|---|---|
| [PBC](https://data.mendeley.com/datasets/snkd93bnjr) (Acevedo et al., 2020) | 17,092 single-cell images, 8 classes | CC BY 4.0 | Cell classifier training |
| [TXL-PBC](https://github.com/lugan113/TXL-PBC_Dataset) (Gan et al., 2025) | 1,260 images, 18,143 YOLO boxes (WBC, RBC, platelets) | MIT | Detection and count evaluation, SAM prompts |
| [AML-Cytomorphology_LMU](https://doi.org/10.7937/tcia.2019.36f5o9ld) (Matek et al., 2019) | 18,365 single-cell images from Munich, 15 classes | CC BY 3.0 | External lab test, suspected blast class, one field of the example slide |

`scripts/fetch_data.py` downloads PBC and TXL-PBC into `data/`, `scripts/modal_fetch_lmu.py` the Munich set
into a Modal volume.

## TXL-PBC sources

TXL-PBC merges four datasets. We keep the images whose original source has a
clear license, and drop the others:

| Source | Images | License | Kept |
|---|---|---|---|
| PBC | 500 | CC BY 4.0 | yes |
| BCCD | 160 | MIT | yes |
| BCDD | 100 | MIT (Hugging Face dataset card) | yes |
| Raabin-WBC | 500 | no official license found | **no** |

The filter is `ALLOWED_SOURCES` in `smearlens/datasets.py`.

## Citations

- Acevedo A., Merino A., Alférez S., Molina Á., Boldú L., Rodellar J.
  *A dataset of microscopic peripheral blood cell images for development of
  automatic recognition systems.* Data in Brief 30, 2020.
  doi:10.1016/j.dib.2020.105474
- Gan L. et al. *TXL-PBC: a curated and re-annotated peripheral blood cell
  dataset integrating multiple sources.* Scientific Data, 2025.
- BCCD Dataset, https://github.com/Shenggan/BCCD_Dataset (MIT).
- Blood Cell Detection Dataset,
  https://huggingface.co/datasets/draaslan/blood-cell-detection (MIT).
- Matek C., Schwarz S., Marr C., Spiekermann K. *A single-cell morphological dataset of
  leukocytes from AML patients and non-malignant controls.* The Cancer Imaging Archive, 2019.
  doi:10.7937/tcia.2019.36f5o9ld (CC BY 3.0). Field 06 of the example slide is `MYO_0029`, converted
  from TIFF to PNG, otherwise unchanged.
