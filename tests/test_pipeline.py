from PIL import Image

from smearlens.datasets import Box
from smearlens.pipeline import Candidate, crop, dedupe, differential, pbc_label, probe_labels, review_flags


def cand(x0, y0, x1, y1, score=0.5, label=""):
    return Candidate(Box("WBC", x0, y0, x1, y1), score, label)


def test_pbc_label_from_original_name():
    assert pbc_label("BA_12345.jpg") == "basophil"
    assert pbc_label("BNE_998.jpg") == "neutrophil"
    assert pbc_label("BloodImage_00095.jpg") is None


def test_crop_is_square_and_enlarged():
    image = Image.new("RGB", (400, 400))
    data = crop(image, Box("WBC", 100, 100, 140, 120), scale=2.0, min_side=10)
    assert Image.open(__import__("io").BytesIO(data)).size == (80, 80)


def test_dedupe_keeps_best_of_nested_boxes():
    kept = dedupe([cand(0, 0, 40, 40, 0.4), cand(10, 10, 30, 30, 0.9), cand(100, 100, 140, 140, 0.3)])
    assert [c.score for c in kept] == [0.9, 0.3]


def test_differential_ignores_non_wbc():
    cells = [cand(0, 0, 1, 1, label=l) for l in ("neutrophil", "neutrophil", "lymphocyte", "not_wbc", "platelet")]
    formula = differential(cells)
    assert formula["neutrophil"] == 2 / 3
    assert formula["lymphocyte"] == 1 / 3
    assert sum(formula.values()) == 1


def test_review_flags_priority_cells_first():
    cells = [cand(0, 0, 1, 1, label=l) for l in ("ig", "neutrophil", "unparsed")]
    assert review_flags(cells) == ["1 ig", "1 unclear"]


def test_probe_labels_picks_highest_logit():
    import numpy as np

    probe = {
        "mean": np.zeros(2),
        "std": np.ones(2),
        "weight": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "bias": np.zeros(2),
        "classes": np.array(["neutrophil", "lymphocyte"]),
    }
    assert probe_labels([[2.0, 1.0], [0.0, 3.0]], probe) == ["neutrophil", "lymphocyte"]


def test_crop_at_scale_keeps_the_cell_centred():
    import io

    from smearlens.pipeline import PBC_RBC_PX, crop_at_scale

    image = Image.new("RGB", (640, 480), (200, 200, 200))
    image.paste((255, 0, 0), (0, 0, 20, 20))
    field = Image.open(io.BytesIO(crop_at_scale(image, Box("WBC", 0, 0, 20, 20), rbc_px=PBC_RBC_PX)))
    assert field.size == (360, 360)
    r, g, b = field.getpixel((180, 180))
    assert r > 200 and g < 60
    assert abs(field.getpixel((10, 10))[0] - 200) < 10
    assert Image.open(io.BytesIO(crop_at_scale(image, Box("WBC", 0, 0, 20, 20), rbc_px=PBC_RBC_PX * 2))).size == (720, 720)
