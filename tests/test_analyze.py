import json

import numpy as np

from smearlens.analyze import candidates_for, count_outside, load_probes, merge_kept, rbc_sizes, report, white_cells
from smearlens.datasets import Box
from smearlens.pipeline import Candidate


def cand(x0, y0, x1, y1, score=0.5):
    return Candidate(Box("WBC", x0, y0, x1, y1), score)


def detections():
    return [
        {"image": "f", "label": "WBC", "boxes": [[0, 0, 40, 40], [10, 10, 30, 30]], "scores": [0.9, 0.2]},
        {"image": "f", "label": "RBC", "boxes": [[100, 0, 130, 30], [200, 0, 240, 40]], "scores": [0.6, 0.4]},
        {"image": "f", "label": "Platelet", "boxes": [[15, 15, 20, 20], [300, 300, 305, 305]], "scores": [0.5, 0.5]},
    ]


def probes(dim=2):
    return {
        "detection": {"mean": np.zeros(dim), "std": np.ones(dim), "weight": np.array([10.0, 0.0]), "bias": 0.0,
                      "threshold": np.array(0.9)},
        "type": {"mean": np.zeros(dim), "std": np.ones(dim), "weight": np.array([[0.0, 5.0], [0.0, -5.0]]),
                 "bias": np.zeros(2), "classes": np.array(["neutrophil", "lymphocyte"])},
        "temperature": 1.0,
        "review_cutoff": 0.99,
    }


def test_candidates_use_score_floors_and_skip_platelets():
    found = candidates_for(detections(), wbc_min=0.3, rbc_min=0.5)["f"]
    assert [c.box.x0 for c in found] == [0, 100]
    assert rbc_sizes(detections())["f"] == 30


def test_merge_keeps_best_of_close_candidates():
    cands = [cand(0, 0, 40, 40), cand(10, 0, 50, 40), cand(200, 200, 240, 240), cand(400, 400, 440, 440)]
    kept = merge_kept(cands, np.array([0.95, 0.99, 0.97, 0.5]), threshold=0.9, rbc_px=30)
    assert [b.x0 for b in kept] == [10, 200]


def test_white_cells_apply_both_probes():
    cands = [cand(0, 0, 40, 40), cand(200, 0, 240, 40)]
    cells = white_cells(cands, np.array([[1.0, 1.0], [-1.0, 0.0]]), rbc_px=30, probes=probes())
    assert len(cells) == 1
    assert cells[0]["type"] == "neutrophil" and cells[0]["needs_review"] is False
    assert cells[0]["p_wbc"] > 0.99


def test_report_counts_cells_outside_white_cells():
    cells = [{"box": [0, 0, 40, 40], "p_wbc": 1.0, "type": "neutrophil", "confidence": 0.9, "needs_review": True}]
    out = report((320, 240), detections(), 30.0, cells, {"total": 1.23456})
    assert out["red_cell_count"] == 1
    assert out["platelet_count"] == 1
    assert out["differential_percent"] == {"neutrophil": 100.0}
    assert out["needs_review_count"] == 1
    assert out["timing_s"] == {"total": 1.235}
    assert count_outside(detections(), "RBC", 0.3, []) == 2


def test_load_probes(tmp_path):
    np.savez(tmp_path / "d.npz", weight=np.ones(2), bias=0.0, mean=np.zeros(2), std=np.ones(2), threshold=0.5)
    np.savez(tmp_path / "t.npz", weight=np.ones((2, 2)), bias=np.zeros(2), mean=np.zeros(2), std=np.ones(2),
             classes=np.array(["a", "b"]))
    (tmp_path / "c.json").write_text(json.dumps({"temperature": 0.8}))
    (tmp_path / "l.json").write_text(json.dumps({"review_cutoff_from_pbc_10pct": 0.96}))
    loaded = load_probes(tmp_path / "d.npz", tmp_path / "t.npz", tmp_path / "c.json", tmp_path / "l.json")
    assert loaded["temperature"] == 0.8 and loaded["review_cutoff"] == 0.96
    assert float(loaded["detection"]["threshold"]) == 0.5
