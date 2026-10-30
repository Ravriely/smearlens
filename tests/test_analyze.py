import json

import numpy as np

from smearlens.analyze import (
    candidates_for,
    count_cells,
    count_outside,
    load_probes,
    in_core,
    merge_kept,
    plan_windows,
    rbc_sizes,
    relative_size,
    report,
    tiles_per_side,
    white_cells,
)
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


def strict_probes():
    found = probes()
    found["detection"]["threshold"] = np.array(0.999)
    return found


def test_white_cells_rescue_the_best_candidate_of_an_empty_field_for_review():
    cells = white_cells([cand(0, 0, 40, 40)], np.array([[0.3, 1.0]]), rbc_px=30, probes=strict_probes())
    assert len(cells) == 1
    assert cells[0]["uncertain_detection"] is True and cells[0]["needs_review"] is True


def test_white_cells_rescue_nothing_below_the_floor_or_next_to_a_sure_cell():
    assert white_cells([cand(0, 0, 40, 40)], np.array([[0.2, 1.0]]), rbc_px=30, probes=strict_probes()) == []
    cands = [cand(0, 0, 40, 40), cand(200, 0, 240, 40)]
    cells = white_cells(cands, np.array([[1.0, 1.0], [0.3, 1.0]]), rbc_px=30, probes=strict_probes())
    assert [c["uncertain_detection"] for c in cells] == [False]


def test_report_counts_cells_outside_white_cells():
    cells = [{"box": [0, 0, 40, 40], "p_wbc": 1.0, "type": "neutrophil", "confidence": 0.9, "needs_review": True}]
    dets = detections() + [{"image": "f", "label": "Platelet", "boxes": [[300, 0, 308, 8]], "scores": [0.5]}]
    out = report((320, 240), dets, 30.0, cells, {"total": 1.23456})
    assert out["red_cell_count"] == 2
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


def test_relative_size_uses_longest_side():
    assert relative_size(Box("RBC", 0, 0, 30, 15), 20.0) == 1.5


def test_count_cells_rejects_boxes_outside_size_range():
    dets = [{"image": "f", "label": "Platelet", "scores": [0.9, 0.9, 0.9, 0.9],
             "boxes": [[0, 0, 2, 2], [10, 0, 18, 8], [30, 0, 48, 18], [60, 0, 90, 30]]}]
    assert count_cells(dets, "Platelet", 0.3, [], 30.0, (0.22, 0.6)) == 2
    assert count_outside(dets, "Platelet", 0.3, []) == 4


def test_count_cells_drops_red_cells_on_white_cells_and_nested_boxes():
    white = [Box("WBC", 0, 0, 40, 40)]
    dets = [{"image": "f", "label": "RBC", "scores": [0.9, 0.8, 0.7, 0.6],
             "boxes": [[5, 5, 35, 35], [100, 0, 130, 30], [105, 5, 133, 33], [200, 0, 230, 30]]}]
    assert count_cells(dets, "RBC", 0.3, white, 30.0, (0.85, 1.6)) == 2


def test_count_cells_drops_red_cell_box_that_covers_a_white_cell():
    white = [Box("WBC", 20, 20, 30, 30)]
    dets = [{"image": "f", "label": "RBC", "scores": [0.9], "boxes": [[0, 0, 30, 30]]}]
    assert count_cells(dets, "RBC", 0.3, white, 30.0, (0.85, 1.6)) == 0
    assert count_outside(dets, "RBC", 0.3, white) == 1


def test_count_cells_uses_score_floor():
    dets = [{"image": "f", "label": "RBC", "scores": [0.9, 0.2], "boxes": [[0, 0, 30, 30], [100, 0, 130, 30]]}]
    assert count_cells(dets, "RBC", 0.3, [], 30.0, (0.85, 1.6)) == 1


def test_tiles_per_side_keeps_single_fields_whole():
    assert tiles_per_side((360, 363), 74.6) == 1
    assert tiles_per_side((640, 480), 94.3) == 1
    assert tiles_per_side((360, 363), None) == 1


def test_tiles_per_side_splits_wide_fields():
    assert tiles_per_side((720, 726), None) == 2
    assert tiles_per_side((1280, 960), 74.6) == 4
    assert tiles_per_side((1280, 960), 120) == 2
    assert tiles_per_side((4000, 3000), 30) == 4


def test_windows_overlap_and_cores_cover_each_point_once():
    size = (100, 80)
    plans = plan_windows(size, 2, margin=0.2)
    assert len(plans) == 4
    window, core = plans[0]
    assert (window.x0, window.y0, window.x1, window.y1) == (0.0, 0.0, 60.0, 48.0)
    for x, y in [(0, 0), (49.9, 39.9), (50, 40), (99, 79), (100, 80)]:
        assert sum(in_core([x, y, x, y], core, size) for _, core in plans) == 1
