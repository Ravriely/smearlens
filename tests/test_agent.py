from smearlens.agent import Cell, SlideReviewer, check_report, differential, out_of_range


def cells(*labels, confidence=0.9):
    return [Cell(i, "f.png", [0, 0, 1, 1], label, confidence, label) for i, label in enumerate(labels)]


def test_differential_in_percent():
    diff = differential(cells("neutrophil", "neutrophil", "lymphocyte", "monocyte"))
    assert diff["neutrophil"] == 50.0 and diff["lymphocyte"] == 25.0 and diff["basophil"] == 0.0


def test_out_of_range_flags_erythroblasts_and_high_neutrophils():
    diff = differential(cells(*["neutrophil"] * 9, "erythroblast"))
    assert set(out_of_range(diff)) == {"neutrophil", "lymphocyte", "monocyte", "eosinophil", "erythroblast"}


def test_check_report_catches_diagnostic_words():
    assert check_report("Pattern suggests acute leukemia, start treatment") == ["leukemia", "treatment"]
    assert check_report("2 immature granulocytes to review; differential indicative") == []


def test_submit_report_rejects_diagnosis_then_accepts():
    reviewer = SlideReviewer(cells("neutrophil"), fields=1, second_opinion=lambda c: {})
    assert reviewer.tool("submit_report", {"report": "Probable infection."})["accepted"] is False
    assert reviewer.tool("submit_report", {"report": "1 white cell counted."})["accepted"] is True
    assert reviewer.review.rejected_reports == 1


def test_flag_cell_and_summary():
    reviewer = SlideReviewer(cells("ig", "neutrophil", confidence=0.5), fields=2, second_opinion=lambda c: {})
    reviewer.tool("flag_cell", {"cell_id": 0, "priority": "high", "reason": "immature granulocyte"})
    summary = reviewer.tool("get_slide_summary", {})
    assert summary["white_cells"] == 2 and summary["low_confidence_cells"] == 2
    assert summary["enough_for_differential"] is False
    assert reviewer.review.flags[0]["type"] == "ig"
