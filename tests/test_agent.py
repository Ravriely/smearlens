import json

from PIL import Image

from smearlens.agent import (
    Cell,
    SlideReviewer,
    cells_from_reports,
    check_report,
    differential,
    out_of_range,
    review_slide,
)


def cells(*labels, confidence=0.9, needs_review=False):
    return [Cell(i, "f.png", [0, 0, 1, 1], label, confidence, needs_review) for i, label in enumerate(labels)]


def backend_report(*cells):
    return {
        "white_cells": [
            {"box": [10, 10, 50, 50], "p_wbc": 0.99, "type": t, "confidence": c, "needs_review": c < 0.8}
            for t, c in cells
        ],
        "red_cell_count": 12,
        "platelet_count": 2,
    }


class FakeChat:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def __call__(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        calls = self.script.pop(0)
        message = {"role": "assistant", "content": ""}
        if calls:
            message["tool_calls"] = [
                {"id": f"call-{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                for i, (n, a) in enumerate(calls)
            ]
        return {"choices": [{"message": message}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def test_differential_in_percent():
    diff = differential(cells("neutrophil", "neutrophil", "lymphocyte", "monocyte"))
    assert diff["neutrophil"] == 50.0 and diff["lymphocyte"] == 25.0 and diff["basophil"] == 0.0


def test_out_of_range_flags_erythroblasts_and_high_neutrophils():
    diff = differential(cells(*["neutrophil"] * 9, "erythroblast"))
    assert set(out_of_range(diff)) == {"neutrophil", "lymphocyte", "monocyte", "eosinophil", "erythroblast"}


def test_check_report_catches_diagnostic_words():
    assert check_report("Pattern suggests acute leukemia, start treatment") == ["leukemia", "treatment"]
    assert check_report("2 immature granulocytes to review; differential indicative") == []


def test_cells_from_reports_numbers_from_one_and_keeps_field():
    reports = [backend_report(("neutrophil", 0.95)), backend_report(), backend_report(("ig", 0.5), ("monocyte", 0.9))]
    found = cells_from_reports(reports, ["a", "b", "c"])
    assert [c.uncertain_detection for c in found] == [False, False, False]
    assert [(c.id, c.field, c.label, c.needs_review) for c in found] == [
        (1, "a", "neutrophil", False),
        (2, "c", "ig", True),
        (3, "c", "monocyte", False),
    ]


def test_submit_report_rejects_diagnosis_then_accepts():
    reviewer = SlideReviewer(cells("neutrophil"), fields=1, second_opinion=lambda c: {})
    assert reviewer.tool("submit_report", {"report": "Probable infection."})["accepted"] is False
    assert reviewer.tool("submit_report", {"report": "1 white cell counted."})["accepted"] is True
    assert reviewer.review.rejected_reports == 1


def test_flag_cell_refuses_diagnostic_reason_and_duplicates():
    reviewer = SlideReviewer(cells("ig", "neutrophil", confidence=0.5, needs_review=True), 2, lambda c: {})
    assert reviewer.tool("flag_cell", {"cell_id": 0, "priority": "high", "reason": "looks like leukemia"})["flagged"] is False
    reviewer.tool("flag_cell", {"cell_id": 0, "priority": "high", "reason": "immature granulocyte"})
    reviewer.tool("flag_cell", {"cell_id": 0, "priority": "high", "reason": "again"})
    summary = reviewer.tool("get_slide_summary", {})
    assert summary["white_cells"] == 2 and summary["marked_for_review"] == 2
    assert summary["enough_for_differential"] is False
    assert len(reviewer.review.flags) == 1 and reviewer.review.rejected_flags == 1


def test_list_cells_filters_and_sorts_by_confidence():
    found = [
        Cell(0, "f", [0, 0, 1, 1], "neutrophil", 0.9, False),
        Cell(1, "f", [0, 0, 1, 1], "monocyte", 0.4, True),
        Cell(2, "f", [0, 0, 1, 1], "monocyte", 0.6, True),
    ]
    reviewer = SlideReviewer(found, 1, lambda c: {})
    listed = reviewer.tool("list_cells", {"needs_review_only": True})
    assert listed["matching"] == 2 and [c["cell_id"] for c in listed["cells"]] == [1, 2]
    assert reviewer.tool("list_cells", {"cell_type": "neutrophil"})["matching"] == 1


def test_second_opinion_budget():
    reviewer = SlideReviewer(cells("monocyte", "monocyte"), 1, lambda c: {"label": "neutrophil"}, max_second_opinions=1)
    first = reviewer.tool("ask_second_opinion", {"cell_id": 0})
    assert first["label"] == "neutrophil" and first["classifier"] == "monocyte"
    assert "error" in reviewer.tool("ask_second_opinion", {"cell_id": 1})
    assert reviewer.review.second_opinions[0]["agrees"] is False


def test_review_slide_end_to_end_with_fake_clients():
    reports = [backend_report(("neutrophil", 0.95), ("monocyte", 0.55)), backend_report()]
    images = [Image.new("RGB", (100, 100), "white"), Image.new("RGB", (100, 100), "white")]
    chat = FakeChat(
        [
            [("get_slide_summary", {}), ("get_reference_ranges", {})],
            [("list_cells", {"needs_review_only": True})],
            [("ask_second_opinion", {"cell_id": 1})],
            [("flag_cell", {"cell_id": 1, "priority": "high", "reason": "classifier and second look disagree"})],
            [("submit_report", {"report": "Probable infection."})],
            [("submit_report", {"report": "2 white cells counted, fewer than 100: indicative only."})],
        ]
    )
    vision_calls = []

    def vision(image: bytes, prompt: str):
        vision_calls.append(image)
        return "Segmented nucleus.\nLABEL: neutrophil", 50

    result = review_slide(reports, images, names=["f1", "f2"], chat=chat, vision=vision)

    assert result["accepted"] and result["report"].startswith("2 white cells")
    assert result["rejected_reports"] == 1
    assert result["summary"]["white_cells"] == 2 and result["summary"]["red_cells"] == 24
    assert result["summary"]["fields_without_white_cell"] == 1
    assert result["flags"][0]["field"] == "f1" and result["flags"][0]["box"] == [10, 10, 50, 50]
    assert result["second_opinions"][0]["second_opinion"] == "neutrophil"
    assert len(vision_calls) == 1 and vision_calls[0][:2] == b"\xff\xd8"
    assert result["llm_calls"] == 6
    assert result["tokens"] == {
        "agent_prompt": 600, "agent_completion": 60, "agent": 660, "second_opinions": 50, "total": 710,
    }
    tool_messages = [m for m in chat.seen[-1] if m["role"] == "tool"]
    assert json.loads(tool_messages[-1]["content"])["accepted"] is False


def test_review_slide_without_report_is_not_accepted():
    chat = FakeChat([[]] * 3)
    result = review_slide([backend_report(("neutrophil", 0.9))], [b""], chat=chat, max_steps=3)
    assert result["accepted"] is False and result["report"] == ""


def test_summary_skips_reference_ranges_on_a_tiny_count():
    few = SlideReviewer(cells("neutrophil", "neutrophil"), fields=1, second_opinion=lambda c: {})
    assert few.summary()["outside_reference_ranges"] is None
    enough = SlideReviewer(cells(*["neutrophil"] * 20), fields=1, second_opinion=lambda c: {})
    assert set(enough.summary()["outside_reference_ranges"]) == {"neutrophil", "lymphocyte", "monocyte", "eosinophil"}
