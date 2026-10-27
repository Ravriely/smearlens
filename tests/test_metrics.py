import pytest

from smearlens.datasets import Box
from smearlens.metrics import MatchResult, iou, match_boxes


def box(x0, y0, x1, y1, label="WBC"):
    return Box(label, x0, y0, x1, y1)


def test_iou_identical_boxes():
    assert iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes():
    assert iou(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    assert iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_match_counts_each_truth_once():
    truth = [box(0, 0, 10, 10)]
    predicted = [box(0, 0, 10, 10), box(1, 0, 11, 10)]
    assert match_boxes(predicted, truth) == MatchResult(tp=1, fp=1, fn=0)


def test_match_prefers_best_overlap():
    truth = [box(0, 0, 10, 10), box(8, 0, 18, 10)]
    predicted = [box(7, 0, 17, 10), box(0, 0, 10, 10)]
    assert match_boxes(predicted, truth) == MatchResult(tp=2, fp=0, fn=0)


def test_match_below_threshold_is_a_miss():
    result = match_boxes([box(5, 0, 15, 10)], [box(0, 0, 10, 10)])
    assert result == MatchResult(tp=0, fp=1, fn=1)


def test_scores():
    result = MatchResult(tp=8, fp=2, fn=8)
    assert result.precision == pytest.approx(0.8)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(2 * 0.8 * 0.5 / 1.3)


def test_empty_scores_are_zero():
    assert MatchResult().f1 == 0.0
