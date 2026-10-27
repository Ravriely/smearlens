from pathlib import Path

import pytest

from smearlens.datasets import ALLOWED_SOURCES, iter_txl_pbc, parse_yolo_line, txl_pbc_root

DATA_DIR = Path(__file__).parent.parent / "data"


def test_parse_yolo_line_to_pixels():
    box = parse_yolo_line("1 0.5 0.5 0.2 0.4", 100, 200)
    assert box.label == "RBC"
    assert (box.x0, box.y0, box.x1, box.y1) == pytest.approx((40, 60, 60, 140))


def test_parse_yolo_line_clamps_to_image():
    box = parse_yolo_line("0 0.05 0.95 0.2 0.2", 100, 100)
    assert (box.x0, box.y1) == (0.0, 100)


def test_parse_yolo_line_skips_malformed():
    assert parse_yolo_line("1 0.5 0.5", 100, 100) is None


@pytest.mark.skipif(not txl_pbc_root(DATA_DIR).exists(), reason="TXL-PBC not downloaded")
def test_txl_pbc_excludes_unlicensed_sources():
    samples = list(iter_txl_pbc(DATA_DIR, "test"))
    assert samples
    assert {s.source for s in samples} <= ALLOWED_SOURCES
    assert all(s.boxes for s in samples)
    for s in samples:
        for b in s.boxes:
            assert 0 <= b.x0 < b.x1 <= s.width
            assert 0 <= b.y0 < b.y1 <= s.height
