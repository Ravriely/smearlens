from pathlib import Path

import pytest

from smearlens.datasets import txl_pbc_root
from smearlens.pbc import build_splits, hash_split, pbc_root, txl_pbc_splits

DATA_DIR = Path(__file__).parent.parent / "data"
needs_data = pytest.mark.skipif(
    not (pbc_root(DATA_DIR).exists() and txl_pbc_root(DATA_DIR).exists()),
    reason="PBC or TXL-PBC not downloaded",
)


def test_hash_split_is_stable():
    assert hash_split("NEU_1.jpg") == hash_split("NEU_1.jpg")


def test_hash_split_proportions():
    splits = [hash_split(f"cell_{i}.jpg") for i in range(10_000)]
    assert splits.count("test") / len(splits) == pytest.approx(0.15, abs=0.02)
    assert splits.count("val") / len(splits) == pytest.approx(0.15, abs=0.02)


@needs_data
def test_cells_reused_in_txl_pbc_keep_their_split():
    pinned = txl_pbc_splits(DATA_DIR)
    assert len(pinned) == 500
    by_name = {c.path.name: c.split for c in build_splits(DATA_DIR)}
    assert all(by_name[name] == split for name, split in pinned.items())
