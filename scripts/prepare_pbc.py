import argparse
from collections import Counter
from pathlib import Path

from smearlens.pbc import PBC_CLASSES, build_splits, pbc_root, write_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Split PBC into train/val/test for the cell classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    cells = build_splits(args.data_dir)
    out = pbc_root(args.data_dir).parent / "splits.csv"
    write_splits(cells, out, args.data_dir)

    counts = Counter((c.label, c.split) for c in cells)
    print(f"{'class':<14}{'train':>7}{'val':>7}{'test':>7}")
    for label in PBC_CLASSES:
        print(f"{label:<14}" + "".join(f"{counts[label, s]:>7}" for s in ("train", "val", "test")))
    print(f"Wrote {len(cells)} cells to {out}")


if __name__ == "__main__":
    main()
