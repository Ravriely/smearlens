"""Wide fields: tile 2x2 test fields of one source into one image and score the backend on it.

    uv run python scripts/mosaic_eval.py build --out runs/mosaic
    modal run scripts/modal_app.py $(uv run python scripts/mosaic_eval.py args runs/mosaic/truth.json) \
        --out runs/mosaic/reports.json
    uv run python scripts/mosaic_eval.py score runs/mosaic/truth.json runs/mosaic/reports.json
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from smearlens.datasets import Box, iter_txl_pbc
from smearlens.metrics import MatchResult, match_centers

SEAM_PX = 3


def build(data_dir: Path, out: Path, split: str = "test") -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    by_source: dict[str, list] = {}
    for s in iter_txl_pbc(data_dir, split):
        by_source.setdefault(s.source, []).append(s)
        rows.append(single(s))
    for source, samples in by_source.items():
        for g in range(len(samples) // 4):
            group = samples[4 * g: 4 * g + 4]
            w, h = max(s.width for s in group), max(s.height for s in group)
            mosaic = Image.new("RGB", (2 * w, 2 * h), "white")
            wbc, rbc, plt = [], 0, 0
            for i, s in enumerate(group):
                dx, dy = w * (i % 2), h * (i // 2)
                with Image.open(s.image_path) as im:
                    mosaic.paste(im.convert("RGB"), (dx, dy))
                wbc += [[b.x0 + dx, b.y0 + dy, b.x1 + dx, b.y1 + dy] for b in s.boxes if b.label == "WBC"]
                rbc += s.count("RBC")
                plt += s.count("Platelet")
            path = out / f"{source}-{g:02d}.png"
            mosaic.save(path)
            rows.append({"path": str(path), "kind": "mosaic", "source": source, "wbc_boxes": wbc,
                         "red_cells": rbc, "platelets": plt, "fields": [str(s.image_path) for s in group]})
    return rows


def single(s) -> dict:
    return {"path": str(s.image_path), "kind": "single", "source": s.source,
            "wbc_boxes": [[b.x0, b.y0, b.x1, b.y1] for b in s.boxes if b.label == "WBC"],
            "red_cells": s.count("RBC"), "platelets": s.count("Platelet")}


def off_seams(boxes: list[list[float]], size: dict, kind: str) -> list[Box]:
    if kind != "mosaic":
        return [Box("WBC", *b) for b in boxes]
    sx, sy = size["width"] / 2, size["height"] / 2
    return [
        Box("WBC", *b) for b in boxes
        if not (b[0] <= sx + SEAM_PX and b[2] >= sx - SEAM_PX) and not (b[1] <= sy + SEAM_PX and b[3] >= sy - SEAM_PX)
    ]


def score(truth: list[dict], reports: list[dict], ignore_seams: bool = False) -> dict:
    by_path = {r["path"]: r for r in reports}
    in_mosaic = {f for t in truth if t["kind"] == "mosaic" for f in t["fields"]}
    out = {}
    for kind in ("single", "mosaic"):
        for source in sorted({t["source"] for t in truth}) + ["all"]:
            part = [t for t in truth if t["kind"] == kind and source in ("all", t["source"])
                    and (kind == "mosaic" or t["path"] in in_mosaic)]
            m = MatchResult(0, 0, 0)
            rbc_truth = rbc_found = plt_truth = plt_found = 0
            for t in part:
                r = by_path[t["path"]]
                predicted = [c["box"] for c in r["white_cells"]]
                if ignore_seams:
                    m += match_centers(off_seams(predicted, r["image"], t["kind"]),
                                       off_seams(t["wbc_boxes"], r["image"], t["kind"]))
                else:
                    m += match_centers([Box("WBC", *b) for b in predicted], [Box("WBC", *b) for b in t["wbc_boxes"]])
                rbc_truth += t["red_cells"]
                rbc_found += r["red_cell_count"]
                plt_truth += t["platelets"]
                plt_found += r["platelet_count"]
            out[f"{kind}/{source}"] = {
                "images": len(part), "white_cells_truth": m.tp + m.fn, "found": m.tp, "false": m.fp,
                "f1": round(m.f1, 4), "precision": round(m.precision, 4), "recall": round(m.recall, 4),
                "red_cells_truth": rbc_truth, "red_cells_found": rbc_found,
                "platelets_truth": plt_truth, "platelets_found": plt_found,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data-dir", type=Path, default=Path("data"))
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--split", default="test", choices=("train", "val", "test"))
    a = sub.add_parser("args")
    a.add_argument("truth", type=Path)
    s = sub.add_parser("score")
    s.add_argument("truth", type=Path)
    s.add_argument("reports", type=Path)
    s.add_argument("--out", type=Path)
    s.add_argument("--ignore-seams", action="store_true",
                   help="drop white cell boxes, found or true, that touch a seam between the tiled fields")
    args = parser.parse_args()
    if args.command == "build":
        rows = build(args.data_dir, args.out, args.split)
        (args.out / "truth.json").write_text(json.dumps(rows, indent=1))
        print(sum(r["kind"] == "mosaic" for r in rows), "mosaics")
    elif args.command == "args":
        rows = json.loads(args.truth.read_text())
        used = {f for r in rows if r["kind"] == "mosaic" for f in r["fields"]}
        print(" ".join(f"--image {r['path']}" for r in rows if r["kind"] == "mosaic" or r["path"] in used))
    else:
        result = score(json.loads(args.truth.read_text()), json.loads(args.reports.read_text()), args.ignore_seams)
        text = json.dumps(result, indent=2)
        if args.out:
            args.out.write_text(text + "\n")
        for k, v in result.items():
            print(f"{k:16} n={v['images']:3} WBC F1 {v['f1']:.3f} (P {v['precision']:.2f} R {v['recall']:.2f}) "
                  f"RBC {v['red_cells_found']}/{v['red_cells_truth']} PLT {v['platelets_found']}/{v['platelets_truth']}")


if __name__ == "__main__":
    main()
