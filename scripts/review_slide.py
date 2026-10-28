import argparse
import json
import time
from pathlib import Path

from PIL import Image

from smearlens.agent import Cell, SlideReviewer
from smearlens.datasets import Box, txl_pbc_root
from smearlens.muse import ask_image, parse_label
from smearlens.pipeline import CANDIDATE_CLASSES, NUCLEATED, PROMPT, crop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SmearLens review agent on one slide (a set of fields).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--cells", type=Path, default=Path("results/pipeline/cells_glimmer_rbc.json"))
    parser.add_argument("--source", default=None, help="keep only fields from one source (PBC, BCCD, BCDD)")
    parser.add_argument("--fields", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("results/review"))
    args = parser.parse_args()

    rows = json.loads(args.cells.read_text())
    if args.source:
        rows = [r for r in rows if r["source"] == args.source]
    fields = sorted({r["image"] for r in rows})[: args.fields]
    rows = [r for r in rows if r["image"] in fields and r["label"] in NUCLEATED]
    cells = [Cell(i, r["image"], r["box"], r["label"], r["confidence"], r["muse_label"]) for i, r in enumerate(rows)]

    images_dir = txl_pbc_root(args.data_dir) / "images" / args.split

    vision_tokens = [0]

    def second_opinion(cell: Cell) -> dict:
        image = Image.open(images_dir / cell.field).convert("RGB")
        text, tokens = ask_image(crop(image, Box("WBC", *cell.box)), PROMPT)
        vision_tokens[0] += tokens
        return {"cell_id": cell.id, "label": parse_label(text, CANDIDATE_CLASSES), "reasoning": text[-400:]}

    started = time.time()
    review = SlideReviewer(cells, len(fields), second_opinion, args.max_steps).run()

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"slide-{args.source or 'all'}-{len(fields)}"
    (args.out / f"{name}.md").write_text(review.report + "\n")
    (args.out / f"{name}.json").write_text(
        json.dumps(
            {
                "fields": len(fields),
                "white_cells": len(cells),
                "seconds": round(time.time() - started, 1),
                "agent_tokens": review.tokens,
                "second_opinion_tokens": vision_tokens[0],
                "rejected_reports": review.rejected_reports,
                "flags": review.flags,
                "steps": review.steps,
            },
            indent=2,
        )
    )
    print(review.report)
    print(f"\n{len(review.steps)} tool calls, {len(review.flags)} flags, {review.rejected_reports} rejected reports, "
          f"{review.tokens} + {vision_tokens[0]} tokens, {time.time() - started:.0f} s -> {args.out / name}.md")


if __name__ == "__main__":
    main()
