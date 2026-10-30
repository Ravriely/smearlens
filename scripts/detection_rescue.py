"""Keep white cells just under the detection threshold as cells to review, and score it on TXL-PBC.

    uv run python scripts/detection_rescue.py fields --out runs/rescue/truth.json
    modal run scripts/modal_app.py $(uv run python scripts/detection_rescue.py args runs/rescue/truth.json) \
        --out runs/rescue/reports.json
    uv run python scripts/detection_rescue.py score runs/rescue/truth.json runs/rescue/reports.json
"""

import argparse
import json
from pathlib import Path

from smearlens.analyze import merge_kept
from smearlens.datasets import Box, iter_txl_pbc
from smearlens.metrics import MatchResult, match_centers
from smearlens.pipeline import Candidate

THRESHOLD = 0.9995
RULES = ("every_field", "empty_field_only")
FLOORS = (0.9995, 0.999, 0.998, 0.995, 0.99, 0.98, 0.95, 0.9)


def build(data_dir: Path, splits: list[str]) -> list[dict]:
    return [
        {
            "path": str(s.image_path),
            "split": split,
            "source": s.source,
            "wbc_boxes": [[b.x0, b.y0, b.x1, b.y1] for b in s.boxes if b.label == "WBC"],
        }
        for split in splits
        for s in iter_txl_pbc(data_dir, split)
    ]


def kept_boxes(report: dict, floor: float, rule: str) -> tuple[list[Box], list[Box]]:
    cands = [Candidate(Box("WBC", *c["box"]), 0.0) for c in report["candidate_boxes"]]
    probs = [c["p_wbc"] for c in report["candidate_boxes"]]
    kept = merge_kept(cands, probs, floor, report["rbc_px"])
    p_of = {c.box: p for c, p in zip(cands, probs)}
    sure, unsure = [b for b in kept if p_of[b] >= THRESHOLD], [b for b in kept if p_of[b] < THRESHOLD]
    if rule == "empty_field_only":
        unsure = unsure[:1] if not sure else []
    return sure, unsure


def score(truth: list[dict], reports: list[dict]) -> dict:
    by_path = {r["path"]: r for r in reports}
    groups = sorted({(t["split"], t["source"]) for t in truth}) + sorted({(t["split"], "all") for t in truth})
    out = {}
    for rule, split, source in [(r, *g) for r in RULES for g in groups]:
        part = [t for t in truth if t["split"] == split and source in ("all", t["source"])]
        rows = {}
        for floor in FLOORS:
            total, sure_total, uncertain = MatchResult(0, 0, 0), MatchResult(0, 0, 0), MatchResult(0, 0, 0)
            empty_fields = count_error = 0
            for t in part:
                r = by_path[t["path"]]
                truth_boxes = [Box("WBC", *b) for b in t["wbc_boxes"]]
                sure, unsure = kept_boxes(r, floor, rule)
                both = match_centers(sure + unsure, truth_boxes)
                only_sure = match_centers(sure, truth_boxes)
                total += both
                sure_total += only_sure
                uncertain += MatchResult(both.tp - only_sure.tp, len(unsure) - (both.tp - only_sure.tp), 0)
                empty_fields += bool(truth_boxes) and not (sure or unsure)
                count_error += abs(len(sure) + len(unsure) - len(truth_boxes))
            rows[str(floor)] = {
                "white_cells_truth": total.tp + total.fn,
                "found": total.tp,
                "false": total.fp,
                "missed": total.fn,
                "precision": round(total.precision, 4),
                "recall": round(total.recall, 4),
                "f1": round(total.f1, 4),
                "count_error_per_field": round(count_error / len(part), 3),
                "fields_with_white_cells_but_none_found": empty_fields,
                "added_to_review_right": uncertain.tp,
                "added_to_review_wrong": uncertain.fp,
            }
        out[f"{rule}/{split}/{source}"] = {"fields": len(part), "by_floor": rows}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fields = sub.add_parser("fields")
    fields.add_argument("--data-dir", type=Path, default=Path("data"))
    fields.add_argument("--splits", nargs="+", default=["val", "test"])
    fields.add_argument("--out", type=Path, required=True)
    args_cmd = sub.add_parser("args")
    args_cmd.add_argument("truth", type=Path)
    score_cmd = sub.add_parser("score")
    score_cmd.add_argument("truth", type=Path)
    score_cmd.add_argument("reports", type=Path)
    score_cmd.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.command == "fields":
        rows = build(args.data_dir, args.splits)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))
        print(len(rows), "fields")
    elif args.command == "args":
        print(" ".join(f"--image {r['path']}" for r in json.loads(args.truth.read_text())))
    else:
        result = score(json.loads(args.truth.read_text()), json.loads(args.reports.read_text()))
        text = json.dumps(result, indent=2)
        if args.out:
            args.out.write_text(text + "\n")
        for group, row in result.items():
            print(group, row["fields"], "fields")
            for floor, m in row["by_floor"].items():
                print(f"  {floor:>7} F1 {m['f1']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} "
                      f"err {m['count_error_per_field']:.2f} empty {m['fields_with_white_cells_but_none_found']} "
                      f"+review {m['added_to_review_right']} right / {m['added_to_review_wrong']} wrong")


if __name__ == "__main__":
    main()
