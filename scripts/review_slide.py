import argparse
import json
from pathlib import Path

from smearlens.agent import review_slide


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SmearLens review agent on one slide analysed by the backend.")
    parser.add_argument("--run", type=Path, required=True, help="output of `modal run scripts/modal_app.py --out ...`")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True, help="writes <out>.json and <out>.md")
    args = parser.parse_args()

    fields = json.loads(args.run.read_text())
    paths = [Path(f["path"]) for f in fields]
    result = review_slide(fields, paths, names=[p.name for p in paths], max_steps=args.max_steps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(result["report"] + "\n")
    print(result["report"])
    s, t = result["summary"], result["tokens"]
    print(
        f"\n{s['fields']} fields, {s['white_cells']} white cells, {s['marked_for_review']} marked for review | "
        f"{len(result['steps'])} tool calls, {len(result['second_opinions'])} second opinions, "
        f"{len(result['flags'])} flags, {result['rejected_reports']} rejected reports | "
        f"{t['agent_prompt']} + {t['agent_completion']} agent tokens, {t['second_opinions']} vision tokens, "
        f"{result['seconds']} s -> {args.out}.json"
    )


if __name__ == "__main__":
    main()
