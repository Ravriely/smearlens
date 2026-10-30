"""Build the example slide of the web demo and record a live run of it on the deployed app.

    uv run python scripts/pbc_slide.py fields --splits test --out runs/slide-pbc/truth-test.json
    uv run python scripts/slide_demo.py pick runs/slide-pbc/truth-test.json
    SMEARLENS_WEB_TOKEN=... uv run python scripts/slide_demo.py record

`pick` copies a few TXL-PBC test fields (PBC source) into web/slide/. `record` sends each field to
POST /api/analyze, one at a time as the page does, then all of them to POST /api/review-slide, and
writes web/slide/slide.json and web/slide/review.json, which the page shows as the offline example.
"""

import argparse
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / "web" / "slide"
API = "https://ravriely--smearlens-app-web.us-east.modal.run"
MIX = {"neutrophil": 4, "lymphocyte": 3, "monocyte": 2, "eosinophil": 2, "basophil": 1}


def pick(truth: list[dict], mix: dict[str, int]) -> list[dict]:
    chosen = []
    for kind, n in mix.items():
        chosen += sorted((t for t in truth if t["type"] == kind), key=lambda t: Path(t["path"]).name)[:n]
    return sorted(chosen, key=lambda t: Path(t["path"]).name)


def post(url: str, body: bytes, token: str, content_type: str, timeout: int) -> dict:
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"content-type": content_type, "x-smearlens-token": token}
    )
    try:
        return json.load(urllib.request.urlopen(request, timeout=timeout))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{url}: HTTP {exc.code} {exc.read()[:300]!r}") from None


def record(api: str, token: str) -> None:
    import base64

    fields = json.loads((OUT / "fields.json").read_text())
    names = [f["name"] for f in fields]
    analyses = []
    started = time.perf_counter()
    for name in names:
        result = post(f"{api}/api/analyze", (OUT / name).read_bytes(), token, "image/png", 240)
        print(f"{name}: {result['white_cell_count']} white cells, {result['round_trip_s']} s", flush=True)
        analyses.append(result)
    analysed = time.perf_counter()
    payload = {
        "fields": [
            {"name": n, "analysis": a, "image": base64.b64encode((OUT / n).read_bytes()).decode()}
            for n, a in zip(names, analyses)
        ]
    }
    review = post(f"{api}/api/review-slide", json.dumps(payload).encode(), token, "application/json", 400)
    done = time.perf_counter()
    slide = {
        "fields": [{"name": n, "src": f"slide/{n}", **a} for n, a in zip(names, analyses)],
        "analysis_s": round(analysed - started, 1),
        "review_s": round(done - analysed, 1),
    }
    (OUT / "slide.json").write_text(json.dumps(slide, indent=1) + "\n")
    (OUT / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    s = review["summary"]
    print(json.dumps({k: s[k] for k in ("fields", "white_cells", "differential_percent", "marked_for_review",
                                        "red_cells_per_field", "platelets_per_field")}))
    print(f"analysis {slide['analysis_s']} s, review {slide['review_s']} s, {review['llm_calls']} model calls, "
          f"{len(review['second_opinions'])} second opinions, {len(review['flags'])} flags, "
          f"{review['tokens']['total']} tokens, accepted={review['accepted']}")
    truth = Counter(f["truth"] for f in fields)
    print("labelled cells:", dict(truth), "| typed on their field:",
          sum(any(c["type"] == f["truth"] for c in a["white_cells"]) for f, a in zip(fields, analyses)), "/", len(fields))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pick_cmd = sub.add_parser("pick")
    pick_cmd.add_argument("truth", type=Path, help="output of scripts/pbc_slide.py fields")
    record_cmd = sub.add_parser("record")
    record_cmd.add_argument("--api", default=API)
    args = parser.parse_args()

    if args.command == "pick":
        OUT.mkdir(parents=True, exist_ok=True)
        fields = []
        for i, t in enumerate(pick(json.loads(args.truth.read_text()), MIX), 1):
            name = f"field-{i:02d}.png"
            shutil.copyfile(t["path"], OUT / name)
            fields.append({"name": name, "truth": t["type"], "source": Path(t["path"]).name})
            print(name, t["type"], Path(t["path"]).name)
        (OUT / "fields.json").write_text(json.dumps(fields, indent=1) + "\n")
    else:
        record(args.api.rstrip("/"), os.environ["SMEARLENS_WEB_TOKEN"])


if __name__ == "__main__":
    main()
