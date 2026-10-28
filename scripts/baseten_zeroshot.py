import argparse
import base64
import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from smearlens.pbc import PBC_CLASSES, pbc_root, read_splits

ENDPOINT = "https://inference.baseten.co/v1/chat/completions"
MODEL = "meta-models/muse-spark-open-1026"

PROMPT = (
    "This is a single-cell crop from a peripheral blood smear (May-Grunwald-Giemsa stain). "
    "Classify the cell in the center into exactly one of these classes: "
    + ", ".join(PBC_CLASSES)
    + ". 'ig' means immature granulocyte (promyelocyte, myelocyte or metamyelocyte). "
    "Give one short sentence of morphological reasoning, then end with a line 'LABEL: <class>'."
)


def classify(path: Path, api_key: str, max_tokens: int) -> dict:
    image = base64.b64encode(path.read_bytes()).decode()
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"},
    )
    start = time.time()
    for attempt in range(3):
        try:
            response = json.load(urllib.request.urlopen(request, timeout=180))
            break
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 2:
                return {"pred": "error", "text": str(error), "tokens": 0, "seconds": time.time() - start}
            time.sleep(5 * (attempt + 1))
    text = response["choices"][0]["message"].get("content") or ""
    match = re.search(r"LABEL:\s*\**\s*([a-z]+)", text, re.IGNORECASE)
    pred = match.group(1).lower() if match else "unparsed"
    if pred not in PBC_CLASSES and pred != "unparsed":
        pred = f"other:{pred}"
    return {
        "pred": pred,
        "text": text.strip().replace("\n", " "),
        "tokens": response.get("usage", {}).get("total_tokens", 0),
        "seconds": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot PBC cell classification with Muse Spark on Baseten.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/baseten_zeroshot.csv"))
    args = parser.parse_args()

    api_key = os.environ["BASETEN_API_KEY"]
    cells = [c for c in read_splits(pbc_root(args.data_dir).parent / "splits.csv", args.data_dir) if c.split == "test"]
    rng = random.Random(args.seed)
    sample = []
    for label in PBC_CLASSES:
        pool = [c for c in cells if c.label == label]
        sample += rng.sample(pool, min(args.per_class, len(pool)))
    if len(sample) > args.max_calls:
        raise SystemExit(f"{len(sample)} calls planned, above --max-calls {args.max_calls}")

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda c: classify(c.path, api_key, args.max_tokens), sample))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "pred", "tokens", "seconds", "text"])
        for cell, result in zip(sample, results):
            writer.writerow(
                [cell.path.name, cell.label, result["pred"], result["tokens"], f"{result['seconds']:.1f}", result["text"]]
            )

    correct = sum(c.label == r["pred"] for c, r in zip(sample, results))
    print(f"Accuracy: {correct}/{len(sample)} = {correct / len(sample):.1%}")
    print(f"Tokens: {sum(r['tokens'] for r in results)}")
    confusion = Counter((c.label, r["pred"]) for c, r in zip(sample, results))
    preds = list(PBC_CLASSES) + sorted({r["pred"] for r in results} - set(PBC_CLASSES))
    print("true \\ pred".ljust(14) + "".join(p[:6].rjust(7) for p in preds))
    for label in PBC_CLASSES:
        print(label.ljust(14) + "".join(str(confusion[label, p]).rjust(7) for p in preds))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
