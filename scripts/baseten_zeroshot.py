import argparse
import csv
import random
import time
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from smearlens.muse import ask_image, parse_label
from smearlens.pbc import PBC_CLASSES, pbc_root, read_splits

PROMPT = (
    "This is a single-cell crop from a peripheral blood smear (May-Grunwald-Giemsa stain). "
    "Classify the cell in the center into exactly one of these classes: "
    + ", ".join(PBC_CLASSES)
    + ". 'ig' means immature granulocyte (promyelocyte, myelocyte or metamyelocyte). "
    "Give one short sentence of morphological reasoning, then end with a line 'LABEL: <class>'."
)


def classify(path: Path, max_tokens: int) -> dict:
    start = time.time()
    try:
        text, tokens = ask_image(path.read_bytes(), PROMPT, max_tokens)
    except (urllib.error.URLError, TimeoutError) as error:
        return {"pred": "error", "text": str(error), "tokens": 0, "seconds": time.time() - start}
    return {
        "pred": parse_label(text, PBC_CLASSES),
        "text": text.replace("\n", " "),
        "tokens": tokens,
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

    cells = [c for c in read_splits(pbc_root(args.data_dir).parent / "splits.csv", args.data_dir) if c.split == "test"]
    rng = random.Random(args.seed)
    sample = []
    for label in PBC_CLASSES:
        pool = [c for c in cells if c.label == label]
        sample += rng.sample(pool, min(args.per_class, len(pool)))
    if len(sample) > args.max_calls:
        raise SystemExit(f"{len(sample)} calls planned, above --max-calls {args.max_calls}")

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda c: classify(c.path, args.max_tokens), sample))

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
