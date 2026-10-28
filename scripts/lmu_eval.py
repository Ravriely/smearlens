import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from review_coverage import coverage_curve, ece, softmax

LMU_TO_PBC = {
    "BAS": "basophil",
    "EBO": "erythroblast",
    "EOS": "eosinophil",
    "LYT": "lymphocyte",
    "MON": "monocyte",
    "NGS": "neutrophil",
    "NGB": "neutrophil",
    "MMZ": "ig",
    "MYB": "ig",
    "PMO": "ig",
    "PMB": "ig",
}
UNSEEN = {"MYO": "myeloblast", "MOB": "monoblast", "LYA": "atypical lymphocyte", "KSC": "smudge cell"}


def main() -> None:
    parser = argparse.ArgumentParser(description="External test of the type probe on the Munich LMU dataset.")
    parser.add_argument("--features", type=Path, default=Path("runs/lmu/features_resize0.npz"))
    parser.add_argument("--probe", type=Path, default=Path("runs/glimmer-aug/probe_aug.npz"))
    parser.add_argument("--pbc-features", type=Path, default=Path("runs/glimmer/pbc_features.npz"))
    parser.add_argument("--coverage", type=Path, default=Path("runs/review-coverage.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/lmu/eval.json"))
    args = parser.parse_args()

    probe = dict(np.load(args.probe))
    classes = [str(c) for c in probe["classes"]]
    temperature = json.loads(args.coverage.read_text())["temperature"]

    def predict(features):
        x = (features.astype(np.float32) - probe["mean"]) / probe["std"]
        return softmax(x @ probe["weight"].T + probe["bias"], temperature)

    pbc = np.load(args.pbc_features)
    pbc_conf = predict(pbc["features"][pbc["splits"] == "test"]).max(1)
    review_cutoff = float(np.quantile(pbc_conf, 0.10))

    data = np.load(args.features)
    codes = np.array([p.split("/")[0] for p in data["paths"]])
    probs = predict(data["features"])
    pred = np.array([classes[i] for i in probs.argmax(1)])
    confidence = probs.max(1)

    known = np.isin(codes, list(LMU_TO_PBC))
    truth = np.array([LMU_TO_PBC.get(c, "") for c in codes])
    correct = pred == truth
    per_class = {}
    for label in sorted(set(truth[known])):
        mask = known & (truth == label)
        per_class[label] = {
            "cells": int(mask.sum()),
            "recall": round(float(correct[mask].mean()), 3),
            "predicted_as": dict(Counter(pred[mask]).most_common(3)),
        }
    unseen = {}
    for code, name in UNSEEN.items():
        mask = codes == code
        if mask.any():
            unseen[name] = {
                "cells": int(mask.sum()),
                "sent_to_review_percent": round(100 * float((confidence[mask] < review_cutoff).mean()), 1),
                "mean_confidence": round(float(confidence[mask].mean()), 3),
                "predicted_as": dict(Counter(pred[mask]).most_common(3)),
            }
    result = {
        "images": int(len(codes)),
        "missing_files": int(len(data["missing"])),
        "temperature": temperature,
        "review_cutoff_from_pbc_10pct": round(review_cutoff, 4),
        "known_types": {
            "cells": int(known.sum()),
            "accuracy": round(float(correct[known].mean()), 4),
            "balanced_accuracy": round(float(np.mean([v["recall"] for v in per_class.values()])), 4),
            "ece": round(ece(confidence[known], correct[known]), 4),
            "sent_to_review_percent": round(100 * float((confidence[known] < review_cutoff).mean()), 1),
            "errors_left_after_review_percent": round(
                100 * float(1 - correct[known & (confidence >= review_cutoff)].mean()), 2
            ),
            "coverage": coverage_curve(confidence[known], correct[known], (0, 0.05, 0.1, 0.2, 0.3, 0.5)),
            "per_class": per_class,
        },
        "unseen_types": unseen,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
