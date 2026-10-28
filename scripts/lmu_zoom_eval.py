import argparse
import json
from pathlib import Path

import numpy as np

from lmu_eval import LMU_TO_PBC
from review_coverage import softmax


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy of the type probe on Munich cells rescaled to several zoom factors.")
    parser.add_argument("--features", type=Path, default=Path("runs/lmu/features_zoom.npz"))
    parser.add_argument("--probe", type=Path, default=Path("runs/glimmer-aug/probe_aug.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/lmu/zoom.json"))
    args = parser.parse_args()

    probe = dict(np.load(args.probe))
    classes = [str(c) for c in probe["classes"]]
    data = np.load(args.features)
    codes = np.array([p.split("/")[0] for p in data["paths"]])
    known = np.isin(codes, list(LMU_TO_PBC))
    truth = np.array([LMU_TO_PBC[c] for c in codes[known]])

    rows = []
    for zoom in data["zooms"]:
        x = (data[f"features_{zoom}"][known].astype(np.float32) - probe["mean"]) / probe["std"]
        pred = np.array(classes)[softmax(x @ probe["weight"].T + probe["bias"]).argmax(1)]
        recall = {c: round(float((pred[truth == c] == c).mean()), 3) for c in sorted(set(truth))}
        rows.append(
            {
                "zoom": float(zoom),
                "accuracy": round(float((pred == truth).mean()), 4),
                "balanced_accuracy": round(float(np.mean(list(recall.values()))), 4),
                "recall": recall,
            }
        )
        print(json.dumps(rows[-1]))
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
