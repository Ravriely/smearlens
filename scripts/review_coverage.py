import argparse
import json
from pathlib import Path

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(p[np.arange(len(labels)), labels] + 1e-12).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    return float(min(grid, key=lambda t: nll(logits, labels, t)))


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def coverage_curve(confidence: np.ndarray, correct: np.ndarray, fractions) -> list[dict]:
    order = np.argsort(confidence)
    rows = []
    for f in fractions:
        reviewed = int(round(f * len(order)))
        auto = order[reviewed:]
        rows.append(
            {
                "reviewed_percent": round(100 * f, 1),
                "reviewed_cells": reviewed,
                "errors_left_percent": round(100 * float(1 - correct[auto].mean()), 2) if len(auto) else 0.0,
                "errors_caught_percent": round(
                    100 * float((~correct[order[:reviewed]]).sum() / max(1, (~correct).sum())), 1
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the type probe and measure the review curve.")
    parser.add_argument("--features", type=Path, default=Path("runs/glimmer/pbc_features.npz"))
    parser.add_argument("--probe", type=Path, default=Path("runs/glimmer-aug/probe_aug.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/review-coverage.json"))
    args = parser.parse_args()

    data = np.load(args.features)
    probe = dict(np.load(args.probe))
    x = (data["features"].astype(np.float32) - probe["mean"]) / probe["std"]
    logits = x @ probe["weight"].T + probe["bias"]
    labels, splits = data["labels"], data["splits"]
    val, test = splits == "val", splits == "test"

    temperature = fit_temperature(logits[val], labels[val])
    result = {"temperature": round(temperature, 3)}
    for name, t in (("raw", 1.0), ("calibrated", temperature)):
        p = softmax(logits[test], t)
        confidence = p.max(1)
        correct = p.argmax(1) == labels[test]
        result[name] = {
            "accuracy": round(float(correct.mean()), 4),
            "ece": round(ece(confidence, correct), 4),
            "nll": round(nll(logits[test], labels[test], t), 4),
            "mean_confidence": round(float(confidence.mean()), 4),
        }
    p = softmax(logits[test], temperature)
    confidence, correct = p.max(1), p.argmax(1) == labels[test]
    result["coverage"] = coverage_curve(confidence, correct, (0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3))
    result["test_cells"] = int(test.sum())
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
