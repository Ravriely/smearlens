import argparse
import json
from pathlib import Path

import numpy as np

from lmu_eval import LMU_TO_PBC
from review_coverage import softmax


def train_softmax(x, y, w, b, classes: int, steps: int, lr: float, decay: float, weight=None):
    onehot = np.eye(classes, dtype=np.float32)[y]
    weight = np.ones(len(y), dtype=np.float32) if weight is None else weight
    for _ in range(steps):
        p = softmax(x @ w.T + b)
        g = (p - onehot) * weight[:, None]
        w -= lr * (g.T @ x / weight.sum() + decay * w)
        b -= lr * g.sum(0) / weight.sum()
    return w, b


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt the type probe to the Munich lab with a few labelled cells.")
    parser.add_argument("--lmu", type=Path, default=Path("runs/lmu/features_resize0.npz"))
    parser.add_argument("--pbc", type=Path, default=Path("runs/glimmer/pbc_features.npz"))
    parser.add_argument("--probe", type=Path, default=Path("runs/glimmer-aug/probe_aug.npz"))
    parser.add_argument("--shots", default="0,5,10,25,50,100")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("runs/lmu/adapt.json"))
    args = parser.parse_args()

    probe = dict(np.load(args.probe))
    classes = [str(c) for c in probe["classes"]]
    mean, std = probe["mean"], probe["std"]

    lmu = np.load(args.lmu)
    codes = np.array([p.split("/")[0] for p in lmu["paths"]])
    known = np.isin(codes, list(LMU_TO_PBC))
    x_lmu = ((lmu["features"][known].astype(np.float32)) - mean) / std
    y_lmu = np.array([classes.index(LMU_TO_PBC[c]) for c in codes[known]])

    pbc = np.load(args.pbc)
    train = pbc["splits"] == "train"
    x_pbc = (pbc["features"][train].astype(np.float32) - mean) / std
    y_pbc = pbc["labels"][train]

    present = sorted(set(y_lmu.tolist()))
    results = []
    for shots in [int(s) for s in args.shots.split(",")]:
        runs = []
        for seed in range(args.repeats if shots else 1):
            rng = np.random.default_rng(seed)
            picked = np.concatenate(
                [rng.choice(np.flatnonzero(y_lmu == c), min(shots, (y_lmu == c).sum() // 2), replace=False) for c in present]
            ) if shots else np.array([], dtype=int)
            held = np.setdiff1d(np.arange(len(y_lmu)), picked)
            w, b = probe["weight"].astype(np.float32).copy(), probe["bias"].astype(np.float32).copy()
            if shots:
                x = np.concatenate([x_pbc, x_lmu[picked]])
                y = np.concatenate([y_pbc, y_lmu[picked]])
                weight = np.concatenate([np.ones(len(y_pbc)), np.full(len(picked), len(y_pbc) / len(picked))]).astype(np.float32)
                w, b = train_softmax(x, y, w, b, len(classes), steps=200, lr=0.05, decay=1e-4, weight=weight)
            pred = softmax(x_lmu[held] @ w.T + b).argmax(1)
            truth = y_lmu[held]
            recalls = {classes[c]: float((pred[truth == c] == c).mean()) for c in present}
            runs.append({"accuracy": float((pred == truth).mean()), "balanced": float(np.mean(list(recalls.values()))), "recalls": recalls})
        results.append(
            {
                "shots_per_class": shots,
                "labelled_cells": shots * len(present),
                "accuracy": round(float(np.mean([r["accuracy"] for r in runs])), 4),
                "accuracy_std": round(float(np.std([r["accuracy"] for r in runs])), 4),
                "balanced_accuracy": round(float(np.mean([r["balanced"] for r in runs])), 4),
                "recall": {k: round(float(np.mean([r["recalls"][k] for r in runs])), 3) for k in runs[0]["recalls"]},
            }
        )
        print(json.dumps(results[-1]), flush=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
