import argparse
import json
from pathlib import Path

import numpy as np

from smearlens.analyze import (
    PLATELET_SIZE,
    RBC_COUNT_MIN,
    RBC_MIN,
    RBC_SIZE,
    SAM_THRESHOLD,
    WBC_MIN,
    candidates_for,
    count_cells,
    count_outside,
    merge_kept,
    rbc_sizes,
    wbc_probability,
)
from smearlens.datasets import iter_txl_pbc
from smearlens.pipeline import PBC_RBC_PX

SOURCES = ("PBC", "BCCD", "BCDD")
RBC_GRID = [(m, lo, hi) for m in (0.3, 0.4, 0.5) for lo in (0, 0.7, 0.75, 0.8, 0.85, 0.9) for hi in (1.4, 1.6, 2.0, 99)]
PLATELET_GRID = [(lo, hi) for lo in (0, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28) for hi in (0.4, 0.6, 99)]


def load_split(runs: Path, data_dir: Path, features: Path, probe: dict, split: str) -> list[dict]:
    detections = json.loads((runs / f"detect-{split}-up512" / "detections.json").read_text())
    more = json.loads((runs / f"detect-{split}-neutrophil" / "detections.json").read_text())
    detections += [d for d in more if d["label"] == "WBC"]
    stored = dict(np.load(features / f"features_{split}.npz"))
    candidates = candidates_for(detections, WBC_MIN, RBC_MIN)
    scales = rbc_sizes(detections)
    by_image: dict[str, list[dict]] = {}
    for d in detections:
        by_image.setdefault(d["image"], []).append(d)
    images = []
    for sample in iter_txl_pbc(data_dir, split):
        name = sample.image_path.name
        cands, rbc_px = candidates.get(name, []), scales.get(name, PBC_RBC_PX)
        kept = []
        if cands:
            keys = [f"{name}:{c.box.x0:.0f},{c.box.y0:.0f},{c.box.x1:.0f},{c.box.y1:.0f}" for c in cands]
            p = wbc_probability(np.stack([stored[k] for k in keys]), probe)
            kept = merge_kept(cands, p, float(probe["threshold"]), rbc_px)
        images.append({
            "source": sample.source,
            "rbc_px": rbc_px,
            "white": kept,
            "detections": by_image.get(name, []),
            "truth": {label: sample.count(label) for label in ("WBC", "RBC", "Platelet")},
        })
    return images


def counts(image: dict, filtered: bool, rbc: tuple = (RBC_COUNT_MIN, *RBC_SIZE), platelet: tuple = PLATELET_SIZE) -> dict:
    dets, white, px = image["detections"], image["white"], image["rbc_px"]
    if filtered:
        return {
            "WBC": len(white),
            "RBC": count_cells(dets, "RBC", rbc[0], white, px, rbc[1:]),
            "Platelet": count_cells(dets, "Platelet", SAM_THRESHOLD, white, px, platelet),
        }
    return {
        "WBC": len(white),
        "RBC": count_outside(dets, "RBC", RBC_MIN, white),
        "Platelet": count_outside(dets, "Platelet", SAM_THRESHOLD, white),
    }


def errors(images: list[dict], predicted: list[dict], label: str) -> dict:
    diff = [p[label] - i["truth"][label] for i, p in zip(images, predicted)]
    return {
        "mae": round(float(np.mean(np.abs(diff))), 2),
        "bias": round(float(np.mean(diff)), 2),
        "truth_mean": round(float(np.mean([i["truth"][label] for i in images])), 1),
    }


def table(images: list[dict], filtered: bool) -> dict:
    predicted = [counts(i, filtered) for i in images]
    out = {"all": {label: errors(images, predicted, label) for label in ("WBC", "RBC", "Platelet")}, "images": len(images)}
    for source in SOURCES:
        rows = [(i, p) for i, p in zip(images, predicted) if i["source"] == source]
        out[source] = {label: errors([i for i, _ in rows], [p for _, p in rows], label) for label in ("RBC", "Platelet")}
        out[source]["images"] = len(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Red cell and platelet counts before and after the size filter.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--features", type=Path, default=Path("runs/detection-probe-neutrophil"))
    parser.add_argument("--probe", type=Path, default=Path("models/detection_probe.npz"))
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/14-count-filter.json"))
    args = parser.parse_args()

    probe = dict(np.load(args.probe))
    val = load_split(args.runs, args.data_dir, args.features, probe, "val")
    rbc_grid = sorted(
        (float(np.mean([abs(counts(i, True, rbc=r)["RBC"] - i["truth"]["RBC"]) for i in val])), r) for r in RBC_GRID
    )
    platelet_grid = sorted(
        (float(np.mean([abs(counts(i, True, platelet=p)["Platelet"] - i["truth"]["Platelet"]) for i in val])), p)
        for p in PLATELET_GRID
    )
    test = load_split(args.runs, args.data_dir, args.features, probe, "test")
    summary = {
        "filter": {"rbc_min_score": RBC_COUNT_MIN, "rbc_size": RBC_SIZE, "platelet_size": PLATELET_SIZE},
        "val_grid_best": {
            "rbc": [{"mae": round(m, 3), "min_score": r[0], "size": r[1:]} for m, r in rbc_grid[:8]],
            "platelet": [{"mae": round(m, 3), "size": p} for m, p in platelet_grid[:8]],
        },
        "val": {"before": table(val, False), "after": table(val, True)},
        "test": {"before": table(test, False), "after": table(test, True)},
        "test_wbc_counts_identical": all(counts(i, False)["WBC"] == counts(i, True)["WBC"] for i in test),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
