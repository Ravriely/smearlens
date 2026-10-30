import json
from pathlib import Path

import numpy as np

from smearlens.datasets import Box
from smearlens.metrics import center_distance, contains_center
from smearlens.pipeline import Candidate, dedupe, differential, probe_predict, rbc_scale

PROMPTS = (
    ("WBC", "cell with a large purple nucleus"),
    ("WBC", "neutrophil"),
    ("RBC", "red blood cell"),
    ("Platelet", "small purple dot"),
)
SAM_THRESHOLD = 0.3
WBC_MIN = 0.3
RBC_MIN = 0.5
UPSCALE_BELOW = 512
RBC_COUNT_MIN = 0.3
RBC_SIZE = (0.85, 1.6)
PLATELET_SIZE = (0.22, 0.6)
RESCUE_FLOOR = 0.95
SAM_INPUT = 1008
TILE_BELOW_RBC = 130
TILE_TARGET_RBC = 180
TILE_MIN_SIDE = 500
MAX_TILES_PER_SIDE = 4
TILE_MARGIN = 0.2


def candidates_for(detections: list[dict], wbc_min: float, rbc_min: float) -> dict[str, list[Candidate]]:
    found: dict[str, list[Candidate]] = {}
    for d in detections:
        if d["label"] == "Platelet":
            continue
        threshold = wbc_min if d["label"] == "WBC" else rbc_min
        found.setdefault(d["image"], []).extend(
            Candidate(Box("WBC", *b), s) for b, s in zip(d["boxes"], d["scores"]) if s >= threshold
        )
    return {image: dedupe(c) for image, c in found.items()}


def rbc_sizes(detections: list[dict]) -> dict[str, float]:
    return {
        d["image"]: rbc_scale([max(b[2] - b[0], b[3] - b[1]) for b, s in zip(d["boxes"], d["scores"]) if s >= 0.5])
        for d in detections
        if d["label"] == "RBC"
    }


def tiles_per_side(size: tuple[int, int], rbc_px: float | None) -> int:
    """How many windows per side bring red cells back to the scale SAM 3 sees them at on a single field.

    SAM 3 resizes every image to SAM_INPUT px. On the TXL-PBC fields it finds red cells when they span
    150 to 200 px of its input; on a wider field (more, smaller cells) it loses them. rbc_px is the
    median red cell of a first pass, None when that pass found none.
    """
    longest = max(size)
    if rbc_px is None:
        return 2 if min(size) >= TILE_MIN_SIDE else 1
    seen = rbc_px * SAM_INPUT / longest
    if seen >= TILE_BELOW_RBC:
        return 1
    return max(2, min(MAX_TILES_PER_SIDE, -(-int(TILE_TARGET_RBC) // max(1, int(seen)))))


def plan_windows(size: tuple[int, int], n: int, margin: float = TILE_MARGIN) -> list[tuple[Box, Box]]:
    """n x n windows as (window, core): cores tile the image, each window adds `margin` of a core around it."""
    width, height = size
    plans = []
    for row in range(n):
        for col in range(n):
            core = Box("core", width * col / n, height * row / n, width * (col + 1) / n, height * (row + 1) / n)
            dx, dy = margin * width / n, margin * height / n
            window = Box("window", max(0.0, core.x0 - dx), max(0.0, core.y0 - dy),
                         min(float(width), core.x1 + dx), min(float(height), core.y1 + dy))
            plans.append((window, core))
    return plans


def in_core(box: list[float], core: Box, size: tuple[int, int]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    right = cx < core.x1 or core.x1 >= size[0]
    bottom = cy < core.y1 or core.y1 >= size[1]
    return core.x0 <= cx and core.y0 <= cy and right and bottom


def merge_kept(cands: list[Candidate], probs: np.ndarray, threshold: float, rbc_px: float) -> list[Box]:
    kept: list[Candidate] = []
    for c, p in sorted(zip(cands, probs), key=lambda t: -t[1]):
        if p < threshold:
            break
        if any(contains_center(k.box, c.box) or center_distance(k.box, c.box) < rbc_px for k in kept):
            continue
        kept.append(c)
    return [k.box for k in kept]


def wbc_probability(features: np.ndarray, probe: dict) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float32) - probe["mean"]) / probe["std"]
    return 1 / (1 + np.exp(-np.clip(x @ probe["weight"] + probe["bias"], -30, 30)))


def load_probes(detection: Path, types: Path, coverage: Path, lmu_eval: Path) -> dict:
    return {
        "detection": dict(np.load(detection)),
        "type": dict(np.load(types)),
        "temperature": json.loads(coverage.read_text())["temperature"],
        "review_cutoff": json.loads(lmu_eval.read_text())["review_cutoff_from_pbc_10pct"],
    }


def count_outside(detections: list[dict], label: str, min_score: float, cells: list[Box]) -> int:
    boxes = [
        Candidate(Box(label, *b), s)
        for d in detections
        if d["label"] == label
        for b, s in zip(d["boxes"], d["scores"])
        if s >= min_score
    ]
    return sum(not any(contains_center(w, c.box) for w in cells) for c in dedupe(boxes))


def relative_size(box: Box, rbc_px: float) -> float:
    return max(box.x1 - box.x0, box.y1 - box.y0) / rbc_px


def count_cells(
    detections: list[dict], label: str, min_score: float, cells: list[Box], rbc_px: float, size: tuple[float, float]
) -> int:
    boxes = [
        Candidate(Box(label, *b), s)
        for d in detections
        if d["label"] == label
        for b, s in zip(d["boxes"], d["scores"])
        if s >= min_score and size[0] <= relative_size(Box(label, *b), rbc_px) <= size[1]
    ]
    return sum(
        not any(contains_center(w, c.box) or contains_center(c.box, w) for w in cells) for c in dedupe(boxes)
    )


def white_cells(cands: list[Candidate], features: np.ndarray, rbc_px: float, probes: dict) -> list[dict]:
    detection = probes["detection"]
    p_wbc = wbc_probability(features, detection)
    kept = merge_kept(cands, p_wbc, float(detection["threshold"]), rbc_px)
    rescued = not kept and len(cands) > 0 and float(p_wbc.max()) >= RESCUE_FLOOR
    if rescued:
        kept = [cands[int(np.argmax(p_wbc))].box]
    index = {c.box: i for i, c in enumerate(cands)}
    rows = [index[b] for b in kept]
    types = probe_predict(features[rows], probes["type"], probes["temperature"]) if rows else []
    return [
        {
            "box": [round(v, 1) for v in (b.x0, b.y0, b.x1, b.y1)],
            "p_wbc": round(float(p_wbc[i]), 5),
            "type": label,
            "confidence": round(confidence, 4),
            "needs_review": rescued or confidence < probes["review_cutoff"],
            "uncertain_detection": rescued,
        }
        for b, i, (label, confidence) in zip(kept, rows, types)
    ]


def report(size: tuple[int, int], detections: list[dict], rbc_px: float, cells: list[dict], timing: dict) -> dict:
    boxes = [Box("WBC", *c["box"]) for c in cells]
    formula = differential([Candidate(b, 0.0, c["type"]) for b, c in zip(boxes, cells)])
    return {
        "image": {"width": size[0], "height": size[1]},
        "rbc_px": round(rbc_px, 1),
        "white_cells": cells,
        "white_cell_count": len(cells),
        "needs_review_count": sum(c["needs_review"] for c in cells),
        "red_cell_count": count_cells(detections, "RBC", RBC_COUNT_MIN, boxes, rbc_px, RBC_SIZE),
        "platelet_count": count_cells(detections, "Platelet", SAM_THRESHOLD, boxes, rbc_px, PLATELET_SIZE),
        "differential_percent": {k: round(100 * v, 1) for k, v in formula.items() if v},
        "timing_s": {k: round(v, 3) for k, v in timing.items()},
    }
