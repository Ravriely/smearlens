import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from smearlens.augment import augment
from smearlens.datasets import Box, iter_txl_pbc
from smearlens.metrics import MatchResult, center_distance, contains_center, match_centers
from smearlens.pipeline import Candidate, crop_at_scale, dedupe, rbc_scale

SPLITS = ("train", "val", "test")


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


def is_wbc(c: Candidate, truth: list[Box]) -> bool:
    return any(contains_center(t, c.box) for t in truth)


def embed(keys: list[str], crops: list[bytes], path: Path) -> np.ndarray:
    stored = dict(np.load(path)) if path.exists() else {}
    missing = [(k, c) for k, c in zip(keys, crops) if k not in stored]
    if missing:
        import modal

        from modal_glimmer_probe import app, embed_crops

        with modal.enable_output(), app.run():
            for i in range(0, len(missing), 2000):
                chunk = missing[i : i + 2000]
                stored.update({k: f for (k, _), f in zip(chunk, embed_crops.remote([c for _, c in chunk]))})
                np.savez(path, **stored)
    return np.stack([stored[k] for k in keys]).astype(np.float32)


def jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def fit_logistic(x: np.ndarray, y: np.ndarray, decay: float, steps: int = 400, lr: float = 0.1) -> tuple[np.ndarray, float]:
    w, b = np.zeros(x.shape[1], dtype=np.float32), 0.0
    pos = y.mean()
    weight = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos)).astype(np.float32)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(x @ w + b, -30, 30)))
        g = (p - y) * weight
        w -= lr * (x.T @ g / len(y) + decay * w)
        b -= lr * float(g.mean())
    return w, b


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Glimmer 'white cell or not' probe on SAM 3 candidates.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--tag", default="up512")
    parser.add_argument("--wbc-min", type=float, default=0.3)
    parser.add_argument("--rbc-min", type=float, default=0.5)
    parser.add_argument("--extra-tags", default="", help="comma-separated SAM runs whose WBC boxes are added as candidates")
    parser.add_argument("--augment-copies", type=int, default=0, help="stain-augmented copies of each train crop")
    parser.add_argument("--out", type=Path, default=Path("runs/detection-probe"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = {}
    for split in SPLITS:
        detections = json.loads((args.runs / f"detect-{split}-{args.tag}" / "detections.json").read_text())
        for extra in filter(None, args.extra_tags.split(",")):
            more = json.loads((args.runs / f"detect-{split}-{extra}" / "detections.json").read_text())
            detections += [d for d in more if d["label"] == "WBC"]
        candidates = candidates_for(detections, args.wbc_min, args.rbc_min)
        scales = rbc_sizes(detections)
        samples = list(iter_txl_pbc(args.data_dir, split))
        candidates = {s.image_path.name: candidates.get(s.image_path.name, []) for s in samples}
        keys, crops, labels, owners = [], [], [], []
        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            truth = [b for b in sample.boxes if b.label == "WBC"]
            for c in candidates[sample.image_path.name]:
                keys.append(f"{sample.image_path.name}:{c.box.x0:.0f},{c.box.y0:.0f},{c.box.x1:.0f},{c.box.y1:.0f}")
                crops.append(crop_at_scale(image, c.box, scales.get(sample.image_path.name, 74.6)))
                labels.append(int(is_wbc(c, truth)))
                owners.append(sample.image_path.name)
        print(f"{split}: {len(samples)} images, {len(keys)} candidates, {sum(labels)} on a true white cell", flush=True)
        features = embed(keys, crops, args.out / f"features_{split}.npz")
        if split == "train" and args.augment_copies:
            rng = random.Random(0)
            aug_keys, aug_crops = [], []
            for i in range(args.augment_copies):
                for k, c in zip(keys, crops):
                    aug_keys.append(f"{k}:aug{i}")
                    aug_crops.append(jpeg(augment(Image.open(io.BytesIO(c)).convert("RGB"), rng)))
            features = np.concatenate([features, embed(aug_keys, aug_crops, args.out / "features_train_aug.npz")])
            labels = labels * (1 + args.augment_copies)
        data[split] = {"x": features, "y": np.array(labels, dtype=np.float32), "samples": samples,
                       "candidates": candidates, "owners": owners, "scales": scales}

    mean, std = data["train"]["x"].mean(0), data["train"]["x"].std(0) + 1e-6
    norm = {s: (data[s]["x"] - mean) / std for s in SPLITS}
    best = None
    for decay in (1e-4, 1e-3, 1e-2):
        w, b = fit_logistic(norm["train"], data["train"]["y"], decay)
        p = 1 / (1 + np.exp(-np.clip(norm["val"] @ w + b, -30, 30)))
        for threshold in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99995):
            result = evaluate(data["val"], p, threshold)
            if best is None or result["f1"] > best[0]:
                best = (result["f1"], decay, float(threshold), w, b)
    _, decay, threshold, w, b = best
    probs = {s: 1 / (1 + np.exp(-np.clip(norm[s] @ w + b, -30, 30))) for s in SPLITS}

    summary = {
        "candidates": {s: int(len(data[s]["y"])) for s in SPLITS},
        "positives": {s: int(data[s]["y"].sum()) for s in SPLITS},
        "weight_decay": decay,
        "threshold": round(threshold, 5),
        "val": evaluate(data["val"], probs["val"], threshold),
        "test": evaluate(data["test"], probs["test"], threshold),
        "test_by_source": {
            source: evaluate(data["test"], probs["test"], threshold, source)
            for source in ("PBC", "BCCD", "BCDD")
        },
    }
    np.savez(args.out / "probe_wbc.npz", weight=w, bias=b, mean=mean, std=std, threshold=threshold)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    test_rows = []
    keys_test = list(zip(data["test"]["owners"], probs["test"].tolist()))
    i = 0
    for sample in data["test"]["samples"]:
        for c in data["test"]["candidates"][sample.image_path.name]:
            owner, p = keys_test[i]
            test_rows.append({"image": owner, "box": [c.box.x0, c.box.y0, c.box.x1, c.box.y1], "p_wbc": round(p, 4)})
            i += 1
    (args.out / "test_candidates.json").write_text(json.dumps(test_rows))
    print(json.dumps(summary, indent=2))


def merge_kept(cands: list[Candidate], probs: np.ndarray, threshold: float, rbc_px: float) -> list[Box]:
    kept: list[Candidate] = []
    for c, p in sorted(zip(cands, probs), key=lambda t: -t[1]):
        if p < threshold:
            break
        if any(contains_center(k.box, c.box) or center_distance(k.box, c.box) < rbc_px for k in kept):
            continue
        kept.append(c)
    return [k.box for k in kept]


def evaluate(split: dict, probs: np.ndarray, threshold: float, source: str | None = None) -> dict:
    total, errors, i = MatchResult(), [], 0
    for sample in split["samples"]:
        cands = split["candidates"][sample.image_path.name]
        rbc_px = split["scales"].get(sample.image_path.name, 74.6)
        kept = merge_kept(cands, probs[i : i + len(cands)], threshold, rbc_px)
        i += len(cands)
        if source and sample.source != source:
            continue
        truth = [b for b in sample.boxes if b.label == "WBC"]
        total += match_centers(kept, truth)
        errors.append(abs(len(kept) - len(truth)))
    return {
        "images": len(errors),
        "precision": round(total.precision, 3),
        "recall": round(total.recall, 3),
        "f1": round(total.f1, 3),
        "count_mean_abs_error": round(float(np.mean(errors)), 2) if errors else None,
    }


if __name__ == "__main__":
    main()
