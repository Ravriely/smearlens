import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from smearlens.datasets import CLASSES, Box, iter_txl_pbc
from smearlens.metrics import MatchResult, match_boxes, match_centers

PROMPTS = {"WBC": "white blood cell", "RBC": "red blood cell", "Platelet": "platelet"}
COLORS = {"WBC": (124, 58, 237), "RBC": (220, 38, 38), "Platelet": (234, 179, 8)}


def load_model(version: str):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf

    checkpoint = download_ckpt_from_hf(version=version)
    model = build_sam3_image_model(
        checkpoint_path=checkpoint, load_from_HF=False, enable_inst_interactivity=True
    )
    return model, Sam3Processor(model)


def upscale_factor(image: Image.Image, below: int) -> int:
    return max(1, -(-below // min(image.size))) if below else 1


def segment_concept(
    processor, state, label: str, prompt: str, factor: int = 1, size: tuple[int, int] | None = None
) -> tuple[list[Box], np.ndarray, list[float]]:
    output = processor.set_text_prompt(prompt=prompt, state=state)
    boxes = [Box(label, *(float(v) / factor for v in b)) for b in output["boxes"].cpu().numpy()]
    masks = output["masks"].cpu().numpy()[:, 0]
    if factor > 1 and len(masks):
        masks = np.stack([np.asarray(Image.fromarray(m.astype(np.uint8)).resize(size, Image.NEAREST)) for m in masks])
    scores = [float(x) for x in output["scores"].float().cpu().numpy()]
    return boxes, masks, scores


def segment_box(model, state, box: Box) -> np.ndarray:
    masks, _, _ = model.predict_inst(
        state,
        point_coords=None,
        point_labels=None,
        box=np.array([[box.x0, box.y0, box.x1, box.y1]]),
        multimask_output=False,
    )
    return masks[0].astype(bool)


def fill_ratio(mask: np.ndarray, box: Box) -> float:
    x0, y0 = int(box.x0), int(box.y0)
    x1, y1 = int(np.ceil(box.x1)), int(np.ceil(box.y1))
    inside = mask[y0:y1, x0:x1].sum()
    return float(inside / box.area) if box.area else 0.0


def draw_overlay(image: Image.Image, layers: list[tuple[str, np.ndarray]], truth: list[Box]) -> Image.Image:
    canvas = image.convert("RGBA")
    for label, mask in layers:
        tint = Image.new("RGBA", canvas.size, COLORS[label] + (0,))
        alpha = Image.fromarray((mask * 110).astype(np.uint8))
        tint.putalpha(alpha)
        canvas = Image.alpha_composite(canvas, tint)
    draw = ImageDraw.Draw(canvas)
    for b in truth:
        draw.rectangle((b.x0, b.y0, b.x1, b.y1), outline=COLORS[b.label] + (255,), width=1)
    return canvas.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SAM 3 cell segmentation on TXL-PBC.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--wbc-prompt", default=PROMPTS["WBC"])
    parser.add_argument("--platelet-prompt", default=PROMPTS["Platelet"])
    parser.add_argument("--upscale-below", type=int, default=0, help="upscale images whose short side is below this")
    parser.add_argument("--out", type=Path, default=Path("runs/sam3"))
    args = parser.parse_args()

    prompts = {**PROMPTS, "WBC": args.wbc_prompt, "Platelet": args.platelet_prompt}
    model, processor = load_model(args.version)
    processor.set_confidence_threshold(args.threshold)
    args.out.mkdir(parents=True, exist_ok=True)

    totals: dict[str, MatchResult] = defaultdict(MatchResult)
    center_totals: dict[str, MatchResult] = defaultdict(MatchResult)
    count_errors: dict[str, list[int]] = defaultdict(list)
    wbc_fill: list[float] = []
    per_image, detections = [], []
    started = time.perf_counter()

    samples = [s for _, s in zip(range(args.limit), iter_txl_pbc(args.data_dir, args.split))]
    for sample in samples:
        state = processor.set_image(Image.open(sample.image_path).convert("RGB"))
        for box in (b for b in sample.boxes if b.label == "WBC"):
            wbc_fill.append(fill_ratio(segment_box(model, state, box), box))

    for i, sample in enumerate(samples):
        image = Image.open(sample.image_path).convert("RGB")
        factor = upscale_factor(image, args.upscale_below)
        state = processor.set_image(image.resize((image.width * factor, image.height * factor), Image.LANCZOS))
        layers, row = [], {"image": sample.image_path.name, "source": sample.source}
        for label in CLASSES:
            truth = [b for b in sample.boxes if b.label == label]
            predicted, masks, scores = segment_concept(processor, state, label, prompts[label], factor, image.size)
            result = match_boxes(predicted, truth)
            totals[label] += result
            center_totals[label] += match_centers(predicted, truth)
            count_errors[label].append(len(predicted) - len(truth))
            row[label] = {"truth": len(truth), "predicted": len(predicted), "tp": result.tp}
            detections.append(
                {
                    "image": sample.image_path.name,
                    "label": label,
                    "boxes": [[b.x0, b.y0, b.x1, b.y1] for b in predicted],
                    "scores": scores,
                }
            )
            if len(masks):
                layers.append((label, masks.any(axis=0)))
        draw_overlay(image, layers, sample.boxes).save(args.out / f"{sample.image_path.stem}.jpg")
        per_image.append(row)
        print(f"[{i + 1}/{args.limit}] {sample.image_path.name} ({sample.source})", flush=True)

    summary = {
        "version": args.version,
        "prompts": prompts,
        "threshold": args.threshold,
        "upscale_below": args.upscale_below,
        "split": args.split,
        "images": len(per_image),
        "seconds": round(time.perf_counter() - started, 1),
        "concept": {
            label: {
                "precision": round(totals[label].precision, 3),
                "recall": round(totals[label].recall, 3),
                "f1": round(totals[label].f1, 3),
                "mean_abs_count_error": round(float(np.mean(np.abs(count_errors[label]))), 2),
                "center_precision": round(center_totals[label].precision, 3),
                "center_recall": round(center_totals[label].recall, 3),
                "center_f1": round(center_totals[label].f1, 3),
            }
            for label in CLASSES
        },
        "wbc_box_prompt_fill_ratio": {
            "mean": round(float(np.mean(wbc_fill)), 3) if wbc_fill else None,
            "min": round(float(np.min(wbc_fill)), 3) if wbc_fill else None,
        },
        "per_image": per_image,
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    (args.out / "detections.json").write_text(json.dumps(detections))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
