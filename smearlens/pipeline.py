import io
from collections import Counter
from dataclasses import dataclass

from PIL import Image

from smearlens.datasets import Box
from smearlens.metrics import contains_center

NUCLEATED = ("basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil")
CANDIDATE_CLASSES = NUCLEATED + ("platelet", "not_wbc")
PRIORITY = ("ig", "erythroblast", "basophil")

PBC_PREFIXES = {
    "BA": "basophil",
    "EO": "eosinophil",
    "LY": "lymphocyte",
    "MO": "monocyte",
    "BNE": "neutrophil",
    "SNE": "neutrophil",
    "NEU": "neutrophil",
    "ERB": "erythroblast",
    "IG": "ig",
    "MY": "ig",
    "MMY": "ig",
    "PMY": "ig",
    "PLATELET": "platelet",
}

PROMPT = (
    "This crop comes from a peripheral blood smear (May-Grunwald-Giemsa stain). "
    "A detector proposed the object in the center as a white blood cell, but it can be wrong. "
    "Classify the object in the center into exactly one of: "
    + ", ".join(CANDIDATE_CLASSES)
    + ". 'ig' means immature granulocyte (promyelocyte, myelocyte or metamyelocyte). "
    "'not_wbc' means a red blood cell, a cell fragment, debris or background. "
    "Give one short sentence of morphological reasoning, then end with a line 'LABEL: <class>'."
)


@dataclass
class Candidate:
    box: Box
    score: float
    label: str = ""
    text: str = ""
    confidence: float | None = None


def pbc_label(original_name: str) -> str | None:
    prefix = original_name.split("_")[0].upper()
    return PBC_PREFIXES.get(prefix)


def crop(image: Image.Image, box: Box, scale: float = 1.8, min_side: int = 96) -> bytes:
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    side = max(box.x1 - box.x0, box.y1 - box.y0) * scale
    half = max(side, min_side) / 2
    region = image.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
    buffer = io.BytesIO()
    region.convert("RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def dedupe(candidates: list[Candidate]) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if any(contains_center(k.box, candidate.box) or contains_center(candidate.box, k.box) for k in kept):
            continue
        kept.append(candidate)
    return kept


def differential(cells: list[Candidate]) -> dict[str, float]:
    counts = Counter(c.label for c in cells if c.label in NUCLEATED)
    total = sum(counts.values())
    return {label: counts[label] / total for label in NUCLEATED} if total else {}


def review_flags(cells: list[Candidate]) -> list[str]:
    flags = []
    for label in PRIORITY:
        n = sum(c.label == label for c in cells)
        if n:
            flags.append(f"{n} {label}")
    unclear = sum(c.label == "unparsed" or c.label.startswith("other:") for c in cells)
    if unclear:
        flags.append(f"{unclear} unclear")
    return flags


def probe_predict(features, probe, temperature: float = 1.0) -> list[tuple[str, float]]:
    import numpy as np

    x = (np.asarray(features, dtype=np.float32) - probe["mean"]) / probe["std"]
    logits = (x @ probe["weight"].T + probe["bias"]) / temperature
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    return [(str(probe["classes"][i]), float(p[i])) for i, p in zip(logits.argmax(1), probs)]


def probe_labels(features, probe) -> list[str]:
    return [label for label, _ in probe_predict(features, probe)]


PBC_RBC_PX = 74.6
PBC_FIELD_PX = 360


def rbc_scale(sizes: list[float], fallback: float = PBC_RBC_PX) -> float:
    import statistics

    return statistics.median(sizes) if sizes else fallback


def crop_at_scale(image: Image.Image, box: Box, rbc_px: float) -> bytes:
    import numpy as np

    side = round(PBC_FIELD_PX * rbc_px / PBC_RBC_PX)
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    pixels = np.asarray(image.convert("RGB"))
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    field = Image.new("RGB", (side, side), tuple(int(v) for v in np.median(border, axis=0)))
    field.paste(image.convert("RGB"), (-x0, -y0))
    buffer = io.BytesIO()
    field.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()
