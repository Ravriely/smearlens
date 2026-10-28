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
