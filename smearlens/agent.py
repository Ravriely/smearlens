import json
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from smearlens.muse import ENDPOINT, MODEL
from smearlens.pipeline import NUCLEATED

# Indicative adult ranges, relative counts per 100 white cells. Each lab uses its own.
REFERENCE_RANGES = {
    "neutrophil": (40, 75),
    "lymphocyte": (20, 45),
    "monocyte": (2, 10),
    "eosinophil": (1, 6),
    "basophil": (0, 2),
    "ig": (0, 1),
    "erythroblast": (0, 0),
}
MIN_CELLS_FOR_DIFFERENTIAL = 100
LOW_CONFIDENCE = 0.7

FORBIDDEN = re.compile(
    r"\b(diagnos\w*|leuk(a)?emi\w*|lymphoma\w*|myelodysplas\w*|infection\w*|sepsis|malignan\w*|cancer\w*|"
    r"treat(ment)?\w*|prescri\w*|medication\w*|therap\w*)\b",
    re.IGNORECASE,
)

SYSTEM = """You are the review assistant of SmearLens, a lab workflow tool for peripheral blood smears.
Your reader is the lab biologist who validates the slide. Your job is to prepare the review:
check the automatic cell count, point to what deserves a human look first, and write a short slide report.

Rules:
- You never give a diagnosis, a disease name, a treatment or medical advice. You describe what was counted
  and what should be checked. The biologist and the physician interpret.
- Every number in the report must come from a tool result.
- Use the tools: start with get_slide_summary and get_reference_ranges, inspect uncertain or priority cells
  with list_cells and ask_second_opinion, flag what the biologist should look at with flag_cell, then call
  submit_report once. Keep the report under 200 words.
- Say clearly when fewer than 100 white cells were counted: the differential is then only indicative."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_slide_summary",
            "description": "Counts of the analysed slide: fields, white cells by type, differential in percent, cells with low classifier confidence.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_reference_ranges",
            "description": "Indicative adult reference ranges of the white cell differential, in percent of white cells.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cells",
            "description": "List white cells, optionally only one type or only cells below a confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_type": {"type": "string", "enum": list(NUCLEATED)},
                    "max_confidence": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_second_opinion",
            "description": "Look again at one cell image with the vision model and return its label and reasoning.",
            "parameters": {
                "type": "object",
                "properties": {"cell_id": {"type": "integer"}},
                "required": ["cell_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_cell",
            "description": "Put one cell on the biologist's review list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_id": {"type": "integer"},
                    "priority": {"type": "string", "enum": ["high", "normal"]},
                    "reason": {"type": "string"},
                },
                "required": ["cell_id", "priority", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_report",
            "description": "Submit the final slide report (markdown) for the biologist.",
            "parameters": {
                "type": "object",
                "properties": {"report": {"type": "string"}},
                "required": ["report"],
            },
        },
    },
]


@dataclass
class Cell:
    id: int
    field: str
    box: list[float]
    label: str
    confidence: float | None
    muse_label: str


@dataclass
class Review:
    report: str = ""
    flags: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    tokens: int = 0
    rejected_reports: int = 0


def differential(cells: list[Cell]) -> dict[str, float]:
    counts = Counter(c.label for c in cells)
    return {t: round(100 * counts[t] / len(cells), 1) for t in NUCLEATED} if cells else {}


def out_of_range(diff: dict[str, float]) -> list[str]:
    return [t for t, v in diff.items() if not REFERENCE_RANGES[t][0] <= v <= REFERENCE_RANGES[t][1]]


def check_report(report: str) -> list[str]:
    return sorted({m.group(0).lower() for m in FORBIDDEN.finditer(report)})


class SlideReviewer:
    def __init__(self, cells: list[Cell], fields: int, second_opinion: Callable[[Cell], dict], max_steps: int = 16):
        self.cells = {c.id: c for c in cells}
        self.fields = fields
        self.second_opinion = second_opinion
        self.max_steps = max_steps
        self.review = Review()

    def tool(self, name: str, args: dict):
        cells = list(self.cells.values())
        if name == "get_slide_summary":
            diff = differential(cells)
            return {
                "fields": self.fields,
                "white_cells": len(cells),
                "enough_for_differential": len(cells) >= MIN_CELLS_FOR_DIFFERENTIAL,
                "counts": dict(Counter(c.label for c in cells)),
                "differential_percent": diff,
                "outside_reference_ranges": out_of_range(diff),
                "low_confidence_cells": sum(1 for c in cells if (c.confidence or 1) < LOW_CONFIDENCE),
                "classifier_and_detector_disagree": sum(1 for c in cells if c.muse_label != c.label),
            }
        if name == "get_reference_ranges":
            return {
                "unit": "percent of white cells",
                "note": "indicative adult ranges; the lab's own ranges prevail",
                "ranges": {t: list(r) for t, r in REFERENCE_RANGES.items()},
            }
        if name == "list_cells":
            picked = [
                c
                for c in cells
                if (not args.get("cell_type") or c.label == args["cell_type"])
                and ("max_confidence" not in args or (c.confidence or 1) <= args["max_confidence"])
            ]
            return [
                {"cell_id": c.id, "field": c.field, "type": c.label, "confidence": c.confidence, "detector_label": c.muse_label}
                for c in picked[:40]
            ]
        if name == "ask_second_opinion":
            cell = self.cells.get(int(args["cell_id"]))
            return self.second_opinion(cell) if cell else {"error": "unknown cell_id"}
        if name == "flag_cell":
            cell = self.cells.get(int(args["cell_id"]))
            if not cell:
                return {"error": "unknown cell_id"}
            self.review.flags.append(
                {"cell_id": cell.id, "field": cell.field, "type": cell.label, "priority": args["priority"], "reason": args["reason"]}
            )
            return {"flagged": cell.id, "review_list_size": len(self.review.flags)}
        if name == "submit_report":
            banned = check_report(args["report"])
            if banned:
                self.review.rejected_reports += 1
                return {"accepted": False, "reason": f"remove diagnostic or treatment wording: {', '.join(banned)}"}
            self.review.report = args["report"]
            return {"accepted": True}
        return {"error": f"unknown tool {name}"}

    def chat(self, messages: list[dict]) -> dict:
        body = {"model": MODEL, "max_tokens": 4000, "messages": messages, "tools": TOOLS}
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}", "Content-Type": "application/json"},
        )
        response = json.load(urllib.request.urlopen(request, timeout=300))
        self.review.tokens += response.get("usage", {}).get("total_tokens", 0)
        return response["choices"][0]["message"]

    def run(self) -> Review:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "A new slide is ready for review. Prepare the review."},
        ]
        for _ in range(self.max_steps):
            message = self.chat(messages)
            calls = message.get("tool_calls") or []
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls} if calls else {"role": "assistant", "content": message.get("content") or ""})
            if not calls:
                if self.review.report:
                    break
                messages.append({"role": "user", "content": "Use the tools and finish with submit_report."})
                continue
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.tool(name, args)
                self.review.steps.append({"tool": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
            if self.review.report:
                break
        return self.review
