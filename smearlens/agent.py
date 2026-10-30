import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

from smearlens.datasets import Box
from smearlens.muse import ENDPOINT, MODEL
from smearlens.pipeline import CANDIDATE_CLASSES, NUCLEATED, PROMPT, crop

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
MIN_CELLS_FOR_RANGES = 20
LOW_CONFIDENCE = 0.7
MAX_SECOND_OPINIONS = 12
MAX_LISTED = 40
KNOWN_LIMITS = [
    "The second opinion model is a generalist: 65 % right on single PBC cells, weak on ig and erythroblasts "
    "(experiment 01). The classifier is 97.6 % right on the same data (experiment 05).",
    "Red cells are undercounted on crowded fields (experiment 14); the red cell count is indicative.",
    "Reference ranges are indicative adult ranges; the lab's own ranges prevail.",
]

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
- Start with get_slide_summary and get_reference_ranges. Then list the cells marked for review
  (list_cells with needs_review_only). Ask a second opinion on every cell of a rare type (ig, erythroblast),
  then on the other marked cells, least confident first, until the second opinion budget is used.
- The second opinion comes from a generalist vision model that is weaker than the classifier (see
  known_limits). A disagreement is a reason for a human look, never a correction: do not change counts
  and do not present the second opinion as the right answer.
- Flag with flag_cell the cells the biologist should open first: priority high for a rare type or a
  disagreement on a cell the classifier is unsure about, normal otherwise. Then call submit_report once.
- The report (markdown, under 220 words) has four parts: sample adequacy; differential with the types
  outside the reference ranges; review list (the size of the whole review queue, then the flagged cells
  first); limits (from known_limits, in one or two lines).
- Say clearly when fewer than 100 white cells were counted: the differential is then only indicative.
  Below 20 white cells, outside_reference_ranges is null: do not compare with the reference ranges at all,
  just say the count is too small for that. Never quote tool field names in the report.
- A cell with uncertain_detection was kept only because the field had no sure white cell: it may not be
  a white cell at all. Say so in the review list, and flag it.
- If most types are outside the reference ranges, say the overall pattern is unusual and that slide
  quality and field selection should be checked before the differential is used."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_slide_summary",
            "description": "Counts of the analysed slide: fields, white cells by type, differential in percent, "
            "types outside the reference ranges, cells marked for review, red cells and platelets.",
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
            "description": f"List white cells, least confident first, at most {MAX_LISTED}. "
            "Filter by type, by maximum classifier confidence, or only the cells marked for review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_type": {"type": "string", "enum": list(NUCLEATED)},
                    "max_confidence": {"type": "number"},
                    "needs_review_only": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_second_opinion",
            "description": "Look again at one cell image with the vision model and return its label and reasoning. "
            f"At most {MAX_SECOND_OPINIONS} per slide.",
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

Chat = Callable[[list[dict], list[dict]], dict]
Vision = Callable[[bytes, str], tuple[str, int]]
ImageLike = bytes | str | Path | Image.Image


@dataclass
class Cell:
    id: int
    field: str
    box: list[float]
    label: str
    confidence: float
    needs_review: bool
    uncertain_detection: bool = False


@dataclass
class Review:
    report: str = ""
    flags: list[dict] = field(default_factory=list)
    second_opinions: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    vision_tokens: int = 0
    rejected_reports: int = 0
    rejected_flags: int = 0


def differential(cells: list[Cell]) -> dict[str, float]:
    counts = Counter(c.label for c in cells)
    return {t: round(100 * counts[t] / len(cells), 1) for t in NUCLEATED} if cells else {}


def out_of_range(diff: dict[str, float]) -> list[str]:
    return [t for t, v in diff.items() if not REFERENCE_RANGES[t][0] <= v <= REFERENCE_RANGES[t][1]]


def check_report(report: str) -> list[str]:
    return sorted({m.group(0).lower() for m in FORBIDDEN.finditer(report)})


def cells_from_reports(reports: Sequence[dict], names: Sequence[str]) -> list[Cell]:
    cells = []
    for name, report in zip(names, reports):
        for c in report.get("white_cells", []):
            if c["type"] in NUCLEATED:
                cells.append(
                    Cell(len(cells) + 1, name, c["box"], c["type"], c["confidence"], bool(c["needs_review"]),
                         bool(c.get("uncertain_detection")))
                )
    return cells


def baseten_chat(messages: list[dict], tools: list[dict], max_tokens: int = 4000, retries: int = 3) -> dict:
    body = {"model": MODEL, "max_tokens": max_tokens, "messages": messages, "tools": tools}
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            return json.load(urllib.request.urlopen(request, timeout=300))
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


class SlideReviewer:
    def __init__(
        self,
        cells: list[Cell],
        fields: int,
        second_opinion: Callable[[Cell], dict],
        chat: Chat = baseten_chat,
        counts: dict | None = None,
        max_steps: int = 16,
        max_second_opinions: int = MAX_SECOND_OPINIONS,
    ):
        self.cells = {c.id: c for c in cells}
        self.fields = fields
        self.second_opinion = second_opinion
        self.chat_fn = chat
        self.counts = counts or {}
        self.max_steps = max_steps
        self.max_second_opinions = max_second_opinions
        self.review = Review()

    def summary(self) -> dict:
        cells = list(self.cells.values())
        diff = differential(cells)
        return {
            "fields": self.fields,
            "fields_without_white_cell": self.counts.get("fields_without_white_cell"),
            "white_cells": len(cells),
            "enough_for_differential": len(cells) >= MIN_CELLS_FOR_DIFFERENTIAL,
            "counts": dict(Counter(c.label for c in cells)),
            "differential_percent": diff,
            "outside_reference_ranges": out_of_range(diff) if len(cells) >= MIN_CELLS_FOR_RANGES else None,
            "marked_for_review": sum(c.needs_review for c in cells),
            "uncertain_detections": sum(c.uncertain_detection for c in cells),
            "marked_for_review_by_type": dict(Counter(c.label for c in cells if c.needs_review)),
            "below_confidence_0.7": sum(c.confidence < LOW_CONFIDENCE for c in cells),
            "red_cells": self.counts.get("red_cells"),
            "platelets": self.counts.get("platelets"),
            "red_cells_per_field": _per_field(self.counts.get("red_cells"), self.fields),
            "platelets_per_field": _per_field(self.counts.get("platelets"), self.fields),
            "second_opinion_budget": self.max_second_opinions,
            "known_limits": KNOWN_LIMITS,
        }

    def tool(self, name: str, args: dict):
        cells = list(self.cells.values())
        if name == "get_slide_summary":
            return self.summary()
        if name == "get_reference_ranges":
            return {
                "unit": "percent of white cells",
                "note": "indicative adult ranges; the lab's own ranges prevail",
                "ranges": {t: list(r) for t, r in REFERENCE_RANGES.items()},
            }
        if name == "list_cells":
            picked = sorted(
                (
                    c
                    for c in cells
                    if (not args.get("cell_type") or c.label == args["cell_type"])
                    and (args.get("max_confidence") is None or c.confidence <= args["max_confidence"])
                    and (not args.get("needs_review_only") or c.needs_review)
                ),
                key=lambda c: c.confidence,
            )
            return {
                "matching": len(picked),
                "cells": [
                    {"cell_id": c.id, "field": c.field, "type": c.label, "confidence": c.confidence,
                     "needs_review": c.needs_review, "uncertain_detection": c.uncertain_detection}
                    for c in picked[:MAX_LISTED]
                ],
            }
        if name == "ask_second_opinion":
            cell = self.cells.get(_int(args.get("cell_id")))
            if not cell:
                return {"error": "unknown cell_id"}
            if len(self.review.second_opinions) >= self.max_second_opinions:
                return {"error": f"second opinion budget reached ({self.max_second_opinions} per slide)"}
            opinion = self.second_opinion(cell)
            label = opinion.get("label", "unparsed")
            self.review.second_opinions.append(
                {"cell_id": cell.id, "field": cell.field, "box": cell.box, "classifier": cell.label,
                 "confidence": cell.confidence, "second_opinion": label, "agrees": label == cell.label,
                 "reasoning": opinion.get("reasoning", "")}
            )
            return {"cell_id": cell.id, "classifier": cell.label, **opinion}
        if name == "flag_cell":
            cell = self.cells.get(_int(args.get("cell_id")))
            if not cell:
                return {"error": "unknown cell_id"}
            banned = check_report(str(args.get("reason", "")))
            if banned:
                self.review.rejected_flags += 1
                return {"flagged": False, "reason": f"remove diagnostic or treatment wording: {', '.join(banned)}"}
            if any(f["cell_id"] == cell.id for f in self.review.flags):
                return {"flagged": cell.id, "note": "already on the review list"}
            self.review.flags.append(
                {"cell_id": cell.id, "field": cell.field, "box": cell.box, "type": cell.label,
                 "confidence": cell.confidence, "priority": args.get("priority", "normal"),
                 "reason": args.get("reason", "")}
            )
            return {"flagged": cell.id, "review_list_size": len(self.review.flags)}
        if name == "submit_report":
            report = str(args.get("report", ""))
            banned = check_report(report)
            if banned:
                self.review.rejected_reports += 1
                return {"accepted": False, "reason": f"remove diagnostic or treatment wording: {', '.join(banned)}"}
            self.review.report = report
            return {"accepted": True}
        return {"error": f"unknown tool {name}"}

    def chat(self, messages: list[dict]) -> dict:
        response = self.chat_fn(messages, TOOLS)
        usage = response.get("usage") or {}
        self.review.llm_calls += 1
        self.review.prompt_tokens += usage.get("prompt_tokens", 0)
        self.review.completion_tokens += usage.get("completion_tokens", 0)
        return response["choices"][0]["message"]

    def run(self) -> Review:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "A new slide is ready for review. Prepare the review."},
        ]
        for _ in range(self.max_steps):
            message = self.chat(messages)
            calls = message.get("tool_calls") or []
            reply = {"role": "assistant", "content": message.get("content") or ""}
            messages.append({**reply, "tool_calls": calls} if calls else reply)
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
                result = self.tool(name, args if isinstance(args, dict) else {})
                self.review.steps.append({"tool": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
            if self.review.report:
                break
        return self.review


def _per_field(total: int | None, fields: int) -> float | None:
    return round(total / fields, 1) if total is not None and fields else None


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _open(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.open(image).convert("RGB")


def _muse_vision(image: bytes, prompt: str) -> tuple[str, int]:
    from smearlens.muse import ask_image

    return ask_image(image, prompt)


def review_slide(
    reports: Sequence[dict],
    images: Sequence[ImageLike],
    names: Sequence[str] | None = None,
    chat: Chat = baseten_chat,
    vision: Vision = _muse_vision,
    max_steps: int = 16,
    max_second_opinions: int = MAX_SECOND_OPINIONS,
) -> dict:
    """Review one slide: `reports[i]` is the backend output (smearlens.analyze.report) for `images[i]`."""
    from smearlens.muse import parse_label

    if len(images) != len(reports):
        raise ValueError("one image per report")
    names = list(names) if names else [f"field-{i:03d}" for i in range(len(reports))]
    by_name = dict(zip(names, images))
    cells = cells_from_reports(reports, names)
    counts = {
        "red_cells": sum(r.get("red_cell_count", 0) for r in reports),
        "platelets": sum(r.get("platelet_count", 0) for r in reports),
        "fields_without_white_cell": sum(not r.get("white_cells") for r in reports),
    }
    started = time.perf_counter()
    reviewer: SlideReviewer

    def second_opinion(cell: Cell) -> dict:
        text, tokens = vision(crop(_open(by_name[cell.field]), Box("WBC", *cell.box)), PROMPT)
        reviewer.review.vision_tokens += tokens
        return {"label": parse_label(text, CANDIDATE_CLASSES), "reasoning": text[-400:]}

    reviewer = SlideReviewer(cells, len(reports), second_opinion, chat, counts, max_steps, max_second_opinions)
    review = reviewer.run()
    agent_tokens = review.prompt_tokens + review.completion_tokens
    return {
        "model": MODEL,
        "report": review.report,
        "accepted": bool(review.report),
        "summary": reviewer.summary(),
        "flags": review.flags,
        "second_opinions": review.second_opinions,
        "steps": review.steps,
        "tokens": {
            "agent_prompt": review.prompt_tokens,
            "agent_completion": review.completion_tokens,
            "agent": agent_tokens,
            "second_opinions": review.vision_tokens,
            "total": agent_tokens + review.vision_tokens,
        },
        "llm_calls": review.llm_calls,
        "seconds": round(time.perf_counter() - started, 1),
        "rejected_reports": review.rejected_reports,
        "rejected_flags": review.rejected_flags,
    }
