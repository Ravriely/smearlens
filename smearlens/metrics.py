from dataclasses import dataclass

from smearlens.datasets import Box


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __add__(self, other: "MatchResult") -> "MatchResult":
        return MatchResult(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def match_boxes(predicted: list[Box], truth: list[Box], threshold: float = 0.5) -> MatchResult:
    pairs = sorted(
        ((iou(p, t), i, j) for i, p in enumerate(predicted) for j, t in enumerate(truth)),
        reverse=True,
    )
    used_pred, used_truth = set(), set()
    for score, i, j in pairs:
        if score < threshold:
            break
        if i in used_pred or j in used_truth:
            continue
        used_pred.add(i)
        used_truth.add(j)
    tp = len(used_pred)
    return MatchResult(tp, len(predicted) - tp, len(truth) - tp)
