"""Wide fields: white cell candidates of the whole image and of windows, then fusion rules replayed offline.

    uv run python scripts/mosaic_eval.py build --split val --out runs/wide/val
    modal run scripts/wide_wbc_eval.py --truth runs/wide/val/truth.json --out runs/wide/val/raw.json
    uv run python scripts/wide_wbc_eval.py tune runs/wide/val/truth.json runs/wide/val/raw.json
    uv run python scripts/wide_wbc_eval.py score runs/wide/test/truth.json runs/wide/test/raw.json --rule ...

The GPU pass stores every candidate with its detection probe probability, so each rule is a replay.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from modal_app import PROBES, REMOTE, image
from modal_glimmer_probe import CACHE, cache

GRIDS = ((2, 0.2), (2, 0.5), (3, 0.2), (3, 0.5))
app = modal.App("smearlens-wide-wbc", image=image.add_local_file(Path(__file__).parent / "modal_app.py",
                                                                 f"{REMOTE}/scripts/modal_app.py"))


@app.cls(gpu="L40S", timeout=1800, max_containers=1, scaledown_window=60,
         secrets=[modal.Secret.from_name("huggingface")], volumes={CACHE: cache})
class Probe:
    @modal.enter()
    def load(self) -> None:
        from modal_glimmer_probe import load_encoder
        from sam3_validate import load_model

        from smearlens.analyze import SAM_THRESHOLD, load_probes

        self.sam, self.sam_processor = load_model("sam3")
        self.sam_processor.set_confidence_threshold(SAM_THRESHOLD)
        self.encoder, self.processor, self.merge = load_encoder()
        self.probes = load_probes(*(Path(REMOTE) / "probes" / p.name for p in PROBES.values()))

    def detect(self, part) -> list[dict]:
        from PIL import Image

        from sam3_validate import segment_concept, upscale_factor

        from smearlens.analyze import PROMPTS, UPSCALE_BELOW

        factor = upscale_factor(part, UPSCALE_BELOW)
        state = self.sam_processor.set_image(part.resize((part.width * factor, part.height * factor), Image.LANCZOS))
        out = []
        for label, prompt in PROMPTS:
            boxes, _, scores = segment_concept(self.sam_processor, state, label, prompt, factor, part.size)
            out.append({"image": "field", "label": label, "boxes": [[b.x0, b.y0, b.x1, b.y1] for b in boxes],
                        "scores": scores})
        return out

    def p_wbc(self, field, boxes: list, rbc_px: float, batch: int = 32, disc: float = 0.0) -> list[float]:
        """Detection probe logits, not probabilities: many candidates of a wide field saturate at p = 1.0.

        disc > 0 blanks the crop outside a disc of disc * rbc_px around the candidate, with the field background.
        """
        import io

        import numpy as np
        from PIL import Image

        from modal_glimmer_probe import embed

        from smearlens.datasets import Box
        from smearlens.pipeline import crop_at_scale

        if not boxes:
            return []
        crops = [crop_at_scale(field, Box("WBC", *b), rbc_px) for b in boxes]
        if disc:
            pixels = np.asarray(field)
            fill = np.median(np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]]), axis=0)
            blanked = []
            for c, b in zip(crops, boxes):
                a = np.asarray(Image.open(io.BytesIO(c)).convert("RGB")).copy()
                yy, xx = np.mgrid[: a.shape[0], : a.shape[1]]
                radius = max(disc * rbc_px, 0.6 * max(b[2] - b[0], b[3] - b[1]))
                a[(xx - a.shape[1] / 2) ** 2 + (yy - a.shape[0] / 2) ** 2 > radius**2] = fill
                buffer = io.BytesIO()
                Image.fromarray(a).save(buffer, format="JPEG", quality=92)
                blanked.append(buffer.getvalue())
            crops = blanked
        features = embed(self.encoder, self.processor, self.merge, crops, batch=batch)
        probe = self.probes["detection"]
        x = (np.asarray(features, dtype=np.float32) - probe["mean"]) / probe["std"]
        return [round(float(v), 4) for v in x @ probe["weight"] + probe["bias"]]

    @modal.method()
    def features(self, image_bytes: bytes) -> dict:
        """Whole-image white cell candidates and their Glimmer features, to train a probe for wide fields."""
        import io
        import statistics

        import numpy as np
        from PIL import Image

        from modal_glimmer_probe import embed

        from smearlens.analyze import RBC_MIN, WBC_MIN, candidates_for, rbc_sizes, tiles_per_side
        from smearlens.pipeline import PBC_RBC_PX, crop_at_scale

        field = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        whole = self.detect(field)
        red = [max(b[2] - b[0], b[3] - b[1]) for d in whole if d["label"] == "RBC"
               for b, sc in zip(d["boxes"], d["scores"]) if sc >= RBC_MIN]
        rbc_px = rbc_sizes(whole).get("field", PBC_RBC_PX)
        cands = candidates_for(whole, WBC_MIN, RBC_MIN).get("field", [])
        crops = [crop_at_scale(field, c.box, rbc_px) for c in cands]
        features = embed(self.encoder, self.processor, self.merge, crops) if crops else np.zeros((0, 1))
        return {"size": list(field.size), "rbc_px": rbc_px,
                "tiles": tiles_per_side(field.size, statistics.median(red) if len(red) >= 3 else None),
                "boxes": [[c.box.x0, c.box.y0, c.box.x1, c.box.y1] for c in cands],
                "features": np.asarray(features, dtype=np.float16).tobytes(), "dim": int(features.shape[1])}

    @modal.method()
    def raw(self, image_bytes: bytes, windows: bool = True) -> dict:
        import io
        import statistics

        from PIL import Image

        from smearlens.analyze import RBC_MIN, WBC_MIN, candidates_for, in_core, plan_windows, rbc_sizes
        from smearlens.analyze import tiles_per_side
        from smearlens.pipeline import PBC_RBC_PX

        field = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        whole = self.detect(field)
        red = [max(b[2] - b[0], b[3] - b[1]) for d in whole if d["label"] == "RBC"
               for b, sc in zip(d["boxes"], d["scores"]) if sc >= RBC_MIN]
        tiles = tiles_per_side(field.size, statistics.median(red) if len(red) >= 3 else None)
        rbc_px = rbc_sizes(whole).get("field", PBC_RBC_PX)
        cands = candidates_for(whole, WBC_MIN, RBC_MIN).get("field", [])
        boxes = [[c.box.x0, c.box.y0, c.box.x1, c.box.y1] for c in cands]
        out = {"size": list(field.size), "tiles": tiles, "rbc_px": rbc_px,
               "whole": [{"box": b, "score": c.score, "p": p, "p_disc1": d1, "p_disc15": d15} for b, c, p, d1, d15 in
                         zip(boxes, cands, self.p_wbc(field, boxes, rbc_px), self.p_wbc(field, boxes, rbc_px, disc=1.0),
                             self.p_wbc(field, boxes, rbc_px, disc=1.5))],
               "grids": {}}
        if not windows:
            return out
        for n, margin in GRIDS:
            rows, red_boxes = [], []
            for k, (window, core) in enumerate(plan_windows(field.size, n, margin)):
                x0, y0 = int(window.x0), int(window.y0)
                part = field.crop((x0, y0, int(round(window.x1)), int(round(window.y1))))
                found = self.detect(part)
                for d in found:
                    if d["label"] == "RBC":
                        red_boxes += [[b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0] for b, s in zip(d["boxes"], d["scores"])
                                      if in_core([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], core, field.size)
                                      and s >= RBC_MIN]
                for c in candidates_for(found, WBC_MIN, RBC_MIN).get("field", []):
                    b = [c.box.x0 + x0, c.box.y0 + y0, c.box.x1 + x0, c.box.y1 + y0]
                    rows.append({"box": b, "score": c.score, "window": k, "core": in_core(b, core, field.size),
                                 "window_box": [window.x0, window.y0, window.x1, window.y1]})
            tile_px = statistics.median([max(b[2] - b[0], b[3] - b[1]) for b in red_boxes]) if red_boxes else rbc_px
            p_tile = self.p_wbc(field, [r["box"] for r in rows], tile_px)
            p_whole = self.p_wbc(field, [r["box"] for r in rows], rbc_px)
            for r, a, b in zip(rows, p_tile, p_whole):
                r["p"], r["p_whole_scale"] = a, b
            out["grids"][f"{n}:{margin}"] = {"tile_px": tile_px, "candidates": rows}
        return out


@app.local_entrypoint()
def main(truth: str, out: str, singles: bool = False, features: bool = False) -> None:
    rows = [r for r in json.loads(Path(truth).read_text()) if r["kind"] == "mosaic" or singles]
    probe = Probe()
    started = time.perf_counter()
    if features:
        import numpy as np

        results = list(probe.features.map([Path(r["path"]).read_bytes() for r in rows]))
        arrays = {"x": np.concatenate([np.frombuffer(r["features"], dtype=np.float16).reshape(len(r["boxes"]), -1)
                                       for r in results if r["boxes"]]),
                  "owner": np.array([i for i, r in enumerate(results) for _ in r["boxes"]])}
        meta = [{"path": row["path"], **{k: v for k, v in r.items() if k not in ("features", "dim")}}
                for row, r in zip(rows, results)]
        np.savez(out, **arrays)
        Path(out).with_suffix(".json").write_text(json.dumps(meta))
        print(f"{len(rows)} images, {len(arrays['x'])} candidates in {time.perf_counter() - started:.0f} s, wrote {out}")
        return
    results = []
    for r in rows:
        results.append({"path": r["path"], **probe.raw.remote(Path(r["path"]).read_bytes(),
                                                              windows=r["kind"] == "mosaic")})
        print(r["path"], results[-1]["tiles"], len(results[-1]["whole"]), flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results))
    print(f"{len(results)} images in {time.perf_counter() - started:.0f} s, wrote {out}")


def kept(rows: list[dict], threshold: float, rbc_px: float, key: str = "p") -> list[list[float]]:
    from smearlens.analyze import merge_kept
    from smearlens.datasets import Box
    from smearlens.pipeline import Candidate

    cands = [Candidate(Box("WBC", *r["box"]), 0.0) for r in rows]
    boxes = merge_kept(cands, [r[key] for r in rows], threshold, rbc_px)
    return [[b.x0, b.y0, b.x1, b.y1] for b in boxes]


def agrees(box: list[float], others: list[dict], rbc_px: float) -> list[dict]:
    from smearlens.datasets import Box
    from smearlens.metrics import center_distance, contains_center

    a = Box("WBC", *box)
    return [o for o in others if contains_center(Box("WBC", *o["box"]), a) or contains_center(a, Box("WBC", *o["box"]))
            or center_distance(a, Box("WBC", *o["box"])) < rbc_px / 2]


WHOLE_LOGIT = math.log(0.9995 / 0.0005)
LOGITS = (WHOLE_LOGIT, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0)
DISC_LOGITS = (-4.0, -2.0, 0.0, 2.0, 4.0, WHOLE_LOGIT, 10.0, 14.0)


def predict(raw: dict, rule: str, params: dict) -> list[list[float]]:
    """Thresholds are detection probe logits; the probe of note 09 keeps p >= 0.9995, a logit of 7.6."""
    whole, rbc_px = raw["whole"], raw["rbc_px"]
    t_whole = params.get("t_whole", WHOLE_LOGIT)
    if rule == "whole":
        return kept(whole, t_whole, rbc_px)
    if rule == "disc":
        return kept([w for w in whole if w[params["key"]] >= params["t_disc"]], t_whole, rbc_px)
    grid = raw["grids"][params["grid"]]
    key = params.get("key", "p")
    px = grid["tile_px"] if key == "p" else rbc_px
    rows = [r for r in grid["candidates"] if r["core"] or not params.get("core", True)]
    if rule == "windows":
        return kept(rows, params["t_win"], px, key)
    sure = [r for r in rows if r[key] >= params["t_win"]]
    if rule == "and":
        return kept([w for w in whole if w["p"] >= t_whole and agrees(w["box"], sure, px)], -1e9, px)
    if rule == "or":
        voted = [w for w in whole if w["p"] >= t_whole] + [r for r in sure if not agrees(r["box"], whole, px)]
        return kept(voted, -1e9, px)
    raise ValueError(rule)


def score(truth: list[dict], raws: list[dict], rule: str, params: dict) -> dict:
    from mosaic_eval import off_seams
    from smearlens.metrics import MatchResult, match_centers

    by_path = {r["path"]: r for r in raws}
    out = {}
    for source in sorted({t["source"] for t in truth}) + ["all"]:
        m = MatchResult(0, 0, 0)
        part = [t for t in truth if t["kind"] == "mosaic" and source in ("all", t["source"])]
        for t in part:
            raw = by_path[t["path"]]
            size = {"width": raw["size"][0], "height": raw["size"][1]}
            m += match_centers(off_seams(predict(raw, rule, params), size, "mosaic"),
                               off_seams(t["wbc_boxes"], size, "mosaic"))
        out[source] = {"images": len(part), "white_cells_truth": m.tp + m.fn, "found": m.tp, "false": m.fp,
                       "f1": round(m.f1, 4), "precision": round(m.precision, 4), "recall": round(m.recall, 4)}
    return out


def candidate_rules(raws: list[dict]) -> list[tuple[str, dict]]:
    rules = [("whole", {"t_whole": t}) for t in LOGITS]
    discs = [k for k in ("p_disc1", "p_disc15") if all(k in w for r in raws for w in r["whole"])]
    rules += [("disc", {"key": k, "t_disc": t}) for k in discs for t in DISC_LOGITS]
    for grid in ("2:0.2", "2:0.5", "3:0.2", "3:0.5"):
        for key in ("p", "p_whole_scale"):
            for t in LOGITS:
                rules.append(("windows", {"grid": grid, "key": key, "t_win": t}))
                rules.append(("windows", {"grid": grid, "key": key, "t_win": t, "core": False}))
                rules.append(("and", {"grid": grid, "key": key, "t_win": t}))
                rules.append(("or", {"grid": grid, "key": key, "t_win": t}))
    return rules


def load_split(folder: Path) -> tuple[list[dict], list[dict], "np.ndarray", "np.ndarray"]:
    import numpy as np

    from smearlens.datasets import Box
    from smearlens.metrics import contains_center

    truth = {t["path"]: t for t in json.loads((folder / "truth.json").read_text())}
    meta = json.loads((folder / "features.json").read_text())
    stored = np.load(folder / "features.npz")
    x, labels = stored["x"].astype(np.float32), []
    for m in meta:
        boxes = [Box("WBC", *b) for b in truth[m["path"]]["wbc_boxes"]]
        labels += [int(any(contains_center(t, Box("WBC", *b)) for t in boxes)) for b in m["boxes"]]
    return [truth[m["path"]] for m in meta], meta, x, np.array(labels, dtype=np.float32)


def with_logits(meta: list[dict], logits: "np.ndarray") -> list[dict]:
    raws, i = [], 0
    for m in meta:
        n = len(m["boxes"])
        raws.append({"path": m["path"], "size": m["size"], "rbc_px": m["rbc_px"], "grids": {},
                     "whole": [{"box": b, "p": float(v)} for b, v in zip(m["boxes"], logits[i: i + n])]})
        i += n
    return raws


WIDE_LOGITS = (-2.0, 0.0, 2.0, 4.0, 6.0, WHOLE_LOGIT, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 18.0, 20.0)


def train_wide(folder: Path, out: Path, old_probe: Path) -> dict:
    """A detection probe for wide fields: the same logistic probe as note 09, trained on mosaic candidates."""
    import numpy as np

    from detection_probe import fit_logistic

    splits = {s: load_split(folder / s) for s in ("train", "val", "test")}
    _, _, x_train, y_train = splits["train"]
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    best = None
    for decay, steps in ((d, n) for d in (1e-4, 1e-3, 1e-2, 1e-1) for n in (400, 2000)):
        w, b = fit_logistic((x_train - mean) / std, y_train, decay, steps=steps)
        truth, meta, x, _ = splits["val"]
        raws = with_logits(meta, (x - mean) / std @ w + b)
        for t in WIDE_LOGITS:
            f1 = score(truth, raws, "whole", {"t_whole": t})["all"]["f1"]
            if best is None or f1 > best[0]:
                best = (f1, decay, steps, t, w, b)
    f1, decay, steps, t, w, b = best
    old = dict(np.load(old_probe))
    result = {"weight_decay": decay, "steps": steps, "threshold_logit": t, "candidates": {}, "scores": {},
              "old_probe_scores": {}}
    for name, (truth, meta, x, y) in splits.items():
        raws = with_logits(meta, (x - mean) / std @ w + b)
        result["candidates"][name] = {"all": int(len(y)), "on_a_white_cell": int(y.sum())}
        result["scores"][name] = score(truth, raws, "whole", {"t_whole": t})
        old_raws = with_logits(meta, (x - old["mean"]) / old["std"] @ old["weight"] + old["bias"])
        result["old_probe_scores"][name] = score(truth, old_raws, "whole", {"t_whole": WHOLE_LOGIT})
    threshold = 1 / (1 + math.exp(-t))
    np.savez(out, weight=w.astype(np.float32), bias=np.float32(b), mean=mean, std=std, threshold=threshold)
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("tune", "score"):
        p = sub.add_parser(name)
        p.add_argument("truth", type=Path)
        p.add_argument("raw", type=Path)
        p.add_argument("--out", type=Path)
    w = sub.add_parser("train-wide")
    w.add_argument("folder", type=Path, help="holds train/, val/ and test/ with truth.json and features.npz")
    w.add_argument("--probe", type=Path, required=True)
    w.add_argument("--out", type=Path)
    w.add_argument("--old-probe", type=Path, default=Path("models/detection_probe.npz"))
    sub.choices["score"].add_argument("--rule", action="append", required=True,
                                      help='a JSON pair, e.g. \'["and", {"grid": "2:0.2", "t_win": 14}]\'')
    args = parser.parse_args()
    if args.command == "train-wide":
        result = train_wide(args.folder, args.probe, args.old_probe)
        text = json.dumps(result, indent=1)
        print(text)
        if args.out:
            args.out.write_text(text + "\n")
        return
    truth, raws = json.loads(args.truth.read_text()), json.loads(args.raw.read_text())
    rules = candidate_rules(raws) if args.command == "tune" else [tuple(json.loads(r)) for r in args.rule]
    results = [{"rule": rule, "params": params, "scores": score(truth, raws, rule, params)} for rule, params in rules]
    if args.command == "tune":
        results.sort(key=lambda r: -r["scores"]["all"]["f1"])
    for r in results[:25]:
        s = r["scores"]["all"]
        by = " ".join(f"{k} {v['f1']:.3f}" for k, v in r["scores"].items() if k != "all")
        print(f"{r['rule']:8} {json.dumps(r['params']):70} F1 {s['f1']:.3f} P {s['precision']:.3f} "
              f"R {s['recall']:.3f} | {by}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    cli()
