"""Blast class experiment (note 19): a 9 type probe, the 8 PBC types plus "blast" from Munich.

    modal run --detach scripts/modal_blasts.py::features --part lmu       # Munich features, cached
    modal run --detach scripts/modal_blasts.py::features --part pbc_aug   # augmented PBC features, cached
    modal run --detach scripts/modal_blasts.py::evaluate
    modal volume get smearlens-hf-cache glimmer/blasts runs/blasts

Features are cached in the smearlens-hf-cache volume under glimmer/, so the probes can be trained
again without the GPU encoder. The Munich test split is only used for the final numbers: the weight
decay, the temperature and the blast threshold come from the PBC and Munich val splits.
"""

import io
import json
import random
from pathlib import Path

import modal

from modal_glimmer_probe import CACHE, GPU, REMOTE, REPO, cache, datasets, embed, image, load_encoder, train_probe

blast_image = (
    image.add_local_file(REPO / "scripts" / "modal_glimmer_probe.py", "/root/modal_glimmer_probe.py")
    .add_local_file(REPO / "scripts" / "lmu_eval.py", f"{REMOTE}/scripts/lmu_eval.py")
    .add_local_file(REPO / "scripts" / "review_coverage.py", f"{REMOTE}/scripts/review_coverage.py")
    .add_local_file(REPO / "models" / "type_probe.npz", f"{REMOTE}/models/type_probe.npz")
)
app = modal.App("smearlens-blasts", image=blast_image)
HF = modal.Secret.from_name("huggingface")
LMU_ROOT = Path("/data/lmu/PKG - AML-Cytomorphology_LMU")
OUT = Path(CACHE) / "glimmer"
BLAST_CODES = ("MYO", "MOB")
LMU_SPLIT = (0.6, 0.2)


def jpeg(img) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def lmu_features(encoder, processor, merge):
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from PIL import Image

    rows = [line.split() for line in Path("/data/lmu/annotations.dat").read_text().splitlines() if line.strip()]
    paths = [LMU_ROOT / r[0] for r in rows if (LMU_ROOT / r[0]).exists()]
    features = []
    with ThreadPoolExecutor(16) as pool:
        for i in range(0, len(paths), 1024):
            sources = list(pool.map(lambda p: jpeg(Image.open(p).convert("RGB")), paths[i : i + 1024]))
            features.append(embed(encoder, processor, merge, sources))
            print(f"lmu {i + len(sources)}/{len(paths)}", flush=True)
    np.savez_compressed(
        OUT / "lmu_features.npz",
        features=np.concatenate(features).astype(np.float16),
        paths=np.array([str(p.relative_to(LMU_ROOT)) for p in paths]),
    )


def pbc_aug_features(encoder, processor, merge, copies: int = 2):
    """The same augmented copies and simulated other lab as run_augmented (note 08), same seeds."""
    import sys

    import numpy as np
    from PIL import Image

    sys.path.insert(0, REMOTE)
    from smearlens.augment import augment
    from smearlens.pbc import PBC_CLASSES, read_splits

    data = Path(REMOTE) / "data"
    cells = read_splits(data / "pbc" / "splits.csv", data)
    clean = np.load(OUT / "pbc_features.npz")
    if list(clean["paths"]) != [c.path.name for c in cells]:
        raise RuntimeError("cached features do not match splits.csv")
    rng = random.Random(0)
    train = [c for c in cells if c.split == "train"]
    test = [c for c in cells if c.split == "test"]
    aug_sources = [jpeg(augment(Image.open(c.path).convert("RGB"), rng)) for _ in range(copies) for c in train]
    print(f"augmented {len(aug_sources)}", flush=True)
    aug = embed(encoder, processor, merge, aug_sources)
    shift_rng = random.Random(1)
    shifted = [jpeg(augment(Image.open(c.path).convert("RGB"), shift_rng, strength=1.5)) for c in test]
    np.savez_compressed(
        OUT / "pbc_aug_features.npz",
        aug_features=aug.astype(np.float16),
        aug_labels=np.array([PBC_CLASSES.index(c.label) for _ in range(copies) for c in train]),
        shifted_features=embed(encoder, processor, merge, shifted).astype(np.float16),
    )


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=3 * 3600, secrets=[HF], volumes={CACHE: cache, "/data": datasets})
def embed_part(part: str) -> str:
    encoder, processor, merge = load_encoder()
    {"lmu": lmu_features, "pbc_aug": pbc_aug_features}[part](encoder, processor, merge)
    cache.commit()
    return part


def lmu_split(codes, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    split = np.empty(len(codes), dtype=object)
    for code in sorted(set(codes)):
        idx = rng.permutation(np.flatnonzero(codes == code))
        a, b = int(round(LMU_SPLIT[0] * len(idx))), int(round(sum(LMU_SPLIT) * len(idx)))
        split[idx[:a]], split[idx[a:b]], split[idx[b:]] = "train", "val", "test"
    return split


@app.function(gpu="L4", cpu=8, memory=32768, timeout=3600, volumes={CACHE: cache})
def run_evaluate() -> str:
    import sys
    from collections import Counter

    import numpy as np
    import torch

    sys.path.insert(0, REMOTE)
    sys.path.insert(0, f"{REMOTE}/scripts")
    from lmu_eval import LMU_TO_PBC, UNSEEN
    from review_coverage import coverage_curve, ece, fit_temperature, softmax
    from smearlens.pbc import PBC_CLASSES

    classes9 = list(PBC_CLASSES) + ["blast"]
    blast = classes9.index("blast")
    clean = np.load(OUT / "pbc_features.npz")
    aug = np.load(OUT / "pbc_aug_features.npz")
    lmu = np.load(OUT / "lmu_features.npz")
    f32 = lambda a: a.astype(np.float32)
    pbc_x, pbc_y, pbc_s = f32(clean["features"]), clean["labels"], clean["splits"]
    aug_x, aug_y, shift_x = f32(aug["aug_features"]), aug["aug_labels"], f32(aug["shifted_features"])
    lmu_x = f32(lmu["features"])
    codes = np.array([p.split("/")[0] for p in lmu["paths"]])
    lmu_s = lmu_split(codes)
    lmu_y = np.array([PBC_CLASSES.index(LMU_TO_PBC[c]) if c in LMU_TO_PBC else blast if c in BLAST_CODES else -1 for c in codes])
    known, blasts = lmu_y >= 0, np.isin(codes, BLAST_CODES)
    pick = lambda s: pbc_s == s
    lm = lambda s, extra=True: (lmu_s == s) & extra

    def train(x, y, xv, yv, n):
        (val_acc, decay, head), mean, std = train_probe(x, y, xv, yv, n)
        return {
            "weight": head.weight.detach().cpu().numpy(),
            "bias": head.bias.detach().cpu().numpy(),
            "mean": mean,
            "std": std,
            "classes": np.array(classes9[:n]),
            "weight_decay": decay,
            "val_accuracy": round(val_acc, 4),
        }

    base_x = np.concatenate([pbc_x[pick("train")], aug_x])
    base_y = np.concatenate([pbc_y[pick("train")], aug_y])
    lmu8_tr = lm("train", known & ~blasts)
    lmu8_va = lm("val", known & ~blasts)
    blast_tr, blast_va = lm("train", blasts), lm("val", blasts)
    pv_x, pv_y = pbc_x[pick("val")], pbc_y[pick("val")]
    cat = lambda *a: np.concatenate(a)

    current = dict(np.load(f"{REMOTE}/models/type_probe.npz"))
    models = {
        "current": current,
        "pbc8_retrained": train(base_x, base_y, pv_x, pv_y, 8),
        "adapted8": train(
            cat(base_x, lmu_x[lmu8_tr]), cat(base_y, lmu_y[lmu8_tr]), cat(pv_x, lmu_x[lmu8_va]), cat(pv_y, lmu_y[lmu8_va]), 8
        ),
        "blast9_blasts_only": train(
            cat(base_x, lmu_x[blast_tr]), cat(base_y, lmu_y[blast_tr]), cat(pv_x, lmu_x[blast_va]), cat(pv_y, lmu_y[blast_va]), 9
        ),
        "blast9": train(
            cat(base_x, lmu_x[lmu8_tr | blast_tr]),
            cat(base_y, lmu_y[lmu8_tr | blast_tr]),
            cat(pv_x, lmu_x[lmu8_va | blast_va]),
            cat(pv_y, lmu_y[lmu8_va | blast_va]),
            9,
        ),
    }

    def logits(m, x):
        return ((x - m["mean"]) / m["std"]) @ m["weight"].T + m["bias"]

    def recalls(pred, truth, names):
        return {names[c]: round(float((pred[truth == c] == c).mean()), 4) for c in sorted(set(truth.tolist()))}

    results = {}
    for name, m in models.items():
        n = len(m["classes"])
        names = [str(c) for c in m["classes"]]
        lmu_val = lm("val", known if n == 9 else known & ~blasts)
        val_logits = cat(logits(m, pv_x), logits(m, lmu_x[lmu_val])) if name != "current" else logits(m, pv_x)
        val_truth = cat(pv_y, lmu_y[lmu_val]) if name != "current" else pv_y
        t = 0.867 if name == "current" else fit_temperature(val_logits, val_truth)
        prob = lambda x: softmax(logits(m, x), t)
        p_test = prob(pbc_x[pick("test")])
        y_test = pbc_y[pick("test")]
        cutoff = float(np.quantile(p_test.max(1), 0.10))
        p_shift = prob(shift_x)
        r = {
            "classes": names,
            "weight_decay": m.get("weight_decay"),
            "val_accuracy": m.get("val_accuracy"),
            "temperature": round(t, 3),
            "review_cutoff_pbc_10pct": round(cutoff, 4),
            "pbc_test": {
                "cells": int(len(y_test)),
                "accuracy": round(float((p_test.argmax(1) == y_test).mean()), 4),
                "per_class_recall": recalls(p_test.argmax(1), y_test, names),
                "ece": round(ece(p_test.max(1), p_test.argmax(1) == y_test), 4),
                "called_blast": int((p_test.argmax(1) == blast).sum()) if n == 9 else 0,
                "coverage": coverage_curve(p_test.max(1), p_test.argmax(1) == y_test, (0, 0.05, 0.1, 0.2)),
            },
            "pbc_shifted": {
                "accuracy": round(float((p_shift.argmax(1) == y_test).mean()), 4),
                "called_blast": int((p_shift.argmax(1) == blast).sum()) if n == 9 else 0,
            },
        }
        test = lmu_s == "test"
        p = prob(lmu_x[test])
        pred, conf, y, c = p.argmax(1), p.max(1), lmu_y[test], codes[test]
        k = (y >= 0) & (y != blast)
        is_blast = np.isin(c, BLAST_CODES)
        review = conf < cutoff
        detector = {"review_list": review}
        if n == 9:
            detector["called_blast"] = pred == blast
            detector["called_blast_or_review"] = (pred == blast) | review
        r["munich_test"] = {
            "cells": int(test.sum()),
            "known_types": {
                "cells": int(k.sum()),
                "accuracy": round(float((pred[k] == y[k]).mean()), 4),
                "balanced_accuracy": round(float(np.mean(list(recalls(pred[k], y[k], names).values()))), 4),
                "per_class_recall": recalls(pred[k], y[k], names),
                "sent_to_review_percent": round(100 * float(review[k].mean()), 1),
            },
            "blast_detection": {
                d: {
                    "recall": round(float(flag[is_blast].mean()), 4),
                    "recall_myeloblast": round(float(flag[c == "MYO"].mean()), 4),
                    "recall_monoblast": round(float(flag[c == "MOB"].mean()), 4),
                    "precision": round(float(is_blast[flag].mean()), 4) if flag.any() else None,
                    "flagged_non_blast_percent": round(100 * float(flag[~is_blast].mean()), 2),
                }
                for d, flag in detector.items()
            },
            "unseen_called": {UNSEEN[u]: dict(Counter(names[i] for i in pred[c == u]).most_common(3)) for u in ("LYA", "KSC")},
        }
        if n == 9:
            truth9 = np.where(is_blast, blast, y)
            rows = (truth9 >= 0)
            confusion = np.zeros((9, 9), dtype=int)
            for a, b in zip(truth9[rows], pred[rows]):
                confusion[a, b] += 1
            r["munich_test"]["confusion"] = confusion.tolist()
            r["munich_test"]["blasts_called_as"] = dict(Counter(names[i] for i in pred[is_blast]).most_common(4))
            r["munich_test"]["coverage_all"] = coverage_curve(conf[rows], pred[rows] == truth9[rows], (0, 0.1, 0.2, 0.3))
            pb_val = prob(lmu_x[lm("val", known)])[:, blast]
            val_is_blast = np.isin(codes[lm("val", known)], BLAST_CODES)
            pbc_val_blast = prob(pv_x)[:, blast]
            sweep = []
            for thr in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
                flag = p[:, blast] >= thr
                sweep.append(
                    {
                        "threshold": thr,
                        "val_recall": round(float((pb_val[val_is_blast] >= thr).mean()), 4),
                        "val_precision": round(float(val_is_blast[pb_val >= thr].mean()), 4) if (pb_val >= thr).any() else None,
                        "pbc_val_flagged_percent": round(100 * float((pbc_val_blast >= thr).mean()), 2),
                        "test_recall": round(float(flag[is_blast].mean()), 4),
                        "test_precision": round(float(is_blast[flag].mean()), 4) if flag.any() else None,
                        "pbc_test_flagged_percent": round(100 * float((p_test[:, blast] >= thr).mean()), 2),
                        "pbc_shifted_flagged_percent": round(100 * float((p_shift[:, blast] >= thr).mean()), 2),
                    }
                )
            r["blast_threshold_sweep"] = sweep
        results[name] = r
        print(name, json.dumps({k: r[k] for k in ("pbc_test", "pbc_shifted")})[:400], flush=True)

    split_counts = {s: dict(Counter(codes[lmu_s == s])) for s in ("train", "val", "test")}
    out = OUT / "blasts"
    out.mkdir(exist_ok=True)
    for name in ("blast9", "blast9_blasts_only", "adapted8"):
        m = models[name]
        np.savez(out / f"probe_{name}.npz", **{k: m[k] for k in ("weight", "bias", "mean", "std", "classes")})
    summary = {"munich_split": split_counts, "blast_codes": list(BLAST_CODES), "models": results}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    cache.commit()
    return json.dumps(summary)


@app.local_entrypoint()
def features(part: str = "lmu"):
    print(embed_part.remote(part))


@app.local_entrypoint()
def evaluate():
    out = REPO / "runs" / "blasts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(run_evaluate.remote())
    print(f"wrote {out / 'metrics.json'}")
