"""Train a linear probe on the Muse Glimmer perception encoder, on Modal.

    modal secret create huggingface HF_TOKEN=...
    modal run scripts/modal_glimmer_probe.py

Only the vision tower of Muse-Glimmer-30B is downloaded (one contiguous
3.7 GB byte range of the first shard). Features, probe weights and metrics
are copied back to runs/glimmer.
"""

import io
import json
import os
import tarfile
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent
REMOTE = "/root/smearlens"
MODEL_ID = "meta-models/Muse-Glimmer-30B"
SHARD = "model-00001-of-00002.safetensors"
TRANSFORMERS = "git+https://github.com/huggingface/transformers.git@674c0e396a06aa9d96cdc4741ce9c0853eae64b8"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install("git")
    .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(TRANSFORMERS, "safetensors", "pillow", "numpy", "requests", "huggingface_hub")
    .add_local_dir(REPO / "smearlens", f"{REMOTE}/smearlens")
    .add_local_file(REPO / "data" / "pbc" / "splits.csv", f"{REMOTE}/data/pbc/splits.csv")
    .add_local_dir(REPO / "data" / "pbc" / "PBC_dataset_normal_DIB", f"{REMOTE}/data/pbc/PBC_dataset_normal_DIB")
)

app = modal.App("smearlens-glimmer-probe", image=image)
cache = modal.Volume.from_name("smearlens-hf-cache", create_if_missing=True)
CACHE = "/root/.cache/huggingface"


def fetch_vision_weights(path: Path) -> None:
    import struct

    import requests

    url = f"https://huggingface.co/{MODEL_ID}/resolve/main/{SHARD}"
    auth = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    size = struct.unpack("<Q", requests.get(url, headers={**auth, "Range": "bytes=0-7"}).content)[0]
    header = json.loads(requests.get(url, headers={**auth, "Range": f"bytes=8-{7 + size}"}).content)
    tensors = {k: v for k, v in header.items() if k.startswith("model.vision_tower.")}
    start = min(v["data_offsets"][0] for v in tensors.values())
    end = max(v["data_offsets"][1] for v in tensors.values())
    new_header = {
        k.removeprefix("model.vision_tower."): {
            **v,
            "data_offsets": [v["data_offsets"][0] - start, v["data_offsets"][1] - start],
        }
        for k, v in tensors.items()
    }
    encoded = json.dumps(new_header).encode()
    encoded += b" " * (-len(encoded) % 8)
    base = 8 + size
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        with requests.get(url, headers={**auth, "Range": f"bytes={base + start}-{base + end - 1}"}, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(64 << 20):
                f.write(chunk)
    tmp.rename(path)


def load_encoder():
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionModel

    weights = Path(CACHE) / "glimmer-vision.safetensors"
    if not weights.exists():
        fetch_vision_weights(weights)
        cache.commit()
    config = AutoConfig.from_pretrained(MODEL_ID, token=os.environ["HF_TOKEN"])
    encoder = MuseGlimmerVisionModel._from_config(config.vision_config)
    missing, unexpected = encoder.load_state_dict(load_file(weights), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"missing={missing[:5]} unexpected={unexpected[:5]}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=os.environ["HF_TOKEN"])
    return encoder.to("cuda", torch.bfloat16).eval(), processor, config.vision_config.merge_size


def embed(encoder, processor, merge: int, paths: list, batch: int = 32):
    import numpy as np
    import torch
    from PIL import Image

    out = []
    for i in range(0, len(paths), batch):
        images = [Image.open(p).convert("RGB") for p in paths[i : i + batch]]
        inputs = processor(images=images, return_tensors="pt")
        grid = inputs["image_grid_thw"].cuda()
        with torch.no_grad():
            hidden = encoder(pixel_values=inputs["pixel_values"].cuda().to(torch.bfloat16), grid_thw=grid)
        sizes = (grid.prod(-1) // merge**2).tolist()
        out += [t.float().mean(0).cpu().numpy() for t in torch.split(hidden.last_hidden_state, sizes)]
    return np.stack(out)


def train_probe(x, y, x_val, y_val, classes: int):
    import torch

    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.tensor((x - mean) / std, device="cuda")
    yt = torch.tensor(y, device="cuda")
    xv = torch.tensor((x_val - mean) / std, device="cuda")
    yv = torch.tensor(y_val, device="cuda")
    weights = torch.bincount(yt, minlength=classes).float()
    weights = weights.sum() / (classes * weights)
    best = None
    for decay in (1e-4, 1e-3, 1e-2, 1e-1):
        torch.manual_seed(0)
        head = torch.nn.Linear(xt.shape[1], classes).cuda()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=decay)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        for _ in range(300):
            optimizer.zero_grad()
            loss_fn(head(xt), yt).backward()
            optimizer.step()
        with torch.no_grad():
            accuracy = (head(xv).argmax(1) == yv).float().mean().item()
        if best is None or accuracy > best[0]:
            best = (accuracy, decay, head)
    return best, mean, std


@app.function(
    gpu=os.environ.get("SMEARLENS_GPU", "L40S"),
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={CACHE: cache},
)
def run(per_class: int = 0) -> bytes:
    import random
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, REMOTE)
    from smearlens.pbc import PBC_CLASSES, read_splits

    data = Path(REMOTE) / "data"
    cells = read_splits(data / "pbc" / "splits.csv", data)
    if per_class:
        rng = random.Random(0)
        groups: dict = {}
        for c in cells:
            groups.setdefault((c.label, c.split), []).append(c)
        cells = [c for g in groups.values() for c in rng.sample(g, min(per_class, len(g)))]
    encoder, processor, merge = load_encoder()

    features = embed(encoder, processor, merge, [c.path for c in cells])
    labels = np.array([PBC_CLASSES.index(c.label) for c in cells])
    splits = np.array([c.split for c in cells])

    pick = {s: splits == s for s in ("train", "val", "test")}
    (val_acc, decay, head), mean, std = train_probe(
        features[pick["train"]], labels[pick["train"]], features[pick["val"]], labels[pick["val"]], len(PBC_CLASSES)
    )
    with torch.no_grad():
        logits = head(torch.tensor((features[pick["test"]] - mean) / std, device="cuda"))
    pred = logits.argmax(1).cpu().numpy()
    truth = labels[pick["test"]]
    confusion = np.zeros((len(PBC_CLASSES),) * 2, dtype=int)
    for t, p in zip(truth, pred):
        confusion[t, p] += 1

    out = Path("/tmp/glimmer")
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "pbc_features.npz",
        features=features.astype(np.float16),
        labels=labels,
        splits=splits,
        paths=np.array([c.path.name for c in cells]),
    )
    np.savez(
        out / "probe.npz",
        weight=head.weight.detach().cpu().numpy(),
        bias=head.bias.detach().cpu().numpy(),
        mean=mean,
        std=std,
        classes=np.array(PBC_CLASSES),
    )
    metrics = {
        "encoder": f"{MODEL_ID} vision tower (frozen)",
        "feature_dim": int(features.shape[1]),
        "train": int(pick["train"].sum()),
        "val": int(pick["val"].sum()),
        "test": int(pick["test"].sum()),
        "weight_decay": decay,
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(float((pred == truth).mean()), 4),
        "per_class_recall": {
            c: round(float(confusion[i, i] / confusion[i].sum()), 4) for i, c in enumerate(PBC_CLASSES)
        },
        "confusion": confusion.tolist(),
        "classes": list(PBC_CLASSES),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(out, arcname="glimmer")
    return buffer.getvalue()


@app.local_entrypoint()
def main(per_class: int = 0):
    archive = run.remote(per_class)
    out = REPO / "runs"
    out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out, filter="data")
    print((out / "glimmer" / "metrics.json").read_text())
