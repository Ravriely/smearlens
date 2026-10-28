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

base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install("git")
    .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(TRANSFORMERS, "safetensors", "pillow", "numpy", "requests", "huggingface_hub")
    .pip_install("peft==0.21.2")
    .add_local_dir(REPO / "smearlens", f"{REMOTE}/smearlens")
)
image = (
    base_image.add_local_file(REPO / "data" / "pbc" / "splits.csv", f"{REMOTE}/data/pbc/splits.csv")
    .add_local_dir(REPO / "data" / "pbc" / "PBC_dataset_normal_DIB", f"{REMOTE}/data/pbc/PBC_dataset_normal_DIB")
)

app = modal.App("smearlens-glimmer-probe")
cache = modal.Volume.from_name("smearlens-hf-cache", create_if_missing=True)
datasets = modal.Volume.from_name("smearlens-data", create_if_missing=True)
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


def embed(encoder, processor, merge: int, sources: list, batch: int = 32):
    import io

    import numpy as np
    import torch
    from PIL import Image

    def load(source):
        return Image.open(io.BytesIO(source) if isinstance(source, bytes) else source).convert("RGB")

    out = []
    for i in range(0, len(sources), batch):
        images = [load(s) for s in sources[i : i + batch]]
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


GPU = os.environ.get("SMEARLENS_GPU", "L40S")


@app.function(gpu=GPU, image=base_image, timeout=1800, secrets=[modal.Secret.from_name("huggingface")], volumes={CACHE: cache})
def embed_crops(crops: list[bytes]):
    encoder, processor, merge = load_encoder()
    return embed(encoder, processor, merge, crops)


@app.function(
    gpu=GPU,
    image=base_image,
    timeout=3 * 3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={CACHE: cache, "/data": datasets},
)
def embed_lmu(resize: int = 0) -> bytes:
    import numpy as np
    from PIL import Image

    root = Path("/data/lmu/PKG - AML-Cytomorphology_LMU")
    rows = [line.split() for line in (Path("/data/lmu/annotations.dat")).read_text().splitlines() if line.strip()]
    paths = [root / r[0] for r in rows if (root / r[0]).exists()]

    def load(path):
        image = Image.open(path).convert("RGB")
        if resize:
            image = image.resize((resize, resize), Image.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()

    encoder, processor, merge = load_encoder()
    features = []
    for i in range(0, len(paths), 512):
        features.append(embed(encoder, processor, merge, [load(p) for p in paths[i : i + 512]]))
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        features=np.concatenate(features).astype(np.float16),
        paths=np.array([str(p.relative_to(root)) for p in paths]),
        missing=np.array([r[0] for r in rows if not (root / r[0]).exists()]),
    )
    return buffer.getvalue()


@app.function(
    gpu=GPU,
    image=base_image,
    cpu=8,
    timeout=3 * 3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={CACHE: cache, "/data": datasets},
)
def embed_lmu_zoom(zooms: list[float], per_class: int = 300) -> bytes:
    import random
    from concurrent.futures import ThreadPoolExecutor

    import sys

    import numpy as np
    from PIL import Image

    sys.path.insert(0, REMOTE)
    from smearlens.pipeline import PBC_FIELD_PX

    root = Path("/data/lmu/PKG - AML-Cytomorphology_LMU")
    rows = [line.split() for line in (Path("/data/lmu/annotations.dat")).read_text().splitlines() if line.strip()]
    groups: dict = {}
    for r in rows:
        if (root / r[0]).exists():
            groups.setdefault(r[0].split("/")[0], []).append(root / r[0])
    rng = random.Random(0)
    paths = [p for g in groups.values() for p in rng.sample(g, min(per_class, len(g)))]

    def load(path):
        return Image.open(path).convert("RGB")

    def zoomed(image, zoom):
        scaled = image.resize((round(image.width * zoom), round(image.height * zoom)), Image.LANCZOS)
        offset = (scaled.width - PBC_FIELD_PX) // 2
        pixels = np.asarray(scaled)
        border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
        field = Image.new("RGB", (PBC_FIELD_PX, PBC_FIELD_PX), tuple(int(v) for v in np.median(border, axis=0)))
        field.paste(scaled, (-offset, -offset))
        buffer = io.BytesIO()
        field.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()

    with ThreadPoolExecutor(16) as pool:
        images = list(pool.map(load, paths))
    encoder, processor, merge = load_encoder()
    out = {"paths": np.array([str(p.relative_to(root)) for p in paths]), "zooms": np.array(zooms)}
    for zoom in zooms:
        with ThreadPoolExecutor(16) as pool:
            crops = list(pool.map(lambda im: zoomed(im, zoom), images))
        out[f"features_{zoom}"] = embed(encoder, processor, merge, crops).astype(np.float16)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **out)
    return buffer.getvalue()


@app.local_entrypoint()
def lmu_zoom(zooms: str = "0.5,0.6,0.75,0.9,1.0,1.2", per_class: int = 300):
    out = REPO / "runs" / "lmu"
    out.mkdir(parents=True, exist_ok=True)
    (out / "features_zoom.npz").write_bytes(embed_lmu_zoom.remote([float(z) for z in zooms.split(",")], per_class))
    print(f"wrote {out / 'features_zoom.npz'}")


@app.local_entrypoint()
def lmu(resize: int = 0):
    out = REPO / "runs" / "lmu"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"features_resize{resize}.npz").write_bytes(embed_lmu.remote(resize))
    print(f"wrote {out / f'features_resize{resize}.npz'}")


@app.function(gpu=GPU, image=image, timeout=3600, secrets=[modal.Secret.from_name("huggingface")], volumes={CACHE: cache})
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


def evaluate(head, mean, std, features, labels, classes: int) -> dict:
    import numpy as np
    import torch

    with torch.no_grad():
        pred = head(torch.tensor((features - mean) / std, device="cuda")).argmax(1).cpu().numpy()
    confusion = np.zeros((classes, classes), dtype=int)
    for t, p in zip(labels, pred):
        confusion[t, p] += 1
    return {"accuracy": round(float((pred == labels).mean()), 4), "confusion": confusion.tolist()}


@app.function(gpu=GPU, image=image, timeout=3 * 3600, secrets=[modal.Secret.from_name("huggingface")], volumes={CACHE: cache})
def run_augmented(copies: int = 2) -> bytes:
    import random
    import sys

    import numpy as np
    from PIL import Image

    sys.path.insert(0, REMOTE)
    from smearlens.augment import augment, lab_stats, reinhard
    from smearlens.pbc import PBC_CLASSES, read_splits

    data = Path(REMOTE) / "data"
    cells = read_splits(data / "pbc" / "splits.csv", data)
    clean = np.load(Path(CACHE) / "glimmer" / "pbc_features.npz")
    if list(clean["paths"]) != [c.path.name for c in cells]:
        raise RuntimeError("cached features do not match splits.csv")
    features, labels, splits = clean["features"].astype(np.float32), clean["labels"], clean["splits"]
    encoder, processor, merge = load_encoder()
    rng = random.Random(0)

    def jpeg(image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()

    train = [c for c in cells if c.split == "train"]
    test = [c for c in cells if c.split == "test"]
    aug_sources = [jpeg(augment(Image.open(c.path).convert("RGB"), rng)) for _ in range(copies) for c in train]
    aug_features = embed(encoder, processor, merge, aug_sources)
    aug_labels = np.array([PBC_CLASSES.index(c.label) for _ in range(copies) for c in train])

    shift_rng = random.Random(1)
    shifted = [augment(Image.open(c.path).convert("RGB"), shift_rng, strength=1.5) for c in test]
    target_mean, target_std = lab_stats([Image.open(c.path) for c in train[::20]])
    shifted_features = embed(encoder, processor, merge, [jpeg(i) for i in shifted])
    normalised_features = embed(encoder, processor, merge, [jpeg(reinhard(i, target_mean, target_std)) for i in shifted])
    test_labels = labels[splits == "test"]

    n = len(PBC_CLASSES)
    pick = {s: splits == s for s in ("train", "val", "test")}
    results = {}
    heads = {}
    for name, (x, y) in {
        "clean": (features[pick["train"]], labels[pick["train"]]),
        "augmented": (
            np.concatenate([features[pick["train"]], aug_features]),
            np.concatenate([labels[pick["train"]], aug_labels]),
        ),
    }.items():
        (val_acc, decay, head), mean, std = train_probe(x, y, features[pick["val"]], labels[pick["val"]], n)
        heads[name] = (head, mean, std)
        results[name] = {
            "train_images": int(len(y)),
            "weight_decay": decay,
            "val_accuracy": round(val_acc, 4),
            "test_clean": evaluate(head, mean, std, features[pick["test"]], test_labels, n),
            "test_shifted": evaluate(head, mean, std, shifted_features, test_labels, n),
            "test_shifted_reinhard": evaluate(head, mean, std, normalised_features, test_labels, n),
        }

    out = Path("/tmp/glimmer-aug")
    out.mkdir(exist_ok=True)
    head, mean, std = heads["augmented"]
    np.savez(
        out / "probe_aug.npz",
        weight=head.weight.detach().cpu().numpy(),
        bias=head.bias.detach().cpu().numpy(),
        mean=mean,
        std=std,
        classes=np.array(PBC_CLASSES),
        lab_mean=target_mean,
        lab_std=target_std,
    )
    (out / "metrics_aug.json").write_text(json.dumps({"copies": copies, "classes": list(PBC_CLASSES), **results}, indent=2))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(out, arcname="glimmer-aug")
    return buffer.getvalue()


@app.local_entrypoint()
def augmented(copies: int = 2):
    archive = run_augmented.remote(copies)
    out = REPO / "runs"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out, filter="data")
    print((out / "glimmer-aug" / "metrics_aug.json").read_text())


@app.local_entrypoint()
def main(per_class: int = 0):
    archive = run.remote(per_class)
    out = REPO / "runs"
    out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out, filter="data")
    print((out / "glimmer" / "metrics.json").read_text())
