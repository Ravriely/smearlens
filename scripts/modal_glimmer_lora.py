"""Fine-tune the Muse Glimmer vision encoder with LoRA on PBC, then test on PBC and Munich.

    modal run --detach scripts/modal_glimmer_lora.py
    modal run scripts/modal_glimmer_lora.py --max-steps 20 --eval-per-class 20   # smoke test

LoRA on the attention projections of the last blocks, stain/colour/scale
augmentation on every training image, a linear head warm-started from the
augmented probe. The frozen encoder + probe is evaluated on the same cells
for a before/after comparison. Adapter, head and metrics go to the cache
volume under lora/ and are copied back to runs/lora.
"""

import io
import json
import tarfile
from pathlib import Path

import modal

from modal_glimmer_probe import CACHE, GPU, REMOTE, cache, datasets, image, load_encoder

REPO = Path(__file__).parent.parent
app = modal.App(
    "smearlens-glimmer-lora",
    image=image.add_local_file(REPO / "runs" / "glimmer-aug" / "probe_aug.npz", f"{REMOTE}/probe_aug.npz")
    .add_local_dir(REPO / "scripts", f"{REMOTE}/scripts")
    .add_local_file(REPO / "scripts" / "modal_glimmer_probe.py", "/root/modal_glimmer_probe.py"),
)
LMU = Path("/data/lmu/PKG - AML-Cytomorphology_LMU")


@app.function(
    gpu=GPU,
    cpu=8,
    memory=65536,
    timeout=6 * 3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={CACHE: cache, "/data": datasets},
)
def train(
    epochs: int = 2,
    batch: int = 16,
    rank: int = 16,
    last_blocks: int = 16,
    lr: float = 1e-4,
    max_steps: int = 0,
    eval_per_class: int = 300,
) -> bytes:
    import math
    import random
    import sys
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image

    sys.path.insert(0, REMOTE)
    sys.path.insert(0, f"{REMOTE}/scripts")
    from lmu_eval import LMU_TO_PBC
    from smearlens.augment import augment
    from smearlens.pbc import PBC_CLASSES, read_splits

    torch.manual_seed(0)
    data = Path(REMOTE) / "data"
    cells = read_splits(data / "pbc" / "splits.csv", data)
    split = {s: [c for c in cells if c.split == s] for s in ("train", "val", "test")}
    classes = list(PBC_CLASSES)

    encoder, processor, merge = load_encoder()
    layers = [str(i) for i in range(encoder.config.num_hidden_layers - last_blocks, encoder.config.num_hidden_layers)]
    targets = [f"layers.{i}.attn.{m}" for i in layers for m in ("q_proj", "k_proj", "v_proj", "proj")]
    model = get_peft_model(encoder, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=targets))
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()

    probe = np.load(f"{REMOTE}/probe_aug.npz")
    head = torch.nn.Linear(probe["weight"].shape[1], len(classes)).cuda()
    with torch.no_grad():
        w = torch.tensor(probe["weight"] / probe["std"])
        head.weight.copy_(w)
        head.bias.copy_(torch.tensor(probe["bias"]) - w @ torch.tensor(probe["mean"]))

    def features(images):
        inputs = processor(images=images, return_tensors="pt")
        grid = inputs["image_grid_thw"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(pixel_values=inputs["pixel_values"].cuda().to(torch.bfloat16), grid_thw=grid).last_hidden_state
        sizes = (grid.prod(-1) // merge**2).tolist()
        return torch.stack([t.float().mean(0) for t in torch.split(hidden, sizes)])

    def predict(images, adapter=True):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 32):
                if adapter:
                    f = features(images[i : i + 32])
                else:
                    with model.disable_adapter():
                        f = features(images[i : i + 32])
                out.append(torch.softmax(head_eval(f, adapter), 1).cpu().numpy())
        return np.concatenate(out)

    frozen_head = torch.nn.Linear(head.in_features, len(classes)).cuda()
    frozen_head.load_state_dict(head.state_dict())

    def head_eval(f, adapter):
        return head(f) if adapter else frozen_head(f)

    def metrics(probs, truth):
        pred = probs.argmax(1)
        recall = {classes[c]: round(float((pred[truth == c] == c).mean()), 3) for c in sorted(set(truth.tolist()))}
        return {
            "cells": int(len(truth)),
            "accuracy": round(float((pred == truth).mean()), 4),
            "balanced_accuracy": round(float(np.mean(list(recall.values()))), 4),
            "recall": recall,
        }

    rng = random.Random(0)
    pick = lambda group, n: [c for label in classes for c in rng.sample([x for x in group if x.label == label], min(n, sum(x.label == label for x in group)))]
    val = pick(split["val"], eval_per_class // 3)
    test = split["test"] if eval_per_class >= 300 else pick(split["test"], eval_per_class)
    load = lambda c: Image.open(c.path).convert("RGB")
    with ThreadPoolExecutor(16) as pool:
        val_images = list(pool.map(load, val))
        test_images = list(pool.map(load, test))
    shift_rng = random.Random(1)
    shifted_images = [augment(im, shift_rng, strength=1.5) for im in test_images]

    rows = [line.split() for line in Path("/data/lmu/annotations.dat").read_text().splitlines() if line.strip()]
    groups: dict = {}
    for r in rows:
        if (LMU / r[0]).exists():
            groups.setdefault(r[0].split("/")[0], []).append(r[0])
    lmu_rng = random.Random(0)
    lmu_paths = [p for code in sorted(groups) if code in LMU_TO_PBC or code == "MYO" for p in lmu_rng.sample(groups[code], min(eval_per_class, len(groups[code])))]
    with ThreadPoolExecutor(16) as pool:
        lmu_images = list(pool.map(lambda p: Image.open(LMU / p).convert("RGB"), lmu_paths))
    lmu_codes = np.array([p.split("/")[0] for p in lmu_paths])
    lmu_known = lmu_codes != "MYO"
    lmu_truth = np.array([classes.index(LMU_TO_PBC[c]) for c in lmu_codes[lmu_known]])

    def label_ids(group):
        return np.array([classes.index(c.label) for c in group])

    def evaluate(adapter):
        probs = predict(lmu_images, adapter)
        test_probs = predict(test_images, adapter)
        conf = test_probs.max(1)
        cutoff = float(np.quantile(conf, 0.10))
        return {
            "pbc_test": metrics(test_probs, label_ids(test)),
            "pbc_test_shifted": metrics(predict(shifted_images, adapter), label_ids(test)),
            "munich_known": metrics(probs[lmu_known], lmu_truth),
            "munich_review_cutoff_from_pbc_10pct": round(cutoff, 4),
            "munich_known_sent_to_review_percent": round(100 * float((probs[lmu_known].max(1) < cutoff).mean()), 1),
            "myeloblast_sent_to_review_percent": round(100 * float((probs[~lmu_known].max(1) < cutoff).mean()), 1),
        }

    result = {"config": {"epochs": epochs, "batch": batch, "rank": rank, "last_blocks": last_blocks, "lr": lr}}
    result["frozen"] = evaluate(adapter=False)
    print(json.dumps({"frozen": result["frozen"]}), flush=True)

    train_cells = split["train"]
    counts = np.bincount(label_ids(train_cells), minlength=len(classes))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32).cuda())
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": lr}, {"params": head.parameters(), "lr": lr * 3}], weight_decay=1e-4)
    steps_per_epoch = len(train_cells) // batch
    total = max_steps or epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    def augmented_batch(group, seed):
        local = random.Random(seed)
        return [augment(load(c), local) for c in group]

    step, history = 0, []
    pool = ThreadPoolExecutor(8)
    for epoch in range(epochs):
        order = random.Random(epoch).sample(train_cells, len(train_cells))
        batches = [order[i : i + batch] for i in range(0, steps_per_epoch * batch, batch)]
        pending = pool.submit(augmented_batch, batches[0], step)
        for b, group in enumerate(batches):
            images = pending.result()
            if b + 1 < len(batches):
                pending = pool.submit(augmented_batch, batches[b + 1], step + 1)
            model.train()
            loss = loss_fn(head(features(images)), torch.tensor(label_ids(group)).cuda())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                history.append({"step": step, "loss": round(loss.item(), 4)})
                print(json.dumps(history[-1]), flush=True)
            if step >= total:
                break
        val_probs = predict(val_images)
        history.append({"epoch": epoch + 1, "val": metrics(val_probs, label_ids(val))})
        print(json.dumps(history[-1]), flush=True)
        if step >= total:
            break

    result["history"] = history
    result["lora"] = evaluate(adapter=True)
    print(json.dumps({"lora": result["lora"]}), flush=True)

    out = Path(CACHE) / "lora"
    out.mkdir(exist_ok=True)
    model.save_pretrained(out / "adapter")
    torch.save(head.state_dict(), out / "head.pt")
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    cache.commit()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(out, arcname="lora")
    return buffer.getvalue()


@app.local_entrypoint()
def main(epochs: int = 2, rank: int = 16, last_blocks: int = 16, lr: float = 1e-4, max_steps: int = 0, eval_per_class: int = 300):
    blob = train.remote(epochs=epochs, rank=rank, last_blocks=last_blocks, lr=lr, max_steps=max_steps, eval_per_class=eval_per_class)
    target = REPO / "runs"
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(target, filter="data")
    print((target / "lora" / "metrics.json").read_text())
