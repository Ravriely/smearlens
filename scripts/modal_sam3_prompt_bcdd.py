"""SAM 3 white-cell prompt sweep per source, to find candidates on BCDD.

    modal run scripts/modal_sam3_prompt_bcdd.py

For each prompt, on the train and val images of every source: share of true
white cells with a box whose centre falls inside them (candidate recall), and
boxes per image, at several score cutoffs. Images under 512 px are upscaled.
"""

import json
from pathlib import Path

import modal

from modal_sam3 import REMOTE, hf_cache, image

app = modal.App("smearlens-sam3-prompts", image=image.add_local_file(Path(__file__).parent / "modal_sam3.py", "/root/modal_sam3.py"))
PROMPTS = [
    "cell with a large purple nucleus",
    "white blood cell",
    "leukocyte",
    "cell with a blue nucleus",
    "large cell with a dark nucleus",
    "nucleated cell",
    "lymphocyte",
    "neutrophil",
]
CUTOFFS = (0.05, 0.1, 0.2, 0.3)


@app.function(gpu="L40S", timeout=3600, secrets=[modal.Secret.from_name("huggingface")], volumes={"/root/.cache/huggingface": hf_cache})
def sweep(prompts: list[str]) -> str:
    import sys
    from collections import defaultdict

    from PIL import Image

    sys.path[:0] = [REMOTE, f"{REMOTE}/scripts"]
    from sam3_validate import load_model, segment_concept, upscale_factor
    from smearlens.datasets import iter_txl_pbc
    from smearlens.metrics import contains_center

    model, processor = load_model("sam3")
    processor.set_confidence_threshold(min(CUTOFFS))
    stats = defaultdict(lambda: defaultdict(float))
    for split in ("train", "val"):
        for sample in iter_txl_pbc(Path(REMOTE) / "data", split):
            image = Image.open(sample.image_path).convert("RGB")
            factor = upscale_factor(image, 512)
            state = processor.set_image(image.resize((image.width * factor, image.height * factor), Image.LANCZOS))
            truth = [b for b in sample.boxes if b.label == "WBC"]
            for prompt in prompts:
                boxes, _, scores = segment_concept(processor, state, "WBC", prompt, factor, image.size)
                for cutoff in CUTOFFS:
                    kept = [b for b, s in zip(boxes, scores) if s >= cutoff]
                    row = stats[f"{sample.source}|{prompt}|{cutoff}"]
                    row["images"] += 1
                    row["wbc"] += len(truth)
                    row["found"] += sum(any(contains_center(t, b) for b in kept) for t in truth)
                    row["boxes"] += len(kept)
    out = []
    for key, row in sorted(stats.items()):
        source, prompt, cutoff = key.split("|")
        out.append(
            {
                "source": source,
                "prompt": prompt,
                "cutoff": float(cutoff),
                "candidate_recall": round(row["found"] / max(1, row["wbc"]), 3),
                "boxes_per_image": round(row["boxes"] / row["images"], 2),
            }
        )
    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main():
    out = Path(__file__).parent.parent / "runs" / "sam3-prompts-by-source.json"
    out.write_text(sweep.remote(PROMPTS))
    for row in json.loads(out.read_text()):
        if row["cutoff"] in (0.1, 0.3):
            print(row)
