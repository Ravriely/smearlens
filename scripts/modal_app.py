"""SmearLens inference on Modal: SAM 3 candidates, Glimmer features, two linear probes.

    modal run scripts/modal_app.py --image field.png [--image other.jpg ...] [--out runs/analyze.json]
"""

import argparse
import json
import os
import time
from pathlib import Path

import modal

from modal_glimmer_probe import CACHE, TRANSFORMERS, cache

REPO = Path(__file__).parent.parent
REMOTE = "/root/smearlens"
PROBES = {
    "detection": REPO / "models" / "detection_probe.npz",
    "types": REPO / "models" / "type_probe.npz",
    "coverage": REPO / "docs" / "experiments" / "10-review-coverage.json",
    "lmu_eval": REPO / "docs" / "experiments" / "11-lmu-external.json",
}

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "git+https://github.com/facebookresearch/sam3.git",
        "opencv-python-headless",
        "openpyxl",
        "pillow",
        "einops",
        "pycocotools",
        "psutil",
        "scipy",
    )
    .pip_install(TRANSFORMERS, "safetensors", "numpy", "requests", "huggingface_hub")
    .env({"PYTHONPATH": f"{REMOTE}:{REMOTE}/scripts"})
    .add_local_dir(REPO / "smearlens", f"{REMOTE}/smearlens")
    .add_local_file(REPO / "scripts" / "sam3_validate.py", f"{REMOTE}/scripts/sam3_validate.py")
    .add_local_file(REPO / "scripts" / "modal_glimmer_probe.py", f"{REMOTE}/scripts/modal_glimmer_probe.py")
)
for name, path in PROBES.items():
    image = image.add_local_file(path, f"{REMOTE}/probes/{path.name}")

app = modal.App("smearlens-app", image=image)


@app.cls(
    gpu=os.environ.get("SMEARLENS_GPU", "L40S"),
    timeout=600,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={CACHE: cache},
)
class Analyzer:
    @modal.enter()
    def load(self) -> None:
        from modal_glimmer_probe import load_encoder
        from sam3_validate import load_model

        from smearlens.analyze import SAM_THRESHOLD, load_probes

        started = time.perf_counter()
        self.sam, self.sam_processor = load_model("sam3")
        self.sam_processor.set_confidence_threshold(SAM_THRESHOLD)
        self.encoder, self.processor, self.merge = load_encoder()
        self.probes = load_probes(*(Path(REMOTE) / "probes" / p.name for p in PROBES.values()))
        self.load_s = time.perf_counter() - started

    @modal.method()
    def analyze(self, image_bytes: bytes) -> dict:
        import io

        from PIL import Image

        from modal_glimmer_probe import embed
        from sam3_validate import segment_concept, upscale_factor

        from smearlens.analyze import PROMPTS, RBC_MIN, UPSCALE_BELOW, WBC_MIN
        from smearlens.analyze import candidates_for, rbc_sizes, report, white_cells
        from smearlens.pipeline import PBC_RBC_PX, crop_at_scale

        started = time.perf_counter()
        field = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        factor = upscale_factor(field, UPSCALE_BELOW)
        state = self.sam_processor.set_image(field.resize((field.width * factor, field.height * factor), Image.LANCZOS))
        detections = []
        for label, prompt in PROMPTS:
            boxes, _, scores = segment_concept(self.sam_processor, state, label, prompt, factor, field.size)
            boxes = [[b.x0, b.y0, b.x1, b.y1] for b in boxes]
            detections.append({"image": "field", "label": label, "boxes": boxes, "scores": scores})
        cands = candidates_for(detections, WBC_MIN, RBC_MIN).get("field", [])
        rbc_px = rbc_sizes(detections).get("field", PBC_RBC_PX)
        sam_done = time.perf_counter()
        crops = [crop_at_scale(field, c.box, rbc_px) for c in cands]
        features = embed(self.encoder, self.processor, self.merge, crops) if crops else None
        cells = white_cells(cands, features, rbc_px, self.probes) if crops else []
        done = time.perf_counter()
        timing = {"sam": sam_done - started, "glimmer_and_probes": done - sam_done, "total": done - started}
        result = report(field.size, detections, rbc_px, cells, timing)
        result["candidates"] = len(cands)
        result["timing_s"]["model_load"] = round(self.load_s, 3)
        return result


@app.local_entrypoint()
def main(*arglist: str) -> None:
    parser = argparse.ArgumentParser(prog="modal run scripts/modal_app.py")
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(arglist)
    analyzer = Analyzer()
    results = []
    for path in args.image:
        started = time.perf_counter()
        result = analyzer.analyze.remote(path.read_bytes())
        results.append({"path": str(path), "round_trip_s": round(time.perf_counter() - started, 3), **result})
        print(json.dumps(results[-1]), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
