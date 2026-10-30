"""SmearLens inference on Modal: SAM 3 candidates, Glimmer features, two linear probes.

    modal run scripts/modal_app.py --image field.png [--image other.jpg ...] [--out runs/analyze.json]
    modal deploy scripts/modal_app.py

The deployed app also serves the web demo (web/), POST /api/analyze and POST /api/review from a
small CPU container. /api/review runs the review agent (Muse Spark on Baseten) on the output of
/api/analyze, in its own CPU function that holds the Baseten key. That endpoint checks the token of the Modal secret smearlens-web, then the size of
the image, before it calls the GPU class, so a refused request never starts a GPU.

The deployed app scales to zero. To keep one container warm, see scripts/modal_warm.py,
or deploy with SMEARLENS_WARM=1.
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

MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
TOKEN_HEADER = "x-smearlens-token"
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi==0.115.12", "pillow")
    .env({"PYTHONPATH": "/root/scripts"})
    .add_local_file(REPO / "scripts" / "modal_glimmer_probe.py", "/root/scripts/modal_glimmer_probe.py")
    .add_local_dir(REPO / "web", "/root/web")
)
review_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pillow", "numpy")
    .env({"PYTHONPATH": "/root:/root/scripts"})
    .add_local_file(REPO / "scripts" / "modal_glimmer_probe.py", "/root/scripts/modal_glimmer_probe.py")
    .add_local_dir(REPO / "smearlens", "/root/smearlens")
)
REVIEW_MAX_STEPS = 10
REVIEW_SECOND_OPINIONS = 3


@app.cls(
    gpu=os.environ.get("SMEARLENS_GPU", "L40S"),
    timeout=600,
    scaledown_window=15 * 60,
    max_containers=1,
    min_containers=1 if os.environ.get("SMEARLENS_WARM") == "1" else 0,
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

    def detect(self, image) -> list[dict]:
        from PIL import Image

        from sam3_validate import segment_concept, upscale_factor

        from smearlens.analyze import PROMPTS, UPSCALE_BELOW

        factor = upscale_factor(image, UPSCALE_BELOW)
        state = self.sam_processor.set_image(image.resize((image.width * factor, image.height * factor), Image.LANCZOS))
        detections = []
        for label, prompt in PROMPTS:
            boxes, _, scores = segment_concept(self.sam_processor, state, label, prompt, factor, image.size)
            boxes = [[b.x0, b.y0, b.x1, b.y1] for b in boxes]
            detections.append({"image": "field", "label": label, "boxes": boxes, "scores": scores})
        return detections

    def detect_tiled(self, field, n: int) -> list[dict]:
        from smearlens.analyze import PROMPTS, in_core, plan_windows

        merged = {label: {"image": "field", "label": label, "boxes": [], "scores": []} for label, _ in PROMPTS}
        for window, core in plan_windows(field.size, n):
            x0, y0 = int(window.x0), int(window.y0)
            part = field.crop((x0, y0, int(round(window.x1)), int(round(window.y1))))
            for d in self.detect(part):
                for b, score in zip(d["boxes"], d["scores"]):
                    box = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
                    if in_core(box, core, field.size):
                        merged[d["label"]]["boxes"].append(box)
                        merged[d["label"]]["scores"].append(score)
        return [merged[label] for label in dict.fromkeys(label for label, _ in PROMPTS)]

    @modal.method()
    def analyze(self, image_bytes: bytes, tiling: bool = True) -> dict:
        import io
        import statistics

        from PIL import Image

        from modal_glimmer_probe import embed

        from smearlens.analyze import RBC_MIN, WBC_MIN
        from smearlens.analyze import candidates_for, rbc_sizes, report, tiles_per_side, wbc_probability, white_cells
        from smearlens.pipeline import PBC_RBC_PX, crop_at_scale

        started = time.perf_counter()
        field = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        detections = self.detect(field)
        red = [max(b[2] - b[0], b[3] - b[1]) for d in detections if d["label"] == "RBC"
               for b, sc in zip(d["boxes"], d["scores"]) if sc >= RBC_MIN]
        tiles = tiles_per_side(field.size, statistics.median(red) if len(red) >= 3 else None) if tiling else 1
        counted = self.detect_tiled(field, tiles) if tiles > 1 else detections
        cands = candidates_for(detections, WBC_MIN, RBC_MIN).get("field", [])
        rbc_px = rbc_sizes(detections).get("field", PBC_RBC_PX)
        count_px = rbc_sizes(counted).get("field", PBC_RBC_PX)
        sam_done = time.perf_counter()
        crops = [crop_at_scale(field, c.box, rbc_px) for c in cands]
        features = embed(self.encoder, self.processor, self.merge, crops) if crops else None
        cells = white_cells(cands, features, rbc_px, self.probes) if crops else []
        done = time.perf_counter()
        timing = {"sam": sam_done - started, "glimmer_and_probes": done - sam_done, "total": done - started}
        result = report(field.size, counted, count_px, cells, timing)
        result["candidates"] = len(cands)
        result["tiles_per_side"] = tiles
        p_wbc = wbc_probability(features, self.probes["detection"]) if crops else []
        result["candidate_boxes"] = [
            {"box": [round(v, 1) for v in (c.box.x0, c.box.y0, c.box.x1, c.box.y1)], "p_wbc": round(float(p), 5)}
            for c, p in zip(cands, p_wbc)
        ]
        result["timing_s"]["model_load"] = round(self.load_s, 3)
        result["review_cutoff"] = self.probes["review_cutoff"]
        return result


@app.function(image=review_image, secrets=[modal.Secret.from_name("baseten")], timeout=240, max_containers=2)
def review(analysis: dict, image_bytes: bytes, name: str) -> dict:
    from smearlens.agent import review_slide

    return review_slide(
        [analysis], [image_bytes], names=[name],
        max_steps=REVIEW_MAX_STEPS, max_second_opinions=REVIEW_SECOND_OPINIONS,
    )


@app.function(
    image=web_image,
    secrets=[modal.Secret.from_name("smearlens-web")],
    max_containers=1,
    scaledown_window=60,
    timeout=300,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import base64
    import hmac
    import io

    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from PIL import Image

    token = os.environ["SMEARLENS_WEB_TOKEN"].encode()
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=[TOKEN_HEADER, "content-type"]
    )

    @api.middleware("http")
    async def no_stale_page(request: Request, call_next):
        response = await call_next(request)
        response.headers["cache-control"] = "no-cache"
        return response

    def error(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse({"error": code, "message": message}, status_code=status)

    def authorized(request: Request) -> bool:
        given = request.headers.get(TOKEN_HEADER, "").encode()
        return bool(given) and hmac.compare_digest(given, token)

    async def read_body(request: Request, limit: int) -> bytes | None:
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                return None
        return bytes(body)

    @api.post("/api/analyze")
    async def analyze(request: Request):
        if not authorized(request):
            return error(401, "unauthorized", "Missing or wrong access token.")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BYTES:
            return error(413, "too_large", f"Image larger than {MAX_BYTES // 2**20} MB.")
        body = await read_body(request, MAX_BYTES)
        if body is None:
            return error(413, "too_large", f"Image larger than {MAX_BYTES // 2**20} MB.")
        try:
            with Image.open(io.BytesIO(body)) as field:
                size = field.size
                field.verify()
        except Exception:
            return error(415, "not_an_image", "The file is not a readable JPG, PNG or WebP image.")
        if size[0] * size[1] > MAX_PIXELS:
            return error(413, "too_many_pixels", f"Image of {size[0]}x{size[1]} px, the limit is 4096x4096.")
        started = time.perf_counter()
        try:
            result = await Analyzer().analyze.remote.aio(body)
        except Exception as exc:
            return error(502, "analysis_failed", f"The analysis failed: {type(exc).__name__}.")
        result["round_trip_s"] = round(time.perf_counter() - started, 3)
        return result

    @api.post("/api/review")
    async def review_endpoint(request: Request):
        if not authorized(request):
            return error(401, "unauthorized", "Missing or wrong access token.")
        body = await read_body(request, MAX_BYTES * 4 // 3 + 2**20)
        if body is None:
            return error(413, "too_large", f"Image larger than {MAX_BYTES // 2**20} MB.")
        try:
            payload = json.loads(body)
            analysis, image_bytes = payload["analysis"], base64.b64decode(payload["image"], validate=True)
            with Image.open(io.BytesIO(image_bytes)) as field:
                field.verify()
            if not isinstance(analysis.get("white_cells"), list):
                raise ValueError
        except Exception:
            return error(400, "bad_request", "Send the analysis and the image it was computed on.")
        if not analysis["white_cells"]:
            return error(422, "no_cells", "No white cell to review on this image.")
        started = time.perf_counter()
        try:
            result = await review.remote.aio(analysis, image_bytes, str(payload.get("name") or "field"))
        except Exception as exc:
            return error(502, "review_failed", f"The review failed: {type(exc).__name__}.")
        result["round_trip_s"] = round(time.perf_counter() - started, 3)
        return result

    api.mount("/", StaticFiles(directory="/root/web", html=True), name="web")
    return api


@app.local_entrypoint()
def main(*arglist: str) -> None:
    parser = argparse.ArgumentParser(prog="modal run scripts/modal_app.py")
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-tiling", action="store_true", help="always analyse the whole image in one SAM pass")
    args = parser.parse_args(arglist)
    analyzer = Analyzer()
    results = []
    for path in args.image:
        started = time.perf_counter()
        result = analyzer.analyze.remote(path.read_bytes(), tiling=not args.no_tiling)
        results.append({"path": str(path), "round_trip_s": round(time.perf_counter() - started, 3), **result})
        print(json.dumps(results[-1]), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
