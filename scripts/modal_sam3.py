"""Run scripts/sam3_validate.py on a Modal GPU.

    modal secret create huggingface HF_TOKEN=...
    modal run scripts/modal_sam3.py --limit 10
    SMEARLENS_GPU=L4 modal run scripts/modal_sam3.py  # other GPU type

Results are copied back to runs/<tag> (default runs/sam3).
"""

import io
import os
import tarfile
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent
REMOTE = "/root/smearlens"

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
    .add_local_dir(REPO / "smearlens", f"{REMOTE}/smearlens")
    .add_local_dir(REPO / "scripts", f"{REMOTE}/scripts")
    .add_local_dir(REPO / "data" / "txl-pbc" / "TXL-PBC", f"{REMOTE}/data/txl-pbc/TXL-PBC")
)

app = modal.App("smearlens-sam3", image=image)
hf_cache = modal.Volume.from_name("smearlens-hf-cache", create_if_missing=True)


@app.function(
    gpu=os.environ.get("SMEARLENS_GPU", "L40S"),
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def validate(args: list[str], tag: str = "sam3") -> bytes:
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": REMOTE}
    subprocess.run(
        ["python", "scripts/sam3_validate.py", "--out", f"runs/{tag}", *args],
        cwd=REMOTE,
        env=env,
        check=True,
    )
    hf_cache.commit()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(f"{REMOTE}/runs/{tag}", arcname=tag)
    return buffer.getvalue()


@app.local_entrypoint()
def main(
    limit: int = 10,
    split: str = "test",
    version: str = "sam3.1",
    threshold: float = 0.5,
    wbc_prompt: str = "white blood cell",
    platelet_prompt: str = "platelet",
    tag: str = "sam3",
):
    args = ["--limit", str(limit), "--split", split, "--version", version, "--threshold", str(threshold)]
    archive = validate.remote([*args, "--wbc-prompt", wbc_prompt, "--platelet-prompt", platelet_prompt], tag)
    out = REPO / "runs"
    out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out, filter="data")
    print(f"Results in {out / tag}")
