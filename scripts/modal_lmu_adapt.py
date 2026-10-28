"""Run the few-shot Munich adaptation (lmu_adapt.py) on a multi-core Modal CPU container.

    modal run scripts/modal_lmu_adapt.py
"""

import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = "/root/smearlens"
FILES = ["runs/lmu/features_resize0.npz", "runs/glimmer/pbc_features.npz", "runs/glimmer-aug/probe_aug.npz"]

image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy").add_local_dir(REPO / "scripts", f"{REMOTE}/scripts")
for name in FILES:
    image = image.add_local_file(REPO / name, f"{REMOTE}/{name}")
app = modal.App("smearlens-lmu-adapt", image=image)


@app.function(cpu=16, memory=16384, timeout=3600)
def adapt(shots: str, repeats: int) -> str:
    subprocess.run(
        ["python", "scripts/lmu_adapt.py", "--shots", shots, "--repeats", str(repeats)],
        cwd=REMOTE,
        env={"PYTHONPATH": ".:scripts", "PATH": "/usr/local/bin:/usr/bin:/bin"},
        check=True,
    )
    return Path(f"{REMOTE}/runs/lmu/adapt.json").read_text()


@app.local_entrypoint()
def main(shots: str = "0,5,10,25,50,100", repeats: int = 3):
    out = REPO / "runs" / "lmu" / "adapt.json"
    out.write_text(adapt.remote(shots, repeats))
    print(out.read_text())
