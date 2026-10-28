"""Run the few-shot Munich adaptation (lmu_adapt.py) on a multi-core Modal CPU container.

    modal run scripts/modal_lmu_adapt.py
    modal run scripts/modal_lmu_adapt.py --encoder lora   # features from modal_glimmer_lora.py::features
"""

import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = "/root/smearlens"
FEATURES = {
    "frozen": ["runs/lmu/features_resize0.npz", "runs/glimmer/pbc_features.npz", "runs/glimmer-aug/probe_aug.npz", "runs/lmu/adapt.json"],
    "lora": ["runs/lora/lmu_features.npz", "runs/lora/pbc_features.npz", "runs/lora/probe_lora.npz", "runs/lora/adapt.json"],
}
FILES = [f for files in FEATURES.values() for f in files[:3]]

image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy").add_local_dir(REPO / "scripts", f"{REMOTE}/scripts")
for name in FILES:
    image = image.add_local_file(REPO / name, f"{REMOTE}/{name}")
app = modal.App("smearlens-lmu-adapt", image=image)


@app.function(cpu=16, memory=16384, timeout=3600)
def adapt(shots: str, repeats: int, encoder: str) -> str:
    lmu, pbc, probe, out = FEATURES[encoder]
    subprocess.run(
        ["python", "scripts/lmu_adapt.py", "--shots", shots, "--repeats", str(repeats)]
        + ["--lmu", lmu, "--pbc", pbc, "--probe", probe, "--out", out],
        cwd=REMOTE,
        env={"PYTHONPATH": ".:scripts", "PATH": "/usr/local/bin:/usr/bin:/bin"},
        check=True,
    )
    return Path(f"{REMOTE}/{out}").read_text()


@app.local_entrypoint()
def main(shots: str = "0,5,10,25,50,100", repeats: int = 3, encoder: str = "frozen"):
    out = REPO / FEATURES[encoder][3]
    out.write_text(adapt.remote(shots, repeats, encoder))
    print(out.read_text())
