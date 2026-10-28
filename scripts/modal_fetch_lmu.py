"""Download the Munich AML-Cytomorphology_LMU dataset (TCIA, CC BY 3.0) into a Modal volume.

    modal run scripts/modal_fetch_lmu.py

Matek C, Schwarz S, Marr C, Spiekermann K. A Single-cell Morphological Dataset of
Leukocytes from AML Patients and Non-malignant Controls. The Cancer Imaging Archive,
2019. https://doi.org/10.7937/tcia.2019.36f5o9ld
"""

import subprocess

import modal

FASPEX = (
    "https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context="
    "eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjczNSIsInBhc3Njb2Rl"
    "IjoiYTczZTE1NzU1MjI5MzZkODRhZTY3MTcxMmU1YTg2YWY1ZTZlODI4MyIsInBhY2thZ2VfaWQiOiI3MzUiLCJlbWFpbCI6ImhlbHBA"
    "Y2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0="
)
ANNOTATIONS = "https://www.cancerimagingarchive.net/wp-content/uploads/AML-CYTOMORPHOLOGY_LMU-annotations-dat.zip"

image = (
    modal.Image.debian_slim()
    .apt_install("ruby-full", "build-essential", "curl", "unzip")
    .run_commands("gem install aspera-cli", "ascli config ascp install")
)
app = modal.App("smearlens-fetch-lmu", image=image)
data = modal.Volume.from_name("smearlens-data", create_if_missing=True)


@app.function(timeout=4 * 3600, volumes={"/data": data})
def fetch() -> str:
    target = "/data/lmu"
    subprocess.run(["mkdir", "-p", target], check=True)
    subprocess.run(["curl", "-sL", "-o", f"{target}/annotations.zip", ANNOTATIONS], check=True)
    subprocess.run(["unzip", "-o", "-q", f"{target}/annotations.zip", "-d", target], check=True)
    result = subprocess.run(
        ["ascli", "faspex", "package", "receive", f"--url={FASPEX}", f"--to-folder={target}"],
        capture_output=True,
        text=True,
    )
    data.commit()
    listing = subprocess.run(["du", "-sh", target], capture_output=True, text=True).stdout
    return f"exit={result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}\n{listing}"


@app.local_entrypoint()
def main():
    print(fetch.remote())
