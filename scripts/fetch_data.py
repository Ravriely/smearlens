import argparse
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

TXL_PBC_REPO = "https://github.com/lugan113/TXL-PBC_Dataset.git"
PBC_URL = (
    "https://data.mendeley.com/public-files/datasets/snkd93bnjr/files/"
    "2fc38728-2ae7-4a62-a857-032af82334c3/file_downloaded"
)


def fetch_txl_pbc(data_dir: Path) -> None:
    target = data_dir / "txl-pbc"
    if target.exists():
        print(f"TXL-PBC already in {target}")
        return
    subprocess.run(["git", "clone", "--depth", "1", TXL_PBC_REPO, str(target)], check=True)


def fetch_pbc(data_dir: Path) -> None:
    target = data_dir / "pbc"
    if target.exists():
        print(f"PBC already in {target}")
        return
    archive = data_dir / "pbc.zip"
    print("Downloading PBC (281 MB)...")
    urllib.request.urlretrieve(PBC_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    shutil.rmtree(target / "__MACOSX", ignore_errors=True)
    archive.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public datasets listed in DATA.md.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    fetch_txl_pbc(args.data_dir)
    fetch_pbc(args.data_dir)


if __name__ == "__main__":
    main()
