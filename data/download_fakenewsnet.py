#!/usr/bin/env python3
"""
FakeNewsNet Data Downloader
============================
Downloads the FakeNewsNet CSV files from the source repository
(sanjaykshetri/Misinformation-Detection-ML-Model2) and places them in
the expected data directory.

Usage
-----
    python data/download_fakenewsnet.py

Expected output
---------------
    data/raw/fakenewsnet/politifact_real.csv
    data/raw/fakenewsnet/politifact_fake.csv
    data/raw/fakenewsnet/gossipcop_real.csv
    data/raw/fakenewsnet/gossipcop_fake.csv
"""

import sys
import urllib.request
import urllib.error
from pathlib import Path

DEST_DIR = Path(__file__).parent / "raw" / "fakenewsnet"

# Source: sanjaykshetri/Misinformation-Detection-ML-Model2
_GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "sanjaykshetri/Misinformation-Detection-ML-Model2/main/FakeNewsNet/dataset"
)

EXPECTED_FILES = [
    "politifact_real.csv",
    "politifact_fake.csv",
    "gossipcop_real.csv",
    "gossipcop_fake.csv",
]


def all_files_present() -> bool:
    return all((DEST_DIR / f).exists() for f in EXPECTED_FILES)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_from_github() -> bool:
    """Download CSVs from the GitHub raw URL."""
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for filename in EXPECTED_FILES:
        url = f"{_GITHUB_RAW_BASE}/{filename}"
        dest = DEST_DIR / filename
        if dest.exists():
            print(f"  [skip] {filename} already present")
            continue
        print(f"  Downloading {filename} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size = dest.stat().st_size
            print(f"done ({_human_size(size)})")
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code}")
            # Clean up partial file
            dest.unlink(missing_ok=True)
            return False
        except urllib.error.URLError as exc:
            print(f"network error: {exc.reason}")
            dest.unlink(missing_ok=True)
            return False
    return True


def download_via_git_sparse() -> bool:
    """Fallback: sparse-checkout only the dataset folder."""
    import subprocess, tempfile, shutil

    print("  Using git sparse-checkout ...")
    with tempfile.TemporaryDirectory() as tmp:
        clone_url = (
            "https://github.com/sanjaykshetri/Misinformation-Detection-ML-Model2"
        )
        cmds = [
            ["git", "clone", "--no-checkout", "--filter=blob:none",
             "--depth=1", clone_url, tmp],
            ["git", "-C", tmp, "sparse-checkout", "set", "FakeNewsNet/dataset"],
            ["git", "-C", tmp, "checkout"],
        ]
        for cmd in cmds:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  git error: {result.stderr.strip()}")
                return False
        src = Path(tmp) / "FakeNewsNet" / "dataset"
        DEST_DIR.mkdir(parents=True, exist_ok=True)
        for filename in EXPECTED_FILES:
            shutil.copy2(src / filename, DEST_DIR / filename)
            print(f"  Copied {filename}")
    return True


def main() -> None:
    print("FakeNewsNet Downloader")
    print("=" * 40)
    print(f"Source: sanjaykshetri/Misinformation-Detection-ML-Model2")
    print(f"Destination: {DEST_DIR}\n")

    if all_files_present():
        print(f"All {len(EXPECTED_FILES)} CSV files already present.")
        print("Run the pipeline with:  python data/pipeline/quickstart.py")
        return

    print("Downloading via GitHub raw URLs ...")
    if download_from_github() and all_files_present():
        print(f"\nDone. {len(EXPECTED_FILES)} files saved to {DEST_DIR}")
        print("Next step:  python data/pipeline/quickstart.py")
        return

    print("\nDirect download failed. Trying git sparse-checkout ...")
    if download_via_git_sparse() and all_files_present():
        print(f"\nDone. {len(EXPECTED_FILES)} files saved to {DEST_DIR}")
        print("Next step:  python data/pipeline/quickstart.py")
        return

    print(
        "\nAutomatic download failed.\n"
        "Manual steps:\n"
        "  1. Clone https://github.com/sanjaykshetri/Misinformation-Detection-ML-Model2\n"
        "  2. Copy FakeNewsNet/dataset/*.csv to:\n"
        f"     {DEST_DIR}/\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
