#!/usr/bin/env python3
"""
FakeNewsNet Data Downloader
============================
Downloads the FakeNewsNet CSV files from Kaggle or the official GitHub source
and places them in the expected data directory.

Usage
-----
    python data/download_fakenewsnet.py

Requirements
------------
- kaggle CLI configured (~/.kaggle/kaggle.json with API token), OR
- Manual download from: https://www.kaggle.com/datasets/algord/fake-news

Expected output
---------------
    data/raw/fakenewsnet/politifact_real.csv
    data/raw/fakenewsnet/politifact_fake.csv
    data/raw/fakenewsnet/gossipcop_real.csv
    data/raw/fakenewsnet/gossipcop_fake.csv

Alternative: direct Hugging Face Datasets download (no Kaggle account needed)
"""

import sys
import subprocess
from pathlib import Path

DEST_DIR = Path(__file__).parent / "raw" / "fakenewsnet"
KAGGLE_DATASET = "algord/fake-news"
HF_DATASET = "Fnews/fakenewsnet"

EXPECTED_FILES = [
    "politifact_real.csv",
    "politifact_fake.csv",
    "gossipcop_real.csv",
    "gossipcop_fake.csv",
]


def all_files_present() -> bool:
    return all((DEST_DIR / f).exists() for f in EXPECTED_FILES)


def download_via_kaggle() -> bool:
    """Try downloading via the Kaggle CLI."""
    try:
        import kaggle  # noqa: F401 — just checking importability
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "--unzip", "-p", str(DEST_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[kaggle] Error: {result.stderr.strip()}")
        return False
    print("[kaggle] Download complete.")
    return True


def download_via_huggingface() -> bool:
    """Fallback: download via Hugging Face datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        from datasets import load_dataset

    print(f"[huggingface] Downloading {HF_DATASET} ...")
    try:
        ds = load_dataset(HF_DATASET)
        DEST_DIR.mkdir(parents=True, exist_ok=True)
        for split_name, split_data in ds.items():
            out_path = DEST_DIR / f"{split_name}.csv"
            split_data.to_pandas().to_csv(out_path, index=False)
            print(f"  Saved {out_path} ({len(split_data):,} rows)")
        return True
    except Exception as exc:
        print(f"[huggingface] Failed: {exc}")
        return False


def print_manual_instructions() -> None:
    print(
        "\nManual download instructions\n"
        "============================\n"
        "1. Go to: https://www.kaggle.com/datasets/algord/fake-news\n"
        "2. Click 'Download' (requires a free Kaggle account)\n"
        "3. Unzip and place the four CSV files in:\n"
        f"   {DEST_DIR}/\n\n"
        "Expected filenames:\n"
        "  - politifact_real.csv\n"
        "  - politifact_fake.csv\n"
        "  - gossipcop_real.csv\n"
        "  - gossipcop_fake.csv\n\n"
        "After placing the files, run:\n"
        "  python data/pipeline/quickstart.py\n"
    )


def main() -> None:
    print("FakeNewsNet Downloader")
    print("=" * 40)

    if all_files_present():
        print(f"All {len(EXPECTED_FILES)} CSV files already present in {DEST_DIR}")
        print("Nothing to do. Run the pipeline with:")
        print("  python data/pipeline/quickstart.py")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Try Kaggle first (fastest, highest quality)
    print("Attempting Kaggle download ...")
    if download_via_kaggle() and all_files_present():
        print(f"\nSuccess! Data saved to {DEST_DIR}")
        print("Next step: python data/pipeline/quickstart.py")
        return

    # Fallback to Hugging Face
    print("\nKaggle failed. Trying Hugging Face datasets ...")
    if download_via_huggingface() and all_files_present():
        print(f"\nSuccess! Data saved to {DEST_DIR}")
        print("Next step: python data/pipeline/quickstart.py")
        return

    # Both failed — give manual instructions
    print("\nAutomatic download failed.")
    print_manual_instructions()
    sys.exit(1)


if __name__ == "__main__":
    main()
