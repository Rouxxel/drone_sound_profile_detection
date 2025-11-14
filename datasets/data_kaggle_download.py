# data_kaggle_download.py
# Requires: pip install kagglehub and kaggle.json credentials with API key

import os
import shutil
import kagglehub
from kagglehub.exceptions import KaggleApiHTTPError

DATASET = "tacticularcancer/drone-detection-dataset"
#Extract only the audio files, moge to "audios/" and delete de rest!!!

# Save dataset files in the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"📥 Attempting to download dataset '{DATASET}' using KaggleHub...")

try:
    dataset_cache_path = kagglehub.dataset_download(DATASET)
    print(f"✅ Download complete. Cached at: {dataset_cache_path}")

    print(f"📦 Copying files to: {SCRIPT_DIR}")
    for root, _, files in os.walk(dataset_cache_path):
        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, dataset_cache_path)
            dest_path = os.path.join(SCRIPT_DIR, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)

    print("\n✅ Dataset saved in this folder.")
    print("📂 Files downloaded:")
    for root, _, files in os.walk(SCRIPT_DIR):
        for name in files:
            if name != os.path.basename(__file__):
                print(" -", os.path.relpath(os.path.join(root, name), SCRIPT_DIR))

except KaggleApiHTTPError as e:
    print("❌ KaggleHub Error: Dataset could not be downloaded.")
    print("   → It may be private, deleted, or missing files.")
    print(f"   → Dataset slug: {DATASET}")
    print(f"   → Error: {e}")
