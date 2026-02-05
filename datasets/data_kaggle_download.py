# data_kaggle_download.py
# Requires: pip install kagglehub and kaggle.json credentials with API key
#
# Downloads the Kaggle dataset, copies all .wav files from the nested structure
# (e.g. DroneDetectionThesis/.../Data/Audio/) into datasets/audios/, then
# deletes the downloaded DroneDetectionThesis folder.

import os
from pathlib import Path
import shutil
import kagglehub
from kagglehub.exceptions import KaggleApiHTTPError

DATASET = "tacticularcancer/drone-detection-dataset"

# Same directory as this script (datasets/)
SCRIPT_DIR = Path(__file__).resolve().parent
AUDIOS_DIR = SCRIPT_DIR / "audios"
DRONE_DETECTION_TOP = SCRIPT_DIR / "DroneDetectionThesis"

print(f"📥 Attempting to download dataset '{DATASET}' using KaggleHub...")

try:
    dataset_cache_path = Path(kagglehub.dataset_download(DATASET))
    print(f"✅ Download complete. Cached at: {dataset_cache_path}")

    print(f"📦 Copying dataset into: {SCRIPT_DIR}")
    for root, _, files in os.walk(dataset_cache_path):
        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, dataset_cache_path)
            dest_path = SCRIPT_DIR / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)

    if not DRONE_DETECTION_TOP.exists():
        print(f"⚠️ Expected folder not found: {DRONE_DETECTION_TOP}")
        print("   Skipping .wav copy and cleanup. Check the dataset structure.")
    else:
        # Collect all .wav files under DroneDetectionThesis (handles variable paths e.g. .../Data/Audio/)
        wav_files = list(DRONE_DETECTION_TOP.rglob("*.wav"))
        if not wav_files:
            print("⚠️ No .wav files found under DroneDetectionThesis.")
        else:
            AUDIOS_DIR.mkdir(exist_ok=True)
            print(f"📂 Copying {len(wav_files)} .wav files to: {AUDIOS_DIR}")
            for wav_path in wav_files:
                dest = AUDIOS_DIR / wav_path.name
                shutil.copy2(wav_path, dest)
            print(f"✅ All .wav files copied to {AUDIOS_DIR}")

        # Remove the entire DroneDetectionThesis folder
        print(f"🗑️ Removing folder: {DRONE_DETECTION_TOP}")
        shutil.rmtree(DRONE_DETECTION_TOP)
        print("✅ DroneDetectionThesis folder removed.")

    print("\n✅ Done.")

except KaggleApiHTTPError as e:
    print("❌ KaggleHub Error: Dataset could not be downloaded.")
    print("   → It may be private, deleted, or missing files.")
    print(f"   → Dataset slug: {DATASET}")
    print(f"   → Error: {e}")
