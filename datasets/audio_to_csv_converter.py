"""
Convert audio files in audios/ to MFCC CSV files in converted_csv/.
Both folders are at the same level (sibling to this script in datasets/).
Run from any directory; paths are relative to this script's location.

CSV outputs Column Mapping:
- [0-13]: MFCCs (14 coefficients)
"""

import os
from pathlib import Path
import librosa
import numpy as np
import pandas as pd

# Paths: same level as this script (datasets/)
SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_FOLDER = SCRIPT_DIR / "audios"
OUTPUT_FOLDER = SCRIPT_DIR / "converted_csv"

FILE_TYPES = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

# Create converted_csv folder if it doesn't exist
OUTPUT_FOLDER.mkdir(exist_ok=True)

if not AUDIO_FOLDER.exists():
    print(f"❌ Audio folder not found: {AUDIO_FOLDER}")
    print("   Run data_kaggle_download.py first to create audios/ with .wav files.")
    exit(1)

print(f"📂 Reading from: {AUDIO_FOLDER}")
print(f"📂 Saving to:   {OUTPUT_FOLDER}\n")

processed = 0
for filename in sorted(os.listdir(AUDIO_FOLDER)):
    if not filename.lower().endswith(FILE_TYPES):
        continue

    csv_filename = Path(filename).stem + ".csv"
    csv_path = OUTPUT_FOLDER / csv_filename

    if csv_path.exists():
        print(f"Skipping {filename} -> {csv_filename} already exists.")
        continue

    file_path = AUDIO_FOLDER / filename
    print(f"Processing {filename}...")

    # Load audio
    y, sr = librosa.load(str(file_path), sr=None)

    # Compute MFCCs (14 coefficients recommended for drones)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)

    # Normalize MFCCs
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Transpose: rows = time frames, columns = MFCC coefficients
    mfcc = mfcc.T

    # Save as CSV
    mfcc_df = pd.DataFrame(mfcc)
    mfcc_df.to_csv(csv_path, index=False)

    print(f"  Saved {csv_filename}")
    processed += 1

print(f"\n✅ Done. Processed {processed} new file(s).")
