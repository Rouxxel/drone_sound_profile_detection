"""
Create MFCC heatmap + RMS plots for each CSV in datasets/converted_csv/,
and save them in EDA/plots/. Run from any directory; paths are relative to this script.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

# ---------------- SETTINGS ---------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FOLDER = SCRIPT_DIR.parent / "datasets" / "converted_csv"
PLOTS_DIR = SCRIPT_DIR / "plots"
AUDIOS_FOLDER = SCRIPT_DIR.parent / "datasets" / "audios"

samp_rate = 44100
hop_length = 512

# Set to None to plot ALL CSVs; set to a filename (e.g. "DRONE_001.csv") to plot only that file
plot_one = None

# Create plots folder if it doesn't exist
PLOTS_DIR.mkdir(exist_ok=True)

if not CSV_FOLDER.exists():
    print(f"❌ CSV folder not found: {CSV_FOLDER}")
    print("   Run audio_to_csv_converter.py first to create converted_csv/.")
    exit(1)

print(f"📂 Reading from: {CSV_FOLDER}")
print(f"📂 Saving to:   {PLOTS_DIR}\n")

# Pick which files to process
if plot_one is None:
    csv_files = sorted(f for f in os.listdir(CSV_FOLDER) if f.lower().endswith(".csv"))
else:
    csv_files = [plot_one]

for csv_file in csv_files:
    csv_path = CSV_FOLDER / csv_file

    if not csv_path.exists():
        print(f"Skipping {csv_file} (not found).")
        continue

    base_name = csv_path.stem
    class_name = base_name.split("_")[0]
    plot_path = PLOTS_DIR / f"{base_name}.png"

    if plot_one is None and plot_path.exists():
        print(f"Skipping {csv_file} -> plot already exists.")
        continue

    print(f"Processing {csv_file}...")

    # Load MFCC CSV
    mfcc_df = pd.read_csv(csv_path)
    mfcc_data = mfcc_df.values
    n_frames, n_mfcc = mfcc_data.shape

    # Timestamps from frame count (same convention as converter: hop_length=512)
    timestamps = np.arange(n_frames) * hop_length / samp_rate

    # RMS from corresponding WAV if available
    audio_path = AUDIOS_FOLDER / f"{base_name}.wav"
    if audio_path.exists():
        y, _ = librosa.load(str(audio_path), sr=samp_rate)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_timestamps = np.arange(len(rms)) * hop_length / samp_rate
        has_rms = True
    else:
        has_rms = False

    # Plotting
    n_subplots = 2 if has_rms else 1
    plt.figure(figsize=(14, 6 if has_rms else 3))

    # MFCC heatmap
    plt.subplot(n_subplots, 1, 1)
    plt.imshow(
        mfcc_data.T,
        aspect="auto",
        origin="lower",
        extent=[timestamps[0], timestamps[-1], 1, n_mfcc],
    )
    plt.colorbar(label="Normalized MFCC")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficient")
    plt.title(f"{class_name} – MFCC Heatmap")

    # RMS volume (only if WAV was found)
    if has_rms:
        plt.subplot(2, 1, 2)
        plt.plot(rms_timestamps, rms, color="orange")
        plt.xlabel("Time (s)")
        plt.ylabel("Volume (RMS)")
        plt.title(f"{class_name} – Volume Over Time")
    else:
        plt.suptitle(f"{class_name} – MFCC only (no WAV for RMS)", y=1.02)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"  Saved {plot_path.name}")

print("✅ Done.")
