import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

# ---------------- SETTINGS ---------------- #

csv_folder = "../datasets/converted_csv/"
plots_dir = "plots"

samp_rate = 44100        # sampling rate used for MFCC extraction
hop_length = 512         # hop_length used for MFCC extraction

# Set to None to plot ALL CSVs
# Set to a filename (e.g., "DRONE_001.csv") to plot ONLY that file
plot_one = None

#Create output folder
os.makedirs(plots_dir, exist_ok=True)

# Pick which files to process
if plot_one is None:
    csv_files = [f for f in os.listdir(csv_folder) if f.lower().endswith(".csv")]
else:
    csv_files = [plot_one]  # Only one file

#Iterate through selected files
for csv_file in csv_files:
    csv_path = os.path.join(csv_folder, csv_file)

    #Extract base name (e.g., DRONE_001)
    base_name = os.path.splitext(csv_file)[0]

    #Extract class name (text before first underscore)
    class_name = base_name.split("_")[0]

    #Output plot file
    plot_path = os.path.join(plots_dir, f"{base_name}.png")

    #Skip if it already exists (only when plotting all)
    if plot_one is None and os.path.exists(plot_path):
            print(f"Skipping {csv_file} -> plot already exists.")
            continue
    print(f"Processing {csv_file}...")

    #Corresponding WAV path
    audio_path = csv_path.replace("converted_csv", "audios").replace(".csv", ".wav")

    # --- Load MFCC CSV ---
    mfcc_df = pd.read_csv(csv_path)
    mfcc_data = mfcc_df.values
    n_frames, n_mfcc = mfcc_data.shape

    # --- Timestamps ---
    timestamps = np.arange(n_frames) * hop_length / samp_rate

    # --- RMS volume curve ---
    y, _ = librosa.load(audio_path, sr=samp_rate)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_timestamps = np.arange(len(rms)) * hop_length / samp_rate

    # ----------------- PLOTTING ----------------- #
    plt.figure(figsize=(14, 6))

    # Heatmap
    plt.subplot(2, 1, 1)
    plt.imshow(
            mfcc_data.T,
            aspect='auto',
            origin='lower',
            extent=[timestamps[0], timestamps[-1], 1, n_mfcc]
            )
    plt.colorbar(label="Normalized MFCC")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficient")
    plt.title(f"{class_name} – MFCC Heatmap")

    #RMS volume
    plt.subplot(2, 1, 2)
    plt.plot(rms_timestamps, rms, color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Volume (RMS)")
    plt.title(f"{class_name} – Volume Over Time")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved: {plot_path}")

print("✅ Done.")
