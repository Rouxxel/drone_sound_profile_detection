import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

# --- Paths ---
csv_path = "../converted_csv/DRONE_002.csv"
audio_path = csv_path.replace("converted_csv", "audios").replace(".csv", ".wav")  # corresponding wav
sr = 44100            # sampling rate used when extracting MFCCs
hop_length = 512      # hop_length used during MFCC extraction

# --- Load MFCC CSV ---
mfcc_df = pd.read_csv(csv_path)
mfcc_data = mfcc_df.values
n_frames, n_mfcc = mfcc_data.shape

# --- Compute timestamps for frames ---
timestamps = np.arange(n_frames) * hop_length / sr  # in seconds

# --- Load original audio to compute RMS for volume plot ---
y, _ = librosa.load(audio_path, sr=sr)
rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]  # shape: (n_frames,)

rms_timestamps = np.arange(len(rms)) * hop_length / sr

# --- Plot heatmap of MFCCs ---
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.imshow(mfcc_data.T, aspect='auto', origin='lower',
        extent=[timestamps[0], timestamps[-1], 1, n_mfcc])
plt.colorbar(label='Normalized MFCC')
plt.xlabel("Time (s)")
plt.ylabel("MFCC Coefficient")
plt.title("MFCC Heatmap of DRONE_001.wav")

# --- Plot RMS (volume) over time ---
plt.subplot(2, 1, 2)
plt.plot(rms_timestamps, rms, color='orange')
plt.xlabel("Time (s)")
plt.ylabel("Volume (RMS)")
plt.title("Drone Volume Over Time")
plt.tight_layout()
plt.show()
