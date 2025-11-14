import os
import librosa
import numpy as np
import pandas as pd

#--- Paths ---
audio_folder = "audios"
output_folder = "converted_csv"
file_types = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

#Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

#Iterate through all .wav files in the folder
for filename in os.listdir(audio_folder):
    if filename.lower().endswith(file_types):
        file_path = os.path.join(audio_folder, filename)
        print(f"Processing {filename}...")

        #Load audio
        y, sr = librosa.load(file_path, sr=None)

        #Compute MFCCs (14 coefficients recommended for drones)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)

        #Normalize MFCCs
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        #Transpose so rows = time frames, columns = MFCC coefficients
        mfcc = mfcc.T

        #Convert to DataFrame and save as CSV
        mfcc_df = pd.DataFrame(mfcc)
        csv_filename = os.path.splitext(filename)[0] + ".csv"
        csv_path = os.path.join(output_folder, csv_filename)
        mfcc_df.to_csv(csv_path, index=False)

        print(f"Saved {csv_filename} in {output_folder}")

print("✅ All files processed.")
