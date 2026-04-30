"""
Convert audio files in audios/ to feature-rich CSV files in converted_csv/.
Includes MFCCs, Spectral Centroid, Bandwidth, Rolloff, Flatness, ZCR, and RMS.
Both folders are at the same level (sibling to this script in datasets/).
Run from any directory; paths are relative to this script's location.

CSV outputs Column Mapping:
- [0-13]: MFCCs (14 coefficients)
- [14]:   Spectral Centroid
- [15]:   Spectral Bandwidth
- [16]:   Spectral Rolloff
- [17]:   Spectral Flatness
- [18]:   Zero Crossing Rate
- [19]:   RMS Energy
- [20]:   HNR (Harmonic-to-Noise Ratio)
"""

import os
import sys
from pathlib import Path
import librosa
import numpy as np
import pandas as pd

#------------------------------------------------------
# Paths: same level as this script
#------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent #datasets/
REPO_ROOT = SCRIPT_DIR.parent #root/
AUDIO_FOLDER = SCRIPT_DIR / "audios"

#------------------------------------------------------
# Configuration
#------------------------------------------------------
sys.path.append(str(REPO_ROOT))
from configuration.config_loader import config
OUTPUT_FOLDER = SCRIPT_DIR / config["audio_converters"]["trad_ml_models"]["output_folder_str"]
N_MFCC = config["audio_converters"]["trad_ml_models"]["n_mfcc"]
INCLUDE_HNR = config["audio_converters"]["trad_ml_models"]["include_hnr"]
FILE_TYPES = tuple(config["audio_converters"]["input_file_extensions"])
OUTPUT_TYPE = config["audio_converters"]["output_file_extensions"]

#------------------------------------------------------
# Feature Extraction Functions
#------------------------------------------------------
def extract_features(y, sr, n_mfcc=N_MFCC, include_hnr=INCLUDE_HNR):
    """
    Extract features per frame using a single STFT computation.
    Returns a DataFrame with columns: MFCCs, Spectral features and optionally HNR enabled by default.
    """
    # Compute STFT once and derive power spectrogram
    stft_complex = librosa.stft(y)
    stft_mag = np.abs(stft_complex)
    power_spec = stft_mag**2

    #MFCCs derived from power spectrogram
    #NOTE: librosa.feature.mfcc internally handles the conversion to Mel scale
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(power_spec), sr=sr, n_mfcc=n_mfcc)

    #Extract Spectral features using the already computed magnitude/power
    spectral_centroid = librosa.feature.spectral_centroid(S=stft_mag, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft_mag, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=stft_mag, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(S=stft_mag)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rms_energy = librosa.feature.rms(S=stft_mag)

    #Build the feature list in fixed order
    feature_list = [
        mfccs, 
        spectral_centroid, 
        spectral_bandwidth, 
        spectral_rolloff, 
        spectral_flatness, 
        zero_crossing_rate, 
        rms_energy
    ]

    #Append HNR (Harmonic-to-Noise Ratio)
    #NOTE: enabled by default
    if include_hnr:
        # Estimation using Harmonic-Percussive Source Separation
        y_harm, y_perc = librosa.effects.hpss(y)
        harmonic_rms = librosa.feature.rms(y=y_harm)
        percussive_rms = librosa.feature.rms(y=y_perc)
        hnr = harmonic_rms / (percussive_rms + 1e-6)
        feature_list.append(hnr)

    # COncat all features along axis 0 (rows currently) into a single
    features_concat = np.concatenate(feature_list, axis=0)

    # Transpose so rows = time frames, columns = features (convert to dataframe)
    datfram = pd.DataFrame(features_concat.T)
    datfram = datfram.replace([np.inf, -np.inf], np.nan).fillna(0) #NaN/Inf with 0
    
    return datfram

def run_conversion():
    if not AUDIO_FOLDER.exists(): #Small check
        print(f"!!! Audio folder not found: {AUDIO_FOLDER}")
        print("!!! Run data_kaggle_download.py first to extract audios/ with .wav files.")
        return

    # Create converted_csv folder
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    print(f"-Reading from: {AUDIO_FOLDER}")
    print(f"--Saving to:   {OUTPUT_FOLDER}\n")

    processed = 0
    for filename in sorted(os.listdir(AUDIO_FOLDER)):
        if not filename.lower().endswith(FILE_TYPES):
            continue

        csv_filename = Path(filename).stem + OUTPUT_TYPE
        csv_path = OUTPUT_FOLDER / csv_filename #save in new folder

        if csv_path.exists():
            print(f"!!! Skipping {filename} -> {csv_filename} already exists.")
            continue

        file_path = AUDIO_FOLDER / filename
        print(f"--Processing {filename}...")
        try:
            # Load audio
            y, sr = librosa.load(str(file_path), sr=None)

            # Extract new feature set (14 MFCCs)
            feature_df = extract_features(y, sr, n_mfcc=14, include_hnr=True)

            # Save as CSV
            feature_df.to_csv(csv_path, index=False)
            print(f"---Saved {OUTPUT_FOLDER}/{csv_filename}")
            processed += 1
        except Exception as e:
            print(f"!!! Error processing {filename}: {e}")

    print(f"\nDone. Processed {processed} new file(s).")

if __name__ == "__main__":
    run_conversion()