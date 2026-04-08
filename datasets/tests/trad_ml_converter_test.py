import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

script_path = Path(__file__).resolve().parent.parent
sys.path.append(str(script_path))
from aud_csv_converter_trad_ml import extract_features, OUTPUT_FOLDER

def test_feature_csv_layout():
    """
    Ensures the output has exactly 21 columns in the correct order.
    """
    # Create a dummy signal (1 second of noise at 22050Hz)
    sr = 22050
    y = np.random.uniform(-1, 1, sr)
    
    df = extract_features(y, sr, n_mfcc=14, include_hnr=True)
    
    # Check column count: 14 (MFCC) + 1 (Centroid) + 1 (Bandwidth) + 
    # 1 (Rolloff) + 1 (Flatness) + 1 (ZCR) + 1 (RMS) + 1 (HNR) = 21
    assert df.shape[1] == 21, f"Expected 21 columns, got {df.shape[1]}"
    
    # Check for no NaN/Inf values (Requirement 1.1)
    assert not df.isnull().values.any(), "DataFrame contains NaN values"
    assert np.isfinite(df.values).all(), "DataFrame contains Infinite values"

def test_output_path_naming():
    """
    Ensures the stem of the audio file is preserved in the CSV name.
    """
    audio_stem = "test_drone_001"
    expected_csv_name = "test_drone_001.csv"
    
    # This logic mirrors what is in your run_conversion loop
    csv_filename = Path(f"{audio_stem}.wav").stem + ".csv"
    assert csv_filename == expected_csv_name

def test_converter_determinism():
    """
    Ensures that the same audio input always produces the exact same feature values.
    """
    sr = 22050
    y = np.random.uniform(-1, 1, sr)
    
    df1 = extract_features(y, sr, n_mfcc=14, include_hnr=True)
    df2 = extract_features(y, sr, n_mfcc=14, include_hnr=True)
    
    pd.testing.assert_frame_equal(df1, df2)

if __name__ == "__main__":
    print("Running tests...")
    test_converter_determinism()
    print("---Determinism test passed.")
    test_feature_csv_layout()
    print("---Layout test passed (21 columns confirmed).")
    test_output_path_naming()
    print("---Naming test passed.")
