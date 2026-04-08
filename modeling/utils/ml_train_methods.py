#!/usr/bin/env python3
"""
Traditional ML Models util methods for Audio Classification
"""

from pathlib import Path
import sys
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#custom logger
SCRIPT_DIR = Path(__file__).resolve().parent #multiclass/
REPO_ROOT = SCRIPT_DIR.parent.parent.parent #root/

#custom logger
sys.path.append(str(REPO_ROOT))
from modeling.utils.custom_logger import get_logger
logger = get_logger("ml_models_logger", "ml_models_training.log")

# -------------------------------------------------------------------------
def load_dataset(
    csv_dir:str, 
    test_split:float = 0.2,
    shffl:bool = True, 
    rand_state:int = 42,
    is_binary:bool = False #flag
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a complete audio feature dataset from a folder containing MFCC CSV files. Scans
    specified directory for CSV files, infers their class labels from the filename 
    (BACKGROUND, HELICOPTER, DRONE), loads the MFCC matrices, groups samples by class 
    and performs an train/validation split
    for each class.

    Returns
    -------
    X_train : np.ndarray of object
        List-like array of MFCC matrices used for training.
    y_train : np.ndarray
        Integer class labels for the training set.
    X_val : np.ndarray of object
        List-like array of MFCC matrices used for validation.
    y_val : np.ndarray
        Integer class labels for the validation set.
    """

    csv_path = Path(csv_dir)
    assert csv_path.exists(), f"CSV directory not found: {csv_dir}"

    logger.info(f"Loading dataset from {csv_path}")
    if is_binary:
        data_by_class = {0: [], 1: []}  # 0: NO_DRONE, 1: DRONE
        logger.info("Binary Classification: DRONE (1) vs NO_DRONE (0)")
    else:
        data_by_class = {0: [], 1: [], 2: []} # 0: BG, 1: HELI, 2: DRONE
        logger.info("Binary Classification: DRONE (2) vs HELICOPTER (1) vs BG (0)")

    for file in csv_path.glob("*.csv"):
        name = file.stem.upper()

        #Binary
        if is_binary:
            if name.startswith("DRONE"):
                label = 1
            elif name.startswith("HELICOPTER") or name.startswith("BACKGROUND"):
                label = 0
            else:
                continue
        #Multiclass
        else:
            if name.startswith("BACKGROUND"): label = 0
            elif name.startswith("HELICOPTER"): label = 1
            elif name.startswith("DRONE"): label = 2
            else: continue

        mfcc = pd.read_csv(file).values.astype(np.float32)
        data_by_class[label].append(mfcc)

    X_train, y_train, X_val, y_val = [], [], [], []
    for label, samples in data_by_class.items():
        if len(samples) == 0:
            logger.error(f"No samples found for class {label}.")
            continue
        train_split, val_split = train_test_split(samples, test_size=test_split, shuffle=shffl, random_state=rand_state)
        X_train.extend(train_split)
        y_train.extend([label]*len(train_split))
        X_val.extend(val_split)
        y_val.extend([label]*len(val_split))

    X_train, y_train = np.array(X_train, dtype=object), np.array(y_train)
    X_val, y_val = np.array(X_val, dtype=object), np.array(y_val)
    
    logger.info(f"Total Training: {len(X_train)}, Total Validation: {len(X_val)}")
    if is_binary:
        logger.info(f"Train Breakdown -> NO_DRONE (0): {np.sum(y_train==0)}, DRONE (1): {np.sum(y_train==1)}")
    else:
        logger.info(f"Train Breakdown -> BG (0): {np.sum(y_train==0)}, HELI (1): {np.sum(y_train==1)}, DRONE (2): {np.sum(y_train==2)}")
    
    return X_train, y_train, X_val, y_val

# -------------------------------------------------------------------------
def extract_statistical_features(mfcc: np.ndarray) -> np.ndarray:
    """
    Extract a fixed-length statistical feature vector from an MFCC matrix. Take 
    MFCC time-series matrix (shape: frames × coefficients) and computes 
    summary statistics across time for each MFCC coefficient.

    Returns
    -------
    np.ndarray
        1D feature vector (float32) containing the following statistics
        for each MFCC coefficient:

        - Mean
        - Standard deviation
        - Minimum value
        - Maximum value
        - Median
        - 25th percentile
        - 75th percentile
        - Mean of first-order delta (temporal derivative)
        - Std of first-order delta

        If the MFCC contains fewer than 2 frames, delta features are
        replaced with zeros.
    """
    # Ensure mfcc is a proper numpy array with float dtype
    mfcc = np.asarray(mfcc, dtype=np.float32)  
    features = []
    
    # Mean, std, min, max for each coefficient
    features.extend(np.mean(mfcc, axis=0))
    features.extend(np.std(mfcc, axis=0))
    features.extend(np.min(mfcc, axis=0))
    features.extend(np.max(mfcc, axis=0))
    
    # Median and percentiles
    features.extend(np.median(mfcc, axis=0))
    features.extend(np.percentile(mfcc, 25, axis=0))
    features.extend(np.percentile(mfcc, 75, axis=0))
    
    # Delta features (first derivative)
    delta = np.diff(mfcc, axis=0)
    if len(delta) > 0:
        features.extend(np.mean(delta, axis=0))
        features.extend(np.std(delta, axis=0))
    else:
        features.extend(np.zeros(mfcc.shape[1]))
        features.extend(np.zeros(mfcc.shape[1]))

    #Convert list to np array for vector operations
    vec = np.array(features, dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0) #nan_to_num before return
    
    return vec

# -------------------------------------------------------------------------
def prepare_ml_features(X: np.ndarray) -> np.ndarray:
    """
    Convert a list/array of MFCC matrices into fixed-length ML feature vectors.
    Applies `extract_statistical_features()` to each MFCC sample
    for compact, fixed-size numerical representation suitable for
    classical machine-learning models

    Returns
    -------
    np.ndarray
        2D array of shape (n_samples, n_features), where each row contains
        the statistical feature vector extracted from one MFCC sample.
    """
    logger.info("Extracting statistical features from MFCC...")
    features = []
    for mfcc in X:
        feat = extract_statistical_features(mfcc)
        features.append(feat)
    return np.array(features)


# def main():
#     load_dataset()
#     extract_statistical_features()
#     prepare_ml_features()

# if __name__ == "__main__":
#     main()
