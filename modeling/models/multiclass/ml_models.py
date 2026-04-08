#!/usr/bin/env python3
"""
Traditional ML Models for Audio Classification
----------------------------------------------

This script implements traditional ML models:
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Gradient Boosting

These models work directly on MFCC features.
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# -------------------------------------------------------------------------
# Paths (script in modeling/models/multiclass/; repo root = 3 levels up)
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent #multiclass/
REPO_ROOT = SCRIPT_DIR.parent.parent.parent #root/
CSV_DIR = REPO_ROOT / "datasets" / "trad_ml_csv"
TRAINED_MODEL_ROOT = REPO_ROOT / "trained_models"

#custom logger
sys.path.append(str(REPO_ROOT))
from modeling.utils.custom_logger import get_logger
logger = get_logger("ml_models_logger", "ml_models_training.log")

# Configuration
TEST_SPLIT=0.2
SHUFFLE=True
RANDOM_STATE=42

# -------------------------------------------------------------------------
# Dataset Loading

def load_dataset(
    csv_dir:str, 
    test_split:float, 
    shffl:bool, 
    rand_state:int
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

    class_map = {"BACKGROUND": 0, "HELICOPTER": 1, "DRONE": 2}
    data_by_class = {0: [], 1: [], 2: []}

    for file in csv_path.glob("*.csv"):
        name = file.stem.upper()
        if name.startswith("DRONE"):
            label = class_map["DRONE"]
        elif name.startswith("HELICOPTER"):
            label = class_map["HELICOPTER"]
        elif name.startswith("BACKGROUND"):
            label = class_map["BACKGROUND"]
        else:
            logger.warning(f"Skipping unknown file: {file.name}")
            continue
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
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    return X_train, y_train, X_val, y_val

# -------------------------------------------------------------------------
# Feature Extraction

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
# ML feature preparation
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

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Model Training
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

def train_random_forest(X_train, y_train, X_val, y_val) -> Dict:
    logger.info("\n" + "="*60)
    logger.info("Training Random Forest Classifier")
    logger.info("="*60)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return {
        'model': model,
        'name': 'RandomForest',
        'accuracy': accuracy,
        'predictions': y_pred
    }

def train_svm(X_train, y_train, X_val, y_val) -> Dict:
    logger.info("\n" + "="*60)
    logger.info("Training Support Vector Machine (SVM)")
    logger.info("="*60)
    
    model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'model': model,
        'name': 'SVM',
        'accuracy': accuracy,
        'predictions': y_pred
    }

def train_xgboost(X_train, y_train, X_val, y_val) -> Dict:
    logger.info("\n" + "="*60)
    logger.info("Training XGBoost Classifier")
    logger.info("="*60)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'model': model,
        'name': 'XGBoost',
        'accuracy': accuracy,
        'predictions': y_pred
    }

def train_gradient_boosting(X_train, y_train, X_val, y_val) -> Dict:
    logger.info("\n" + "="*60)
    logger.info("Training Gradient Boosting Classifier")
    logger.info("="*60)
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'model': model,
        'name': 'GradientBoosting',
        'accuracy': accuracy,
        'predictions': y_pred
    }

# -------------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------------

def main():
    logger.info("Starting Traditional ML Models Training...")
    
    # Load dataset
    csv_dir = str(CSV_DIR)
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(csv_dir=csv_dir,test_split=TEST_SPLIT,shffl=SHUFFLE, rand_state=RANDOM_STATE)
    
    # Extract features
    X_train = prepare_ml_features(X_train_raw)
    X_val = prepare_ml_features(X_val_raw)
    
    logger.info(f"Feature vector size: {X_train.shape[1]}")
    logger.info(f"Feature vector length: {X_train.shape[0]}")
    assert X_train.shape[1] == X_val.shape[1], "Feature dimensions mismatch!"
    
    # Standardize features
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Train all models
    results = []
    
    results.append(train_random_forest(X_train, y_train, X_val, y_val))
    results.append(train_svm(X_train, y_train, X_val, y_val))
    results.append(train_xgboost(X_train, y_train, X_val, y_val))
    results.append(train_gradient_boosting(X_train, y_train, X_val, y_val))
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    for result in results:
        logger.info(f"{result['name']}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    
    # Find best model
    best_result = max(results, key=lambda x: x['accuracy'])
    logger.info(f"\nBest Model: {best_result['name']} with {best_result['accuracy']*100:.2f}% accuracy")
    
    # Create trained_models/ml_models at repo root (only if not exists)
    model_dir = TRAINED_MODEL_ROOT / "ml_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = model_dir / f"best_ml_model_{best_result['name'].lower()}.pkl"
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_result['model'], f)
    logger.info(f"Best model saved to: {best_model_path}")
    
    scaler_path = model_dir / "feature_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Feature scaler saved to: {scaler_path}")
    
    # Save all models
    for result in results:
        model_path = model_dir / f"ml_model_{result['name'].lower()}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        logger.info(f"{result['name']} saved to: {model_path}")
    
    # Detailed report for best model
    logger.info("\n" + "="*60)
    logger.info(f"DETAILED REPORT FOR {best_result['name']}")
    logger.info("="*60)
    
    class_names = ['BACKGROUND', 'HELICOPTER', 'DRONE']
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, best_result['predictions'])
    logger.info(f"\n{cm}")
    
    logger.info("\nClassification Report:")
    report = classification_report(y_val, best_result['predictions'], target_names=class_names)
    logger.info(f"\n{report}")
    
    logger.info("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
