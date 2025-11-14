#!/usr/bin/env python3
"""
Traditional ML Models for Audio Classification
----------------------------------------------

This script implements traditional ML models:
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Gradient Boosting

These models work directly on flattened MFCC features.
"""

import os
import pickle
import logging
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
# Logging Setup
# -------------------------------------------------------------------------

class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        t = "%04d-%02d-%02d %02d:%02d:%02d,%03d" % (
            ct.tm_year, ct.tm_mon, ct.tm_mday,
            ct.tm_hour, ct.tm_min, ct.tm_sec,
            int(record.created * 1000) % 1000
        )
        return t

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "ml_models_training.log"

    logger = logging.getLogger("ml_models_logger")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path, mode='w')
    handler.setFormatter(CustomFormatter("%(asctime)s | %(levelname)s | %(message)s"))

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(CustomFormatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(console)

    logger.info("Log handler successfully set")
    logger.info(f"All logs are being saved in {log_path}")

    return logger

logger = setup_logging()

# -------------------------------------------------------------------------
# Dataset Loading
# -------------------------------------------------------------------------

def load_dataset(csv_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        train_split, val_split = train_test_split(samples, test_size=0.2, shuffle=True, random_state=42)
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
# -------------------------------------------------------------------------

def extract_statistical_features(mfcc: np.ndarray) -> np.ndarray:
    """Extract statistical features from MFCC"""
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
    
    return np.array(features)

def prepare_ml_features(X: np.ndarray) -> np.ndarray:
    """Convert MFCC arrays to feature vectors for ML models"""
    logger.info("Extracting statistical features from MFCC...")
    features = []
    for mfcc in X:
        feat = extract_statistical_features(mfcc)
        features.append(feat)
    return np.array(features)

# -------------------------------------------------------------------------
# Model Training
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
    csv_dir = "../datasets/converted_csv"
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(csv_dir)
    
    # Extract features
    X_train = prepare_ml_features(X_train_raw)
    X_val = prepare_ml_features(X_val_raw)
    
    logger.info(f"Feature vector size: {X_train.shape[1]}")
    
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
    
    # Save best model and scaler
    model_dir = Path("trained_model")
    model_dir.mkdir(exist_ok=True)
    
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
