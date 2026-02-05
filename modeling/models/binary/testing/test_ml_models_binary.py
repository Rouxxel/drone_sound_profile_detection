#!/usr/bin/env python3
"""
Binary ML Models Test Script
----------------------------
Tests binary classification: DRONE vs NO_DRONE
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

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
    log_path = log_dir / "ml_models_binary_testing.log"

    logger = logging.getLogger("ml_binary_test_logger")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    handler.setFormatter(CustomFormatter("%(asctime)s | %(levelname)s | %(message)s"))

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(CustomFormatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(console)

    logger.info("Log handler successfully set")
    logger.info(f"All logs are being saved in {log_path}")

    return logger

logger = setup_logging()

def load_dataset(csv_dir: str):
    csv_path = Path(csv_dir)
    assert csv_path.exists(), f"CSV directory not found: {csv_dir}"

    logger.info(f"Loading dataset from {csv_path}")

    class_names = {0: "NO_DRONE", 1: "DRONE"}
    data_by_class = {0: [], 1: []}

    for file in csv_path.glob("*.csv"):
        name = file.stem.upper()
        if name.startswith("DRONE"):
            label = 1
        elif name.startswith("HELICOPTER") or name.startswith("BACKGROUND"):
            label = 0
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
    return X_train, y_train, X_val, y_val, class_names

def extract_statistical_features(mfcc: np.ndarray) -> np.ndarray:
    mfcc = np.asarray(mfcc, dtype=np.float32)
    features = []
    
    features.extend(np.mean(mfcc, axis=0))
    features.extend(np.std(mfcc, axis=0))
    features.extend(np.min(mfcc, axis=0))
    features.extend(np.max(mfcc, axis=0))
    features.extend(np.median(mfcc, axis=0))
    features.extend(np.percentile(mfcc, 25, axis=0))
    features.extend(np.percentile(mfcc, 75, axis=0))
    
    delta = np.diff(mfcc, axis=0)
    if len(delta) > 0:
        features.extend(np.mean(delta, axis=0))
        features.extend(np.std(delta, axis=0))
    else:
        features.extend(np.zeros(mfcc.shape[1]))
        features.extend(np.zeros(mfcc.shape[1]))
    
    return np.array(features, dtype=np.float32)

def prepare_ml_features(X: np.ndarray) -> np.ndarray:
    logger.info("Extracting statistical features from MFCC...")
    features = []
    for mfcc in X:
        feat = extract_statistical_features(mfcc)
        features.append(feat)
    return np.array(features)

def test_model(model, model_name: str, X_val, y_val, class_names: Dict):
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name}")
    logger.info('='*60)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = confusion_matrix(y_val, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    
    logger.info("\nClassification Report:")
    report = classification_report(y_val, y_pred, 
                                   target_names=[class_names[i] for i in sorted(class_names.keys())])
    logger.info(f"\n{report}")
    
    logger.info("\nPer-class Accuracy:")
    for i in sorted(class_names.keys()):
        class_mask = y_val == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            logger.info(f"{class_names[i]}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    return {
        'name': model_name,
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm
    }

def main():
    logger.info("Starting Binary ML Models Testing...")
    
    model_dir = Path("../../trained_model/binary/ml_models")
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        logger.error("Please train the models first by running ml_models_binary.py")
        return
    
    scaler_path = model_dir / "feature_scaler_binary.pkl"
    if not scaler_path.exists():
        logger.error(f"Feature scaler not found: {scaler_path}")
        logger.error("Please train the models first by running ml_models_binary.py")
        return

    logger.info(f"Loading feature scaler from {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    csv_dir = "../../../datasets/converted_csv"
    X_train_raw, y_train, X_val_raw, y_val, class_names = load_dataset(csv_dir)
    
    X_val = prepare_ml_features(X_val_raw)
    X_val = scaler.transform(X_val)
    
    logger.info(f"Feature vector size: {X_val.shape[1]}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    model_files = {
        'RandomForest': model_dir / "ml_model_binary_randomforest.pkl",
        'SVM': model_dir / "ml_model_binary_svm.pkl",
        'XGBoost': model_dir / "ml_model_binary_xgboost.pkl",
        'GradientBoosting': model_dir / "ml_model_binary_gradientboosting.pkl"
    }
    
    results = []
    
    for model_name, model_path in model_files.items():
        if model_path.exists():
            logger.info(f"\nLoading {model_name} from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            result = test_model(model, model_name, X_val, y_val, class_names)
            results.append(result)
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    if results:
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        
        for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
            logger.info(f"{result['name']:<20}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        
        best_result = max(results, key=lambda x: x['accuracy'])
        logger.info(f"\nBest Model: {best_result['name']} with {best_result['accuracy']*100:.2f}% accuracy")
    
    logger.info("\nTesting completed successfully!")

if __name__ == "__main__":
    main()
