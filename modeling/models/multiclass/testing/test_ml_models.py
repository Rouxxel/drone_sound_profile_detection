#!/usr/bin/env python3
"""
Traditional ML Models Test Script
---------------------------------

This script tests trained ML models (Random Forest, SVM, XGBoost, Gradient Boosting)
on the validation set.
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths (script in modeling/models/multiclass/testing/; repo root = 4 levels up)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
CSV_DIR = REPO_ROOT / "datasets" / "trad_ml_csv" #adjust for testing
TRAINED_MODEL_ROOT = REPO_ROOT / "trained_models" / "multiclass" #adjust for testing
CLASS_NAMES = {0: "BACKGROUND", 1: "HELICOPTER", 2: "DRONE"}

# Logging (shared format; log file: logs/ml_models_binary_testing.log)
sys.path.append(str(REPO_ROOT))
from modeling.utils.custom_logger import get_logger
from modeling.utils.ml_train_methods import load_dataset, extract_statistical_features, prepare_ml_features
logger = get_logger("ml_test_logger", "ml_models_testing.log")

# -------------------------------------------------------------------------
# Model Testing
# -------------------------------------------------------------------------
def test_model(model, model_name: str, X_val, y_val, class_names: Dict):
    """Test a single model and return results"""
    logger.info('='*60)
    logger.info(f"Testing {model_name}")
    logger.info('='*60)
    
    # Predictions
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    
    # Classification Report
    logger.info("Classification Report:")
    report = classification_report(y_val, y_pred,target_names=[class_names[i] for i in sorted(class_names.keys())])
    logger.info(f"\n{report}")
    
    # Per-class accuracy
    logger.info("Per-class Accuracy:")
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

# -------------------------------------------------------------------------
# Main Testing Pipeline
# -------------------------------------------------------------------------

def main():
    logger.info("Starting ML Models Testing...")
    
    # Check for trained models
    model_dir = TRAINED_MODEL_ROOT
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        logger.error("Please train the models first by running ml_models.py")
        return
    
    # Check for scaler
    scaler_path = model_dir / "feature_scaler.pkl"
    if not scaler_path.exists():
        logger.error(f"Feature scaler not found: {scaler_path}")
        logger.error("Please train the models first by running ml_models.py")
        return

    # Load scaler
    logger.info(f"Loading feature scaler from {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load dataset
    csv_dir = str(CSV_DIR)
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(csv_dir)
    
    # Extract and scale features
    X_val = prepare_ml_features(X_val_raw)
    X_val = scaler.transform(X_val)
    
    logger.info(f"Feature vector size: {X_val.shape[1]}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Find all trained ML models
    model_files = {
        'RandomForest': model_dir / "ml_model_randomforest.pkl",
        'SVM': model_dir / "ml_model_svm.pkl",
        'XGBoost': model_dir / "ml_model_xgboost.pkl",
        'GradientBoosting': model_dir / "ml_model_gradientboosting.pkl"
    }
    
    results = []
    
    # Test each model
    for model_name, model_path in model_files.items():
        if model_path.exists():
            logger.info(f"Loading {model_name} from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            result = test_model(model=model, model_name=model_name, X_val=X_val, y_val=y_val, class_names=CLASS_NAMES)
            results.append(result)
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    # Summary comparison
    if results:
        logger.info("="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        
        for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
            logger.info(f"{result['name']:<20}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        
        best_result = max(results, key=lambda x: x['accuracy'])
        logger.info(f"Best Model: {best_result['name']} with {best_result['accuracy']*100:.2f}% accuracy")
    
    logger.info("Testing completed successfully!")

if __name__ == "__main__":
    main()
