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

#custom logger
sys.path.append(str(REPO_ROOT))
from modeling.utils.custom_logger import get_logger
from configuration.config_loader import config
from modeling.utils.ml_train_methods import extract_statistical_features, load_dataset, prepare_ml_features
logger = get_logger("ml_models_logger", "ml_models_training.log")

# Configuration
CSV_DIR = REPO_ROOT / "datasets" / config["audio_converters"]["trad_ml_models"]["output_folder_str"]
TRAINED_MODEL_ROOT = REPO_ROOT / config["ml_models"]["output_folder_str"]
SPECIFIC_FOLDER = TRAINED_MODEL_ROOT / config["ml_models"]["multiclass"]["general"]["folder"]
TEST_SPLIT=config["ml_models"]["multiclass"]["general"]["test_split"]
SHUFFLE=config["ml_models"]["multiclass"]["general"]["shuffle"]
RANDOM_STATE=config["ml_models"]["multiclass"]["general"]["random_state"]
CLASSES=config["ml_models"]["multiclass"]["general"]["classes"]

#Random forest
RF_N_ESTIMATORS=config["ml_models"]["multiclass"]["random_forest"]["n_estimators"]
RF_MAX_DEPTH=config["ml_models"]["multiclass"]["random_forest"]["max_depth"]
MIN_SAMP_SPLT=config["ml_models"]["multiclass"]["random_forest"]["min_samples_split"]
MIN_SAMP_LEAF=config["ml_models"]["multiclass"]["random_forest"]["min_samples_leaf"]
RF_RNDM_STATE=config["ml_models"]["multiclass"]["random_forest"]["random_state"]
RF_N_JOBS=config["ml_models"]["multiclass"]["random_forest"]["n_jobs"]
RF_VERBOSE=config["ml_models"]["multiclass"]["random_forest"]["verbose"]

#SVM
GAMMA=config["ml_models"]["multiclass"]["svm"]["gamma"]
SVM_C=config["ml_models"]["multiclass"]["svm"]["c"]
PROBABILITY=config["ml_models"]["multiclass"]["svm"]["probability"]
KERNEL=config["ml_models"]["multiclass"]["svm"]["kernel"]
SVM_RNDM_STATE=config["ml_models"]["multiclass"]["svm"]["random_state"]
SVM_VERBOSE=config["ml_models"]["multiclass"]["svm"]["verbose"]

#XGboost
XG_N_ESTIMATORS=config["ml_models"]["multiclass"]["xgboost"]["n_estimators"]
XG_MAX_DEPTH=config["ml_models"]["multiclass"]["xgboost"]["max_depth"]
XG_LEARNING_RATE=config["ml_models"]["multiclass"]["xgboost"]["learning_rate"]
XG_SUBSAMPLE=config["ml_models"]["multiclass"]["xgboost"]["subsample"]
XG_COLSAMPLE_BYTREE=config["ml_models"]["multiclass"]["xgboost"]["colsample_bytree"]
XG_RNDM_STATE=config["ml_models"]["multiclass"]["xgboost"]["random_state"]
XG_N_JOBS=config["ml_models"]["multiclass"]["xgboost"]["n_jobs"]
XG_VERBOSE=config["ml_models"]["multiclass"]["xgboost"]["verbose"]

#Gradient boosting
GB_N_ESTIMATORS=config["ml_models"]["multiclass"]["gradient"]["n_estimators"]
GB_MAX_DEPTH=config["ml_models"]["multiclass"]["gradient"]["max_depth"]
GB_LEARNING_RATE=config["ml_models"]["multiclass"]["gradient"]["learning_rate"]
GB_SUBSAMPLE=config["ml_models"]["multiclass"]["gradient"]["subsample"]
GB_RNDM_STATE=config["ml_models"]["multiclass"]["gradient"]["random_state"]
GB_VERBOSE=config["ml_models"]["multiclass"]["gradient"]["verbose"]

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Model Training
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

def train_random_forest(X_train, y_train, X_val, y_val) -> Dict:
    logger.info("="*60)
    logger.info("Training Random Forest Classifier")
    logger.info("="*60)
    
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=MIN_SAMP_SPLT,
        min_samples_leaf=MIN_SAMP_LEAF,
        random_state=RF_RNDM_STATE,
        n_jobs=RF_N_JOBS,
        verbose=RF_VERBOSE
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
    logger.info("="*60)
    logger.info("Training Support Vector Machine (SVM)")
    logger.info("="*60)
    
    model = SVC(
        kernel=KERNEL,
        C=SVM_C,
        probability=PROBABILITY,
        gamma=GAMMA,
        random_state=SVM_RNDM_STATE,
        verbose=SVM_VERBOSE
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
    logger.info("="*60)
    logger.info("Training XGBoost Classifier")
    logger.info("="*60)
    
    model = xgb.XGBClassifier(
        n_estimators=XG_N_ESTIMATORS,
        max_depth=XG_MAX_DEPTH,
        learning_rate=XG_LEARNING_RATE,
        subsample=XG_SUBSAMPLE,
        colsample_bytree=XG_COLSAMPLE_BYTREE,
        random_state=XG_RNDM_STATE,
        n_jobs=XG_N_JOBS,
        verbosity=XG_VERBOSE
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
    logger.info("="*60)
    logger.info("Training Gradient Boosting Classifier")
    logger.info("="*60)
    
    model = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE,
        subsample=GB_SUBSAMPLE,
        random_state=GB_RNDM_STATE,
        verbose=GB_VERBOSE
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
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(csv_dir=csv_dir,test_split=TEST_SPLIT,shffl=SHUFFLE, rand_state=RANDOM_STATE, is_binary=False)
    
    # Extract features
    X_train = prepare_ml_features(X_train_raw)
    X_val = prepare_ml_features(X_val_raw)
    
    logger.info(f"Feature vector size: {X_train.shape[1]}")
    logger.info(f"Feature vector length: {X_train.shape[1]}")
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
    logger.info("="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    for result in results:
        logger.info(f"{result['name']}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    
    # Find best model
    best_result = max(results, key=lambda x: x['accuracy'])
    logger.info(f"Best Model: {best_result['name']} with {best_result['accuracy']*100:.2f}% accuracy")
    
    # Create trained_models/ml_models at repo root (only if not exists)
    model_dir = TRAINED_MODEL_ROOT / SPECIFIC_FOLDER
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
    logger.info("="*60)
    logger.info(f"DETAILED REPORT FOR {best_result['name']}")
    logger.info("="*60)
    
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_val, best_result['predictions'])
    logger.info(f"\n{cm}")
    
    logger.info("Classification Report:")
    report = classification_report(y_val, best_result['predictions'], target_names=CLASSES)
    logger.info(f"\n{report}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
