#!/usr/bin/env python3
"""
Binary ML Models Test Script
----------------------------
Tests binary classification: DRONE vs NO_DRONE
"""

import sys
import pickle
from pathlib import Path
from typing import Dict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------------------------------------------------
# main Paths
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent #testing/
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent #root/
AUDIO_FOLDER = REPO_ROOT / "datasets" / "audios"

#custom logger, config and utils
sys.path.append(str(REPO_ROOT))
from modeling.utils.custom_logger import get_logger
from configuration.config_loader import config
from modeling.utils.ml_train_methods import load_dataset, prepare_ml_features
logger = get_logger("test_ml_models_logger", "testing_ml_models.log")

# General Configuration
TRAINED_MODEL_ROOT = REPO_ROOT / config["ml_models"]["output_folder_str"]
SPECIFIC_FOLDER = TRAINED_MODEL_ROOT / config["ml_models"]["binary"]["general"]["folder"]
CSV_DIR = REPO_ROOT / "datasets" / config["audio_converters"]["trad_ml_models"]["output_folder_str"] #change as necessary for testing
CLASS_NAMES = {0: "NO_DRONE", 1: "DRONE"} #change as necessary for testing

# -------------------------------------------------------------------------
# Model Testing
# -------------------------------------------------------------------------
def test_model(model, model_name: str, X_val, y_val, class_names: Dict):
    logger.info('='*60)
    logger.info(f"Testing {model_name}")
    logger.info('='*60)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = confusion_matrix(y_val, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    
    logger.info("Classification Report:")
    report = classification_report(
        y_val, y_pred,
        target_names=[class_names[i] for i in sorted(class_names.keys())]
        )
    logger.info(f"\n{report}")
    
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

def main():
    logger.info("Starting Binary ML Models Testing...")
    
    model_dir = TRAINED_MODEL_ROOT / SPECIFIC_FOLDER
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
    
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(csv_dir=CSV_DIR,is_binary=True)
    
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
            logger.info(f"Loading {model_name} from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            result = test_model(model=model, model_name=model_name, X_val=X_val, y_val=y_val, class_names=CLASS_NAMES)
            results.append(result)
        else:
            logger.warning(f"Model file not found: {model_path}")
    
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
