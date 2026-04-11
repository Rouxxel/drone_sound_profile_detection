#!/usr/bin/env python3
"""
Robust CNN Model Test Script
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
from modeling.utils.cnn_train_methods import load_dataset, preprocess_data
logger = get_logger("test_cnn_models_logger", "testing_cnn_models_multiclass.log")

# General Configuration
TRAINED_MODEL_ROOT = REPO_ROOT / config["cnn_models"]["output_folder_str"]
SPECIFIC_FOLDER = TRAINED_MODEL_ROOT / config["cnn_models"]["multiclass"]["general"]["folder"]
CLASS_NAMES = {"BACKGROUND": 0, "HELICOPTER": 1, "DRONE": 2} #change as necessary for testing

# Logging (shared format; log file: logs/tiny_cnn_testing.log)
from modeling.utils.custom_logger import get_logger
logger = get_logger("tiny_cnn_test_logger", "tiny_cnn_testing.log")

# -------------------------------------------------------------------------
# Main Testing Pipeline
# -------------------------------------------------------------------------
def main():
    logger.info("Starting Robust CNN model testing...")
    
    model_path = TRAINED_MODEL_ROOT / SPECIFIC_FOLDER / "robust_cnn_audio_model.pkl"
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please train the model first by running robust_cnn_model.py")
        return
    
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    
    aud_dir = str(AUDIO_FOLDER)
    class_names = {v: k for k, v in CLASS_NAMES.items()}
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(audio_dir=aud_dir,class_map=CLASS_NAMES, n_mels=64, is_binary=False)
    
    X_val = preprocess_data(X_val_raw)
    
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Input shape: {X_val.shape[1:]}")
    
    logger.info("Evaluating model on validation set...")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    logger.info(f"Validation Loss: {loss:.4f}")
    logger.info(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_val, y_pred)
    logger.info(f"Accuracy Score: {acc:.4f} ({acc*100:.2f}%)")
    
    cm = confusion_matrix(y_val, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    
    logger.info("\nClassification Report:")
    report = classification_report(y_val, y_pred, target_names=[class_names[i] for i in sorted(class_names.keys())])
    logger.info(f"\n{report}")
    
    logger.info("\nPer-class Accuracy:")
    for i in sorted(class_names.keys()):
        class_mask = y_val == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            logger.info(f"{class_names[i]}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    logger.info("\nTesting completed successfully!")

if __name__ == "__main__":
    main()
