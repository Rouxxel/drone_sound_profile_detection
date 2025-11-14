#!/usr/bin/env python3
"""
Robust CNN Model Test Script
"""

import os
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    log_path = log_dir / "robust_cnn_testing.log"

    logger = logging.getLogger("robust_cnn_test_logger")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    handler.setFormatter(CustomFormatter("%(asctime)s | %(levelname)s | %(message)s"))

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(console_handler)

    logger.info("Log handler successfully set")
    logger.info(f"All logs are being saved in {log_path}")

    return logger

logger = setup_logging()

def load_dataset(csv_dir: str):
    csv_path = Path(csv_dir)
    assert csv_path.exists(), f"CSV directory not found: {csv_dir}"

    logger.info(f"Loading dataset from {csv_path}")

    class_map = {"BACKGROUND": 0, "HELICOPTER": 1, "DRONE": 2}
    class_names = {0: "BACKGROUND", 1: "HELICOPTER", 2: "DRONE"}
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
        logger.info(f"Class {class_names[label]}: {len(train_split)} train, {len(val_split)} validation")

    X_train, y_train = np.array(X_train, dtype=object), np.array(y_train)
    X_val, y_val = np.array(X_val, dtype=object), np.array(y_val)
    return X_train, y_train, X_val, y_val, class_names

def preprocess_data(X: np.ndarray) -> np.ndarray:
    max_frames = max(x.shape[0] for x in X)
    feature_dim = X[0].shape[1]
    X_out = np.zeros((len(X), max_frames, feature_dim), dtype=np.float32)
    for i, mfcc in enumerate(X):
        frames = mfcc.shape[0]
        X_out[i, :frames, :] = mfcc
    X_out = np.expand_dims(X_out, axis=-1)
    return X_out

def main():
    logger.info("Starting Robust CNN model testing...")
    
    model_path = Path("../trained_model/robust_cnn/robust_cnn_audio_model.pkl")
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please train the model first by running robust_cnn_model.py")
        return
    
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    
    csv_dir = "../../datasets/converted_csv"
    X_train_raw, y_train, X_val_raw, y_val, class_names = load_dataset(csv_dir)
    
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
