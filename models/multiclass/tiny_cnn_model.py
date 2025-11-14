#!/usr/bin/env python3
"""
Tiny CNN Audio Classifier Training Script
-----------------------------------------

This script loads MFCC CSV features for:
- Drone
- Helicopter
- Background

It builds and trains a small CNN model optimized for low-resource environments.

Data is auto-discovered from:
    datasets/converted_csv/*.csv

Each file must follow the naming format:
    DRONE_*.csv
    HELICOPTER_*.csv
    BACKGROUND_*.csv
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Suppress TensorFlow oneDNN info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    log_path = log_dir / "tiny_cnn_training.log"

    logger = logging.getLogger("tiny_cnn_logger")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    handler.setFormatter(CustomFormatter("%(asctime)s | %(levelname)s | %(message)s"))

    # Remove old handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

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

    logger.info(f"Dataset successfully found and loaded from {csv_path}")

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
        logger.info(f"Loaded {file.name} -> class {label}")

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
        logger.info(f"Class {label}: {len(train_split)} train, {len(val_split)} validation")

    X_train, y_train = np.array(X_train, dtype=object), np.array(y_train)
    X_val, y_val = np.array(X_val, dtype=object), np.array(y_val)
    return X_train, y_train, X_val, y_val

# -------------------------------------------------------------------------
# Data Preprocessing
# -------------------------------------------------------------------------

def preprocess_data(X: np.ndarray) -> np.ndarray:
    max_frames = max(x.shape[0] for x in X)
    feature_dim = X[0].shape[1]
    X_out = np.zeros((len(X), max_frames, feature_dim), dtype=np.float32)
    for i, mfcc in enumerate(X):
        frames = mfcc.shape[0]
        X_out[i, :frames, :] = mfcc
    X_out = np.expand_dims(X_out, axis=-1)
    return X_out

# -------------------------------------------------------------------------
# Model Definition
# -------------------------------------------------------------------------

def build_tiny_cnn(input_shape: Tuple[int, int, int], num_classes: int = 3) -> models.Sequential:
    logger.info("Building Tiny CNN model...")
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(1e-3),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    logger.info("Model built successfully")
    return model

# -------------------------------------------------------------------------
# Plotting Function
# -------------------------------------------------------------------------

def plot_training(history):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history["accuracy"], label="Train Acc")
    plt.plot(history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")
    plt.subplot(1,2,2)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig("trained_model/tiny_cnn/training_history_tiny_cnn.png")
    logger.info("Training plot saved to training_history_tiny_cnn.png")

# -------------------------------------------------------------------------
# Main Training Pipeline
# -------------------------------------------------------------------------

def main():
    logger.info("Starting Tiny CNN training...")
    csv_dir = "../../datasets/converted_csv"
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(csv_dir)
    X_train = preprocess_data(X_train_raw)
    X_val = preprocess_data(X_val_raw)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Input shape to model: {X_train.shape[1:]}")

    model = build_tiny_cnn(input_shape=X_train.shape[1:], num_classes=3)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=8,
                        verbose=1).history

    # Create trained_models/tiny_cnn directory
    model_dir = Path("trained_model") / "tiny_cnn"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model as pickle
    model_path = model_dir / "tiny_cnn_audio_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to: {model_path}")
    
    # Also save as h5 for compatibility
    h5_path = model_dir / "tiny_cnn_audio_model.h5"
    model.save(h5_path)
    logger.info(f"Model also saved as H5 to: {h5_path}")

    plot_training(history)
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
