#!/usr/bin/env python3
"""
Robust CNN Audio Classifier Training Script
-------------------------------------------

This script implements a more robust CNN architecture with:
- Deeper network with more convolutional layers
- Data augmentation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Cross-validation support

Data is auto-discovered from:
    datasets/converted_csv/*.csv
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from keras import layers, models, optimizers, callbacks
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
    log_path = log_dir / "robust_cnn_training.log"

    logger = logging.getLogger("robust_cnn_logger")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    handler.setFormatter(CustomFormatter("%(asctime)s | %(levelname)s | %(message)s"))

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
# Data Preprocessing & Augmentation
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

def augment_data(X: np.ndarray, y: np.ndarray, augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment data with time shifting and noise addition
    """
    logger.info(f"Augmenting data with factor {augmentation_factor}...")
    X_aug, y_aug = [], []
    
    for i in range(len(X)):
        # Original sample
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Augmented samples
        for _ in range(augmentation_factor - 1):
            sample = X[i].copy()
            
            # Time shift
            shift = np.random.randint(-10, 10)
            if shift != 0:
                sample = np.roll(sample, shift, axis=0)
            
            # Add noise
            noise = np.random.normal(0, 0.005, sample.shape)
            sample = sample + noise
            
            X_aug.append(sample)
            y_aug.append(y[i])
    
    logger.info(f"Data augmented: {len(X)} -> {len(X_aug)} samples")
    return np.array(X_aug), np.array(y_aug)

# -------------------------------------------------------------------------
# Model Definition
# -------------------------------------------------------------------------

def build_robust_cnn(input_shape: Tuple[int, int, int], num_classes: int = 3) -> models.Sequential:
    logger.info("Building Robust CNN model...")
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Conv Block
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        
        # Dense Layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Model built successfully")
    logger.info(f"Total parameters: {model.count_params():,}")
    return model

# -------------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------------

def get_callbacks(model_dir: Path):
    """Setup training callbacks"""
    
    # Early stopping
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Model checkpoint
    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(model_dir / 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    return [early_stop, reduce_lr, checkpoint]

# -------------------------------------------------------------------------
# Plotting Function
# -------------------------------------------------------------------------

def plot_training(history, save_path: str = "trained_model/robust_cnn/training_history_robust_cnn.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history["accuracy"], label="Train Acc", linewidth=2)
    axes[0, 0].plot(history["val_accuracy"], label="Val Acc", linewidth=2)
    axes[0, 0].set_title("Model Accuracy", fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history["loss"], label="Train Loss", linewidth=2)
    axes[0, 1].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0, 1].set_title("Model Loss", fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if "lr" in history:
        axes[1, 0].plot(history["lr"], linewidth=2, color='green')
        axes[1, 0].set_title("Learning Rate", fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy difference
    acc_diff = np.array(history["accuracy"]) - np.array(history["val_accuracy"])
    axes[1, 1].plot(acc_diff, linewidth=2, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title("Overfitting Monitor (Train - Val Acc)", fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy Difference")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Training plot saved to {save_path}")

# -------------------------------------------------------------------------
# Main Training Pipeline
# -------------------------------------------------------------------------

def main():
    logger.info("Starting Robust CNN training...")
    
    # Load dataset
    csv_dir = "../datasets/converted_csv"
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(csv_dir)
    
    # Preprocess
    X_train = preprocess_data(X_train_raw)
    X_val = preprocess_data(X_val_raw)
    
    # Augment training data
    X_train, y_train = augment_data(X_train, y_train, augmentation_factor=3)
    
    logger.info(f"Training samples (after augmentation): {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Input shape to model: {X_train.shape[1:]}")
    
    # Build model
    model = build_robust_cnn(input_shape=X_train.shape[1:], num_classes=3)
    
    # Create trained_model/robust_cnn directory
    model_dir = Path("trained_model") / "robust_cnn"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup callbacks
    callback_list = get_callbacks(model_dir)
    
    # Train model
    logger.info("Starting training with callbacks...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=callback_list,
        verbose=1
    ).history
    
    # Save final model
    model_path = model_dir / "robust_cnn_audio_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to: {model_path}")
    
    # Also save as h5
    h5_path = model_dir / "robust_cnn_audio_model.h5"
    model.save(h5_path)
    logger.info(f"Model also saved as H5 to: {h5_path}")
    
    # Plot training history
    plot_training(history)
    
    # Final evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Final Validation Loss: {val_loss:.4f}")
    logger.info(f"Final Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
