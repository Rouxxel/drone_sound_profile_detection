#!/usr/bin/env python3
"""
Tiny CNN Binary Audio Classifier Training Script
------------------------------------------------

This script loads audio files (.wav) for:
- Drone
- Helicopter
- Background

It computes log-mel spectrograms (32–64 Mel bands), normalizes them,
and trains a small CNN model optimized for low-resource environments.

Data is auto-discovered from:
    datasets/audios/*.wav

Each file must follow the naming format:
    DRONE_*.wav
    HELICOPTER_*.wav
    BACKGROUND_*.wav
"""

import os
import sys
import logging
import pickle
import librosa
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Suppress TensorFlow oneDNN info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------------------------------------------------
# main Paths
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent #multiclass/
REPO_ROOT = SCRIPT_DIR.parent.parent.parent #root/
AUDIO_FOLDER = REPO_ROOT / "datasets" / "audios"

#custom logger, config and utils
sys.path.append(str(REPO_ROOT))
from modeling.utils.custom_logger import get_logger
from configuration.config_loader import config
#from modeling.utils. import 
logger = get_logger("tiny_cnn_logger", "tiny_cnn_training.log")

# General Configuration
TRAINED_MODEL_ROOT = REPO_ROOT / config["cnn_models"]["output_folder_str"]
SPECIFIC_FOLDER = TRAINED_MODEL_ROOT / config["cnn_models"]["multiclass"]["general"]["folder"]

#Multiclass tiny cnn config
N_MELS = config["cnn_models"]["multiclass"]["tiny_cnn"]["n_mels"]
CONV1_FILT= config["cnn_models"]["multiclass"]["tiny_cnn"]["conv1_filters"]
CONV1_KRNL= config["cnn_models"]["multiclass"]["tiny_cnn"]["conv1_kernel"]
CONV2_FILT= config["cnn_models"]["multiclass"]["tiny_cnn"]["conv2_filters"]
CONV2_KRNL= config["cnn_models"]["multiclass"]["tiny_cnn"]["conv2_kernel"]
DENSE_UNITS= config["cnn_models"]["multiclass"]["tiny_cnn"]["dense_units"]
DROPOUT= config["cnn_models"]["multiclass"]["tiny_cnn"]["dropout"]
LEARNING_RATE= config["cnn_models"]["multiclass"]["tiny_cnn"]["learning_rate"]
ACTIVATION= config["cnn_models"]["multiclass"]["tiny_cnn"]["activation"]
PADDING= config["cnn_models"]["multiclass"]["tiny_cnn"]["padding"]
LOSS= config["cnn_models"]["multiclass"]["tiny_cnn"]["loss"]
METRICS= config["cnn_models"]["multiclass"]["tiny_cnn"]["metrics"]
N_CLASSES = config["cnn_models"]["multiclass"]["general"]["n_classes"]
CLASS_MAP = config["cnn_models"]["multiclass"]["general"]["class_map"]

# -------------------------------------------------------------------------
# Data Loading and preprocess
# -------------------------------------------------------------------------
def compute_logmel(audio_path: Path, n_mels: int = 64) -> np.ndarray:
    """Load audio → compute log-mel → normalize → return (time, mel_bins)."""
    y, sr = librosa.load(str(audio_path), sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)
    return logmel.T  #CNN expects = (time_steps, mel_bins)

def load_dataset(audio_dir: str, class_map: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    audio_path = Path(audio_dir)
    assert audio_path.exists(), f"Audio directory not found: {audio_dir}"

    logger.info(f"Dataset successfully found and loaded from {audio_path}")

    data_by_class = {0: [], 1: [], 2: []}

    # Load .wav files, NOT CSVs
    for file in audio_path.glob("*.wav"):
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

        logmel = compute_logmel(audio_path=file,n_mels=N_MELS)
        data_by_class[label].append(logmel)
        logger.info(f"Loaded {file.name} -> class {label}, shape={logmel.shape}")

    # Split per class to maintain balance
    X_train, y_train, X_val, y_val = [], [], [], []

    for label, samples in data_by_class.items():
        if len(samples) == 0:
            logger.error(f"No samples found for class {label}")
            continue

        train_split, val_split = train_test_split(
            samples, test_size=0.2, shuffle=True, random_state=42
        )

        X_train.extend(train_split)
        y_train.extend([label] * len(train_split))
        X_val.extend(val_split)
        y_val.extend([label] * len(val_split))

        logger.info(f"Class {label}: {len(train_split)} train, {len(val_split)} validation")

    X_train, y_train = np.array(X_train, dtype=object), np.array(y_train)
    X_val, y_val = np.array(X_val, dtype=object), np.array(y_val)
    return X_train, y_train, X_val, y_val

def preprocess_data(X: np.ndarray) -> np.ndarray:
    """
        Pad variable-length log-mel spectrograms to a uniform time dimension
        and add a channel axis for CNN input.
    """
    max_frames = max(sample.shape[0] for sample in X)
    feature_dim = X[0].shape[1]

    X_out = np.zeros((len(X), max_frames, feature_dim), dtype=np.float32)

    for i, spec in enumerate(X):
        frames = spec.shape[0]
        X_out[i, :frames, :] = spec

    # Add channel dimension for CNN: (samples, time, mel_bins, 1)
    X_out = np.expand_dims(X_out, axis=-1)

    return X_out

# -------------------------------------------------------------------------
# Model Definition
# -------------------------------------------------------------------------

def build_tiny_cnn(input_shape: Tuple[int, int, int], num_classes: int = 3) -> models.Sequential:
    """
    Build a lightweight CNN for audio classification using log-mel spectrogram inputs.

    Returns
    -------
    keras.models.Sequential
        A compiled tiny CNN model suitable for low-resource deployment.
    """
    logger.info("Building Tiny CNN model...")
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # First conv block
        layers.Conv2D(CONV1_FILT, tuple(CONV1_KRNL), activation=ACTIVATION, padding=PADDING),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Second conv block
        layers.Conv2D(CONV2_FILT, tuple(CONV2_KRNL), activation=ACTIVATION, padding=PADDING),
        layers.BatchNormalization(),

        # Global pooling + dense
        layers.GlobalAveragePooling2D(),
        layers.Dense(DENSE_UNITS, activation=ACTIVATION),
        layers.Dropout(DROPOUT),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss=LOSS,
        metrics=['accuracy']
        #metrics=METRICS TODO: solve other metrics capabilities
    )
    logger.info("Model built successfully")
    return model

# -------------------------------------------------------------------------
# Plotting Function
# -------------------------------------------------------------------------

def plot_training(history):
    """
    Plot training and validation accuracy and loss.

    Saves
    -----
    PNG plot of training metrics to
    """
    metrics = [k for k in history.keys() if not k.startswith("val_")]

    n_metrics = len(metrics)
    plt.figure(figsize=(5*n_metrics, 4))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, n_metrics, i)
        plt.plot(history[metric], label=f"Train {metric.capitalize()}")
        val_metric = f"val_{metric}"
        if val_metric in history:
            plt.plot(history[val_metric], label=f"Val {metric.capitalize()}")
        plt.title(metric.capitalize())
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(TRAINED_MODEL_ROOT / SPECIFIC_FOLDER / "training_history_tiny_cnn.png")
    logger.info("Training plot saved to training_history_tiny_cnn.png")

# -------------------------------------------------------------------------
# Main Training Pipeline
# -------------------------------------------------------------------------

def main():
    logger.info("Starting Tiny CNN training...")
    aud_dir = str(AUDIO_FOLDER)
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(audio_dir=aud_dir,class_map=CLASS_MAP)
    X_train = preprocess_data(X_train_raw)
    X_val = preprocess_data(X_val_raw)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Input shape to model: {X_train.shape[1:]}")

    model = build_tiny_cnn(input_shape=X_train.shape[1:], num_classes=N_CLASSES)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=8,
                        verbose=1).history

    # Create trained_model/tiny_cnn directory at repo root
    model_dir = TRAINED_MODEL_ROOT / SPECIFIC_FOLDER
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created model directory: {model_dir}")
    else:
        logger.info(f"Model directory already exists: {model_dir}")
    
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
