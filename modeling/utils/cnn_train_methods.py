#!/usr/bin/env python3
"""
CNN Models util methods for Audio Classification
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import librosa
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent #utils/
REPO_ROOT = SCRIPT_DIR.parent.parent.parent #root/

#custom logger
sys.path.append(str(REPO_ROOT))
#from custom_logger import get_logger #local testing only
from modeling.utils.custom_logger import get_logger
logger = get_logger("cnn_logger", "cnn_models_training.log")

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

def load_dataset(
    audio_dir: str,
    class_map: Dict[str, int],
    n_mels: int = 64,
    is_binary: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    audio_path = Path(audio_dir)
    assert audio_path.exists(), f"Audio directory not found: {audio_dir}"

    logger.info(f"Dataset successfully found and loaded from {audio_dir}")

    # Prepare data container automatically
    class_ids = sorted(set(class_map.values()))
    data_by_class = {cid: [] for cid in class_ids}

    logger.info(f"{'Binary' if is_binary else 'Multiclass'} classification active. Classes: {class_map}")

    # Extract class ids if its binary
    if is_binary:
        drone_id     = class_map["DRONE"]
        no_drone_id  = class_map["NO_DRONE"]
    
    # Load .wav files
    for file in audio_path.glob("*.wav"):
        name = file.stem.upper()

        # BINARY CLASSIFICATION
        if is_binary:
            if name.startswith("DRONE"):
                label = drone_id
            else:
                # Everything else is NO-DRONE
                label = no_drone_id

        # MULTICLASS CLASSIFICATION
        else:
            if name.startswith("BACKGROUND"):
                label = class_map["BACKGROUND"]
            elif name.startswith("HELICOPTER"):
                label = class_map["HELICOPTER"]
            elif name.startswith("DRONE"):
                label = class_map["DRONE"]
            else:
                logger.warning(f"Skipping unknown file: {file.name}")
                continue

        # Compute features
        logmel = compute_logmel(file, n_mels=n_mels)
        data_by_class[label].append(logmel)

        logger.info(f"Loaded: {file.name} → class={label}, shape={logmel.shape}")

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

        logger.info(f"Class {label}: {len(train_split)} train / {len(val_split)} val")

    return (
        np.array(X_train, dtype=object),
        np.array(y_train),
        np.array(X_val, dtype=object),
        np.array(y_val),
    )

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
# Plotting
# -------------------------------------------------------------------------
def plot_training(history, save_path):
    """
    Plot training & validation curves for all metrics found in history.history.
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
    plt.savefig(save_path)
    logger.info(f"Training plot saved to {save_path}")