#!/usr/bin/env python3
"""
CNN Models util methods for Audio Classification
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import librosa
import numpy as np
from keras import callbacks
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

#------------------------------------------------------
# Path
#------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent #utils/
REPO_ROOT = SCRIPT_DIR.parent.parent.parent #root/

#------------------------------------------------------
# Custom logger
#------------------------------------------------------
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

def augment_data(X: np.ndarray, y: np.ndarray, augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment data with time shifting and noise addition
    """
    logger.info(f"Augmenting data with factor {augmentation_factor}...")
    X_aug, y_aug = [], []
    
    for i, x_sample in enumerate(X):
        #original sample
        X_aug.append(x_sample)
        y_aug.append(y[i]) 
        
        #Generate augmented samples
        for _ in range(augmentation_factor - 1):
            sample = x_sample.copy()
            
            # --- Augmentation Logic ---
            shift = np.random.randint(-10, 10)
            if shift != 0:
                sample = np.roll(sample, shift, axis=0)
            
            noise = np.random.normal(0, 0.005, sample.shape)
            sample = sample + noise
            
            #Append
            X_aug.append(sample)
            y_aug.append(y[i])
    
    logger.info(f"Data augmented: {len(X)} -> {len(X_aug)} samples")
    return np.array(X_aug), np.array(y_aug)

# -------------------------------------------------------------------------
# Plotting and callbacks
# -------------------------------------------------------------------------
def plot_training(history, save_path):
    """
    Handles any metrics by adding an overfitting monitor
    and saves to disk.
    """
    # If passing the Keras history object, extract the dictionary
    hist_dict = history.history if hasattr(history, 'history') else history
    
    metrics = [k for k in hist_dict.keys() if not k.startswith("val_")]
    n_metrics = len(metrics) + 1  # +1 for the Overfitting Monitor
    
    # Calculate grid size (e.g., 2 columns, rows as needed)
    cols = 2
    rows = (n_metrics + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        axes[i].plot(hist_dict[metric], label=f"Train", linewidth=2)
        if f"val_{metric}" in hist_dict:
            axes[i].plot(hist_dict[f"val_{metric}"], label=f"Val", linewidth=2)
        
        axes[i].set_title(metric.replace("_", " ").capitalize(), fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Add the Overfitting Monitor (specifically for Accuracy)
    if "accuracy" in hist_dict and "val_accuracy" in hist_dict:
        diff = np.array(hist_dict["accuracy"]) - np.array(hist_dict["val_accuracy"])
        axes[len(metrics)].plot(diff, color='red', linewidth=2)
        axes[len(metrics)].axhline(y=0, color='black', linestyle='--')
        axes[len(metrics)].set_title("Overfitting Monitor (Acc Diff)", fontsize=12, fontweight='bold')
        axes[len(metrics)].grid(True, alpha=0.3)

    # Clean up empty subplots
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close() # Close to free up memory
    logger.info(f"Training plot saved to {save_path}")

def get_callbacks(model_dir: Path):
    """Setup training callbacks"""
    
    #Early stopping
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
