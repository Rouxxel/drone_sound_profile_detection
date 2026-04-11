#!/usr/bin/env python3
"""
Binary Tiny CNN Audio Classifier Training Script
------------------------------------------------

Binary classification: DRONE vs NO_DRONE
- DRONE: Drone sounds (class 1)
- NO_DRONE: Background + Helicopter sounds (class 0)
"""

import os
import sys
import pickle
import logging
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
# main Paths
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent #binary/
REPO_ROOT = SCRIPT_DIR.parent.parent.parent #root/
AUDIO_FOLDER = REPO_ROOT / "datasets" / "audios"

#custom logger, config and utils
sys.path.append(str(REPO_ROOT))
from modeling.utils.custom_logger import get_logger
from configuration.config_loader import config
from modeling.utils.cnn_train_methods import load_dataset, preprocess_data, plot_training
logger = get_logger("cnn_logger", "cnn_models_training.log")

# General Configuration
TRAINED_MODEL_ROOT = REPO_ROOT / config["cnn_models"]["output_folder_str"]
SPECIFIC_FOLDER = TRAINED_MODEL_ROOT / config["cnn_models"]["binary"]["general"]["folder"]

#binary tiny cnn config
PLOT_NAME = config["cnn_models"]["binary"]["tiny_cnn"]["plot_name.png"]
EPOCH = config["cnn_models"]["binary"]["tiny_cnn"]["epoch"]
BATCH_SIZE = config["cnn_models"]["binary"]["tiny_cnn"]["batch_size"]
N_MELS = config["cnn_models"]["binary"]["tiny_cnn"]["n_mels"]
LAYERS = config["cnn_models"]["binary"]["tiny_cnn"]["layers"]
LEARNING_RATE= config["cnn_models"]["binary"]["tiny_cnn"]["learning_rate"]
ACTIVATION= config["cnn_models"]["binary"]["tiny_cnn"]["activation"]
LAST_ACTIVATION= config["cnn_models"]["binary"]["tiny_cnn"]["last_activation"]
PADDING= config["cnn_models"]["binary"]["tiny_cnn"]["padding"]
LOSS= config["cnn_models"]["binary"]["tiny_cnn"]["loss"]
METRICS= config["cnn_models"]["binary"]["tiny_cnn"]["metrics"]
#N_CLASSES = config["cnn_models"]["binary"]["general"]["n_classes"]
CLASS_MAP = config["cnn_models"]["binary"]["general"]["class_map"]

# -------------------------------------------------------------------------
# Model Definition
# -------------------------------------------------------------------------
def build_tiny_cnn_binary(input_shape: Tuple[int, int, int]) -> models.Sequential:
    logger.info("Building Binary Tiny CNN model...")
    logger.info(f"Layer 1 Conv2D layer with {LAYERS['conv1']['filters']} filters and kernel size {LAYERS['conv1']['kernel']}, padding '{PADDING}', activation '{ACTIVATION}' and MaxPooling {LAYERS['conv1']['max_pool']}")
    logger.info(f"Layer 2 Conv2D layer with {LAYERS['conv2']['filters']} filters and kernel size {LAYERS['conv2']['kernel']}, padding '{PADDING}' and activation '{ACTIVATION}'")
    logger.info(f"Dense layer with {LAYERS['dense']['units']} units, dropout {LAYERS['dense']['dropout']}, activation '{ACTIVATION}' and last activation '{LAST_ACTIVATION}'")
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(
            LAYERS["conv1"]["filters"], 
            tuple(LAYERS["conv1"]["kernel"]), 
            activation=ACTIVATION, 
            padding=PADDING),
        layers.BatchNormalization(),
        layers.MaxPooling2D(tuple(LAYERS["conv1"]["max_pool"])),
        
        # Second conv block
        layers.Conv2D(
            LAYERS["conv2"]["filters"], 
            tuple(LAYERS["conv2"]["kernel"]), 
            activation=ACTIVATION),
        layers.BatchNormalization(),
        
        # Global pooling + dense
        layers.GlobalAveragePooling2D(),
        layers.Dense(LAYERS["dense"]["units"], activation=ACTIVATION),
        layers.Dropout(LAYERS["dense"]["dropout"]),
        layers.Dense(1, activation=LAST_ACTIVATION)  # Binary output
    ])
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss=LOSS,  # Binary loss
        metrics=['accuracy']
        #metrics=METRICS TODO: solve other metrics capabilities
        )
    logger.info("Model built successfully")
    return model

# -------------------------------------------------------------------------
# Main Training Pipeline
# -------------------------------------------------------------------------
def main():
    logger.info("Starting Tiny CNN Binary Training...")
    aud_dir = str(AUDIO_FOLDER)
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(audio_dir=aud_dir,class_map=CLASS_MAP,n_mels=N_MELS,is_binary=True)
    X_train = preprocess_data(X_train_raw)
    X_val = preprocess_data(X_val_raw)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Input shape to model: {X_train.shape[1:]}")
    logger.info(f"Class distribution - Train: NO_DRONE={np.sum(y_train==0)}, DRONE={np.sum(y_train==1)}")
    logger.info(f"Class distribution - Val: NO_DRONE={np.sum(y_val==0)}, DRONE={np.sum(y_val==1)}")

    model = build_tiny_cnn_binary(input_shape=X_train.shape[1:])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=8,
                        verbose=1).history

    # Create trained_models/binary/tiny_cnn at repo root (only if not exists)
    model_dir = TRAINED_MODEL_ROOT / SPECIFIC_FOLDER
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model as pickle
    model_path = model_dir / "tiny_cnn_binary_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to: {model_path}")
    
    #Also save as native keras extension for better compatibility
    keras_path = model_dir / "tiny_cnn_binary_model.keras"
    model.save(keras_path)
    logger.info(f"Model also saved as Keras to: {keras_path}")
    
    # Also save as h5 for compatibility
    h5_path = model_dir / "tiny_cnn_binary_model.h5"
    model.save(h5_path)
    logger.info(f"Model also saved as H5 to: {h5_path}")

    PLOT_PATH = model_dir / PLOT_NAME
    plot_training(history=history,save_path=PLOT_PATH)
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
