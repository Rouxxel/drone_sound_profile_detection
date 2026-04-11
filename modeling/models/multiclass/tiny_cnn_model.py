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
import pickle
from pathlib import Path
from typing import Tuple
from keras import layers, models, optimizers

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
from modeling.utils.cnn_train_methods import load_dataset, preprocess_data, plot_training
#from modeling.utils. import 
logger = get_logger("cnn_logger", "cnn_models_training.log")

# General Configuration
TRAINED_MODEL_ROOT = REPO_ROOT / config["cnn_models"]["output_folder_str"]
SPECIFIC_FOLDER = TRAINED_MODEL_ROOT / config["cnn_models"]["multiclass"]["general"]["folder"]

#Multiclass tiny cnn config
PLOT_NAME = config["cnn_models"]["multiclass"]["tiny_cnn"]["plot_name.png"]
EPOCH = config["cnn_models"]["multiclass"]["tiny_cnn"]["epoch"]
BATCH_SIZE = config["cnn_models"]["multiclass"]["tiny_cnn"]["batch_size"]
N_MELS = config["cnn_models"]["multiclass"]["tiny_cnn"]["n_mels"]
LAYERS = config["cnn_models"]["multiclass"]["tiny_cnn"]["layers"]
LEARNING_RATE= config["cnn_models"]["multiclass"]["tiny_cnn"]["learning_rate"]
ACTIVATION= config["cnn_models"]["multiclass"]["tiny_cnn"]["activation"]
LAST_ACTIVATION= config["cnn_models"]["multiclass"]["tiny_cnn"]["last_activation"]
PADDING= config["cnn_models"]["multiclass"]["tiny_cnn"]["padding"]
LOSS= config["cnn_models"]["multiclass"]["tiny_cnn"]["loss"]
METRICS= config["cnn_models"]["multiclass"]["tiny_cnn"]["metrics"]
N_CLASSES = config["cnn_models"]["multiclass"]["general"]["n_classes"]
CLASS_MAP = config["cnn_models"]["multiclass"]["general"]["class_map"]

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
    
    logger.info(f"Layer 1 Conv2D layer with {LAYERS['conv1']['filters']} filters and kernel size {LAYERS['conv1']['kernel']}, padding '{PADDING}', activation '{ACTIVATION}' and MaxPooling {LAYERS['conv1']['max_pool']}")
    logger.info(f"Layer 2 Conv2D layer with {LAYERS['conv2']['filters']} filters and kernel size {LAYERS['conv2']['kernel']}, padding '{PADDING}', activation '{ACTIVATION}'")
    logger.info(f"Dense layer with {LAYERS['dense']['units']} units, dropout {LAYERS['dense']['dropout']}, activation '{ACTIVATION}', last activation '{LAST_ACTIVATION}' and GlobalAveragePooling2D")
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
            activation=ACTIVATION, 
            padding=PADDING),
        layers.BatchNormalization(),

        # Global pooling + dense
        layers.GlobalAveragePooling2D(),
        layers.Dense(LAYERS["dense"]["units"], activation=ACTIVATION),
        layers.Dropout(LAYERS["dense"]["dropout"]),
        layers.Dense(num_classes, activation=LAST_ACTIVATION)
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
# Main Training Pipeline
# -------------------------------------------------------------------------

def main():
    logger.info("Starting Tiny CNN Multiclass Training...")
    aud_dir = str(AUDIO_FOLDER)
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(audio_dir=aud_dir,class_map=CLASS_MAP,n_mels=N_MELS,is_binary=False)
    X_train = preprocess_data(X_train_raw)
    X_val = preprocess_data(X_val_raw)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Input shape to model: {X_train.shape[1:]}")

    model = build_tiny_cnn(input_shape=X_train.shape[1:], num_classes=N_CLASSES)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCH,
                        batch_size=BATCH_SIZE,
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
    
    #Also save as native keras extension for better compatibility
    keras_path = model_dir / "tiny_cnn_binary_model.keras"
    model.save(keras_path)
    logger.info(f"Model also saved as Keras to: {keras_path}")
    
    # Also save as h5 for compatibility
    h5_path = model_dir / "tiny_cnn_audio_model.h5"
    model.save(h5_path)
    logger.info(f"Model also saved as H5 to: {h5_path}")

    PLOT_PATH = model_dir / PLOT_NAME
    plot_training(history=history,save_path=PLOT_PATH)
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
