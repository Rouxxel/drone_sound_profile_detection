#!/usr/bin/env python3
"""
Robust CNN Audio Classifier Training Script
-------------------------------------------


It computes log-mel spectrograms (64–128 Mel bands), normalizes them,
and trains a robust CNN model.

This script implements a more robust CNN architecture with:
- Deeper network with more convolutional layers
- Data augmentation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Cross-validation support

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
from modeling.utils.cnn_train_methods import load_dataset, preprocess_data, plot_training, augment_data, get_callbacks
logger = get_logger("cnn_logger", "cnn_models_training.log")

# General Configuration
TRAINED_MODEL_ROOT = REPO_ROOT / config["cnn_models"]["output_folder_str"]
SPECIFIC_FOLDER = TRAINED_MODEL_ROOT / config["cnn_models"]["multiclass"]["general"]["folder"]

#Multiclass robust cnn config
PLOT_NAME = config["cnn_models"]["multiclass"]["robust_cnn"]["plot_name.png"]
EPOCH = config["cnn_models"]["multiclass"]["robust_cnn"]["epoch"]
BATCH_SIZE = config["cnn_models"]["multiclass"]["robust_cnn"]["batch_size"]
N_MELS = config["cnn_models"]["multiclass"]["robust_cnn"]["n_mels"]
LAYERS = config["cnn_models"]["multiclass"]["robust_cnn"]["layers"]
AUG_FACTOR = config["cnn_models"]["multiclass"]["robust_cnn"]["aug_factor"]
LEARNING_RATE= config["cnn_models"]["multiclass"]["robust_cnn"]["learning_rate"]
ACTIVATION= config["cnn_models"]["multiclass"]["robust_cnn"]["activation"]
LAST_ACTIVATION= config["cnn_models"]["multiclass"]["robust_cnn"]["last_activation"]
PADDING= config["cnn_models"]["multiclass"]["robust_cnn"]["padding"]
LOSS= config["cnn_models"]["multiclass"]["robust_cnn"]["loss"]
METRICS= config["cnn_models"]["multiclass"]["robust_cnn"]["metrics"]
N_CLASSES = config["cnn_models"]["multiclass"]["general"]["n_classes"]
CLASS_MAP = config["cnn_models"]["multiclass"]["general"]["class_map"]

# -------------------------------------------------------------------------
# Model Definition
# -------------------------------------------------------------------------

def build_robust_cnn(input_shape: Tuple[int, int, int], num_classes: int = 3) -> models.Sequential:
    logger.info("Building Robust CNN model...")
    logger.info(f"Layer 1 Conv2D layer with {LAYERS['conv1']['filters']} filters and kernel size {LAYERS['conv1']['kernel']}, padding '{PADDING}', activation '{ACTIVATION}' and MaxPooling {LAYERS['conv1']['max_pool']}")
    logger.info(f"Layer 2 Conv2D layer with {LAYERS['conv2']['filters']} filters and kernel size {LAYERS['conv2']['kernel']}, padding '{PADDING}', activation '{ACTIVATION}' and MaxPooling {LAYERS['conv2']['max_pool']}")
    logger.info(f"Layer 3 Conv2D layer with {LAYERS['conv3']['filters']} filters and kernel size {LAYERS['conv3']['kernel']}, padding '{PADDING}', activation '{ACTIVATION}' and GlobalAveragePooling2D")
    logger.info(f"Dense layer with {LAYERS['dense']['units_1']} units, dropout {LAYERS['dense']['dropout_1']}, activation '{ACTIVATION}' and last activation '{LAST_ACTIVATION}'")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Conv Block
        layers.Conv2D(
            LAYERS["conv1"]["filters"], 
            tuple(LAYERS["conv1"]["kernel"]), 
            activation=ACTIVATION, 
            padding=PADDING
            ),
        layers.BatchNormalization(),
        layers.Conv2D(
            LAYERS["conv1"]["filters"], 
            tuple(LAYERS["conv1"]["kernel"]), 
            activation=ACTIVATION, 
            padding=PADDING
            ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(tuple(LAYERS["conv1"]["max_pool"])),
        layers.Dropout(LAYERS["conv1"]["dropout"]),
        
        # Second Conv Block
        layers.Conv2D(
            LAYERS["conv2"]["filters"], 
            tuple(LAYERS["conv2"]["kernel"]), 
            activation=ACTIVATION, 
            padding=PADDING
            ),
        layers.BatchNormalization(),
        layers.Conv2D(
            LAYERS["conv2"]["filters"], 
            tuple(LAYERS["conv2"]["kernel"]), 
            activation=ACTIVATION, 
            padding=PADDING
            ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(tuple(LAYERS["conv2"]["max_pool"])),
        layers.Dropout(LAYERS["conv2"]["dropout"]),
        
        # Third Conv Block
        layers.Conv2D(
            LAYERS["conv3"]["filters"], 
            tuple(LAYERS["conv3"]["kernel"]), 
            activation=ACTIVATION, 
            padding=PADDING
            ),
        layers.BatchNormalization(),
        layers.Conv2D(
            LAYERS["conv3"]["filters"], 
            tuple(LAYERS["conv3"]["kernel"]), 
            activation=ACTIVATION, 
            padding=PADDING
            ),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(LAYERS["conv3"]["dropout"]),
        
        # Dense Layers
        layers.Dense(LAYERS["dense"]["units_1"], activation=ACTIVATION),
        layers.BatchNormalization(),
        layers.Dropout(LAYERS["dense"]["dropout_1"]),
        layers.Dense(LAYERS["dense"]["units_2"], activation=ACTIVATION),
        layers.Dropout(LAYERS["dense"]["dropout_2"]),
        layers.Dense(num_classes, activation=LAST_ACTIVATION)
    ])
    
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss=LOSS,
        metrics=['accuracy']
    )
    
    logger.info("Model built successfully")
    logger.info(f"Total parameters: {model.count_params():,}")
    return model

# -------------------------------------------------------------------------
# Main Training Pipeline
# -------------------------------------------------------------------------

def main():
    """Main training pipeline for the robust CNN model."""
    logger.info("Starting Robust CNN Multiclass Training...")
    
    # Load dataset
    aud_dir = str(AUDIO_FOLDER)
    X_train_raw, y_train, X_val_raw, y_val = load_dataset(audio_dir=aud_dir, class_map=CLASS_MAP, n_mels=N_MELS, is_binary=False)
    X_train = preprocess_data(X_train_raw)
    X_val = preprocess_data(X_val_raw)
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Input shape to model: {X_train.shape[1:]}")
    
    # Augment training data
    X_train, y_train = augment_data(X_train, y_train, augmentation_factor=AUG_FACTOR)
    logger.info(f"Training samples (after augmentation): {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Input shape to model: {X_train.shape[1:]}")
    
    # Build model
    model = build_robust_cnn(input_shape=X_train.shape[1:], num_classes=N_CLASSES)
    
    # Setup callbacks
    callback_list = get_callbacks(model_dir=SPECIFIC_FOLDER)
    
    # Train model
    logger.info("Starting training with callbacks...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        callbacks=callback_list,
        verbose=1
    ).history
    
    # Create trained_models/robust_cnn at repo root (only if not exists)
    model_dir = TRAINED_MODEL_ROOT / SPECIFIC_FOLDER
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final model
    model_path = model_dir / "robust_cnn_audio_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to: {model_path}")
    
    #Also save as native keras extension for better compatibility
    keras_path = model_dir / "tiny_cnn_binary_model.keras"
    model.save(keras_path)
    logger.info(f"Model also saved as Keras to: {keras_path}")
    
    # Also save as h5
    h5_path = model_dir / "robust_cnn_audio_model.h5"
    model.save(h5_path)
    logger.info(f"Model also saved as H5 to: {h5_path}")
    
    # Plot training history
    PLOT_PATH = model_dir / PLOT_NAME
    plot_training(history=history,save_path=PLOT_PATH)
    
    # Final evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Final Validation Loss: {val_loss:.4f}")
    logger.info(f"Final Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
