# Drone Sound Profile Detection

A machine learning project for detecting and classifying audio signatures of drones, helicopters, and background noise using Convolutional Neural Networks (CNN).

## Project Overview

This project implements a lightweight CNN model optimized for low-resource environments to classify audio into three categories:
- **Drone** sounds
- **Helicopter** sounds
- **Background** noise

The model uses MFCC (Mel-Frequency Cepstral Coefficients) features extracted from audio files for classification.

## Project Structure

```
drone_sound_profile_detection/
├── datasets/
│   ├── audios/                    # Raw audio files (.wav)
│   ├── converted_csv/             # MFCC features in CSV format
│   │   ├── DRONE_*.csv           # 30 drone audio samples
│   │   ├── HELICOPTER_*.csv      # 30 helicopter audio samples
│   │   └── BACKGROUND_*.csv      # 30 background noise samples
│   └── wav_to_csv_converter.py   # Script to convert WAV to MFCC CSV
├── model/
│   ├── tiny_cnn_model.py         # Training script
│   └── test_model.py             # Testing/evaluation script
├── trained_model/                 # Saved trained models (created after training)
│   ├── tiny_cnn_audio_model.pkl  # Model saved as pickle
│   └── tiny_cnn_audio_model.h5   # Model saved as H5 (Keras format)
├── logs/                          # Training and testing logs
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Dataset

The dataset consists of 90 audio samples (30 per class):
- **DRONE_001.csv** to **DRONE_030.csv**
- **HELICOPTER_001.csv** to **HELICOPTER_030.csv**
- **BACKGROUND_001.csv** to **BACKGROUND_030.csv**

Each CSV file contains MFCC features (14 coefficients) extracted from audio files.

**Data Split:**
- Training: 80% (24 samples per class = 72 total)
- Validation: 20% (6 samples per class = 18 total)

## Model Architecture

The Tiny CNN model consists of:
- 2 Convolutional layers (16 and 32 filters)
- Batch Normalization layers
- Max Pooling layer
- Global Average Pooling layer
- Dense layers with Dropout (0.2)
- Softmax output layer (3 classes)

**Model Parameters:**
- Optimizer: Adam (learning rate: 0.001)
- Loss: Sparse Categorical Crossentropy
- Epochs: 50
- Batch Size: 8

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/drone_sound_profile_detection.git
cd drone_sound_profile_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Convert Audio to MFCC Features (if needed)

If you have raw audio files, convert them to MFCC CSV format:

```bash
cd datasets
python wav_to_csv_converter.py
```

This will process all `.wav` files in the `audios/` folder and save MFCC features to `converted_csv/`.

### 2. Train the Model

Navigate to the model directory and run the training script:

```bash
cd model
python tiny_cnn_model.py
```

This will:
- Load the dataset from `../datasets/converted_csv/`
- Split data into 80/20 train/validation sets
- Train the CNN model for 50 epochs
- Save the trained model to `trained_model/` directory as both `.pkl` and `.h5` formats
- Generate a training history plot (`training_history_tiny_cnn.png`)
- Log all training details to `logs/tiny_cnn_training.log`

### 3. Test the Model

After training, evaluate the model's performance:

```bash
cd model
python test_model.py
```

This will:
- Load the trained model from `trained_model/tiny_cnn_audio_model.pkl`
- Evaluate on the validation set
- Display accuracy, confusion matrix, and classification report
- Log all results to `logs/tiny_cnn_testing.log`

## Model Output

The trained model is saved in two formats:
- **PKL format** (`tiny_cnn_audio_model.pkl`): Python pickle format for easy loading
- **H5 format** (`tiny_cnn_audio_model.h5`): Keras native format for compatibility

## Results

After training, you'll find:
- **Training logs**: `logs/tiny_cnn_training.log`
- **Testing logs**: `logs/tiny_cnn_testing.log`
- **Training plot**: `training_history_tiny_cnn.png` (shows accuracy and loss curves)
- **Model files**: `trained_model/` directory

The test script provides:
- Overall validation accuracy
- Confusion matrix
- Per-class precision, recall, and F1-score
- Per-class accuracy breakdown

## Requirements

- Python 3.7+
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Librosa (for audio processing)
- Matplotlib (for plotting)

See `requirements.txt` for specific versions.

## Features

- ✅ Lightweight CNN architecture optimized for low-resource environments
- ✅ MFCC feature extraction for audio classification
- ✅ 80/20 train-validation split with fixed random seed for reproducibility
- ✅ Model saved in both PKL and H5 formats
- ✅ Comprehensive logging system
- ✅ Automated testing and evaluation script
- ✅ Confusion matrix and classification metrics
- ✅ Training history visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Audio dataset sources and preprocessing techniques
- TensorFlow/Keras for the deep learning framework
- Librosa for audio feature extraction
