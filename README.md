# Drone Sound Profile Detection

A comprehensive machine learning project for detecting and classifying audio signatures of drones, helicopters, and background noise using both Deep Learning (CNN) and Traditional ML models.

The models should be able to detect:
1. FPV quadcopters
2. Fixed‑wing loitering munitions (Lancet, KUB, V2U, etc.)

## Project Overview

This project implements multiple machine learning approaches for audio classification with two modes:

### Multiclass Classification (3 classes)
- **Drone** sounds
- **Helicopter** sounds
- **Background** noise

### Binary Classification (2 classes)
- **DRONE**: Drone sounds
- **NO_DRONE**: Background + Helicopter sounds combined

The project includes:
- **Tiny CNN**: Lightweight model for low-resource environments
- **Robust CNN**: Advanced deep learning model with data augmentation and callbacks
- **Traditional ML Models**: Random Forest, SVM, XGBoost, and Gradient Boosting

All models use MFCC (Mel-Frequency Cepstral Coefficients) features extracted from audio files for classification.

## Project Structure

```
drone_sound_profile_detection/
├── datasets/
│   ├── audios/                      # Raw audio files (.wav)
│   ├── converted_csv/               # MFCC features in CSV format
│   │   ├── DRONE_*.csv             # 30 drone audio samples
│   │   ├── HELICOPTER_*.csv        # 30 helicopter audio samples
│   │   └── BACKGROUND_*.csv        # 30 background noise samples
│   └── wav_to_csv_converter.py     # Script to convert WAV to MFCC CSV
├── model/
│   ├── multiclass/                 # 3-class classification (Drone, Helicopter, Background)
│   │   ├── tiny_cnn_model.py
│   │   ├── robust_cnn_model.py
│   │   ├── ml_models.py
│   │   └── testing/
│   ├── binary/                     # 2-class classification (Drone vs No-Drone)
│   │   ├── tiny_cnn_binary.py
│   │   ├── ml_models_binary.py
│   │   └── testing/
│   │       ├── test_tiny_cnn_binary.py
│   │       └── test_ml_models_binary.py
│   └── predict_single_audio.py     # Single audio prediction script
├── trained_model/                   # Saved trained models (created after training)
│   ├── multiclass/                 # 3-class models
│   │   ├── tiny_cnn/
│   │   ├── robust_cnn/
│   │   └── ml_models/
│   └── binary/                     # 2-class models (Drone vs No-Drone)
│       ├── tiny_cnn/
│       │   ├── tiny_cnn_binary_model.pkl
│       │   └── tiny_cnn_binary_model.h5
│       └── ml_models/
│           ├── ml_model_binary_*.pkl
│           ├── best_ml_model_binary_*.pkl
│           └── feature_scaler_binary.pkl
├── logs/                            # Training and testing logs
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
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

## Model Architectures

### 1. Tiny CNN (Lightweight)
- 2 Convolutional layers (16 and 32 filters)
- Batch Normalization layers
- Max Pooling layer
- Global Average Pooling layer
- Dense layers with Dropout (0.2)
- **Parameters**: ~50K
- **Epochs**: 50, **Batch Size**: 8
- **Best for**: Low-resource environments, edge devices

### 2. Robust CNN (Advanced)
- 6 Convolutional layers (32, 32, 64, 64, 128, 128 filters)
- Multiple Batch Normalization and Dropout layers
- Data augmentation (time shifting, noise addition)
- Early stopping and learning rate scheduling
- Model checkpointing
- **Parameters**: ~500K
- **Epochs**: 100 (with early stopping), **Batch Size**: 16
- **Best for**: Maximum accuracy, sufficient computational resources

### 3. Traditional ML Models
All models use statistical features extracted from MFCC:
- **Random Forest**: 200 trees, max depth 20
- **SVM**: RBF kernel, C=10
- **XGBoost**: 200 estimators, max depth 10
- **Gradient Boosting**: 200 estimators, learning rate 0.1
- **Best for**: Fast training, interpretability, smaller datasets

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

### 2. Train Models

Choose between **Multiclass** (3 classes) or **Binary** (Drone vs No-Drone) classification:

#### Multiclass Classification (Drone, Helicopter, Background)

```bash
cd model/multiclass

# Option A: Tiny CNN
python tiny_cnn_model.py

# Option B: Robust CNN
python robust_cnn_model.py

# Option C: ML Models
python ml_models.py
```

#### Binary Classification (Drone vs No-Drone)

```bash
cd model/binary

# Option A: Tiny CNN Binary
python tiny_cnn_binary.py

# Option B: ML Models Binary
python ml_models_binary.py
```

**Binary classification is recommended when:**
- You only need to detect drone presence
- You want higher accuracy (simpler problem)
- You need faster inference

### 3. Test Models

After training, evaluate model performance:

#### Multiclass Models
```bash
cd model/multiclass/testing
python test_tiny_model.py      # Test tiny CNN
python test_robust_model.py    # Test robust CNN
python test_ml_models.py       # Test all ML models
```

#### Binary Models
```bash
cd model/binary/testing
python test_tiny_cnn_binary.py      # Test binary CNN
python test_ml_models_binary.py     # Test binary ML models
```

Each test script provides:
- Validation accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Per-class accuracy breakdown

## Model Outputs

### CNN Models
Saved in multiple formats:
- **PKL format**: Python pickle format for easy loading
- **H5 format**: Keras format for compatibility
- **KERAS format**: Best model checkpoint (Robust CNN only)

### ML Models
- Individual model files: `ml_model_randomforest.pkl`, `ml_model_svm.pkl`, etc.
- Best model: `best_ml_model_*.pkl`
- Feature scaler: `feature_scaler.pkl` (required for predictions)

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

Actual Tested models
# Drone Sound Profile Detection: Model Comparison

| Model               | Validation Accuracy | Background Accuracy | Helicopter Accuracy | Drone Accuracy | Notes / Observations |
|--------------------|------------------|------------------|------------------|---------------|--------------------|
| **SVM**             | 94.44%           | 100%             | 100%             | 83.33%        | Best overall model. Strong per-class performance even with small dataset. |
| **RandomForest**    | 88.89%           | 83.33%           | 100%             | 83.33%        | Performs well, slightly worse than SVM. Consistent per-class accuracy. |
| **XGBoost**         | 88.89%           | 83.33%           | 100%             | 83.33%        | Similar to RandomForest; good for small datasets, but slightly behind SVM. |
| **GradientBoosting**| 83.33%           | 83.33%           | 83.33%           | 83.33%        | Lower overall accuracy. Per-class accuracy is balanced but less precise. |
| **Robust CNN v1**   | 38.89%           | 100%             | 16.67%           | 0%            | Severely biased toward Background class. Too few samples to learn features effectively. |
| **Robust CNN v2**   | 55.56%           | 16.67%           | 66.67%           | 83.33%        | Improvement over v1, but still poor for Background class. Small dataset limits CNN performance. |

---

### **Explanation Based on Dataset**

- **Dataset Size**: 30 files per class, 10 seconds each → 90 audio samples total (15 minutes).  
- **Traditional ML Models**:
  - Work very well on small datasets.
  - SVM is the top performer due to its robustness with limited data and high-dimensional features (e.g., MFCCs or spectral features).
  - RandomForest and XGBoost are also solid; GradientBoosting slightly less accurate.
- **Robust CNN Models**:
  - Struggle significantly with small data.
  - CNN v1 overfits to the Background class; cannot generalize to Drone or Helicopter sounds.
  - CNN v2 improves slightly but still cannot match traditional ML models.
- **Key Insight**:
  - For your current dataset, **traditional ML models are far more reliable than CNNs**.  
  - CNNs require either more data, extensive augmentation, or transfer learning to perform better.

---

**Recommendation:** Stick with SVM or other traditional ML models for now. If you want to use CNNs, consider augmenting your dataset or using pretrained audio CNNs (transfer learning) to compensate for the small dataset.

## Requirements

- Python 3.7+
- TensorFlow/Keras (for CNN models)
- NumPy
- Pandas
- Scikit-learn (for ML models and metrics)
- XGBoost (for XGBoost classifier)
- Librosa (for audio processing)
- Matplotlib (for plotting)

See `requirements.txt` for specific versions.

## Model Comparison

| Model | Accuracy* | Training Time | Model Size | Inference Speed | Best Use Case |
|-------|-----------|---------------|------------|-----------------|---------------|
| Tiny CNN | ~85-90% | 5-10 min | ~200KB | Fast | Edge devices, real-time |
| Robust CNN | ~90-95% | 15-30 min | ~2MB | Medium | Maximum accuracy |
| Random Forest | ~80-85% | 2-3 min | ~1MB | Very Fast | Interpretability |
| SVM | ~75-80% | 3-5 min | ~500KB | Fast | Small datasets |
| XGBoost | ~85-90% | 2-3 min | ~1MB | Very Fast | Best ML model |
| Gradient Boosting | ~80-85% | 3-5 min | ~1MB | Fast | Alternative to XGBoost |

*Accuracy may vary based on dataset and hyperparameters

## Features

### Deep Learning (CNN)
- ✅ Tiny CNN for edge devices and real-time processing
- ✅ Robust CNN with advanced techniques (augmentation, callbacks)
- ✅ Data augmentation (time shifting, noise addition)
- ✅ Early stopping and learning rate scheduling
- ✅ Model checkpointing for best weights

### Traditional ML
- ✅ Multiple algorithms: Random Forest, SVM, XGBoost, Gradient Boosting
- ✅ Automatic model comparison and selection
- ✅ Statistical feature extraction from MFCC
- ✅ Feature standardization with saved scaler

### General
- ✅ MFCC feature extraction for audio classification
- ✅ 80/20 train-validation split with reproducibility
- ✅ Models saved in multiple formats (PKL, H5)
- ✅ Comprehensive logging system
- ✅ Automated testing and evaluation scripts
- ✅ Confusion matrix and detailed classification metrics
- ✅ Training history visualization

## Which Model Should I Use?

**Choose Tiny CNN if:**
- You need real-time inference
- Running on edge devices (Raspberry Pi, mobile)
- Limited computational resources
- Fast training is priority

**Choose Robust CNN if:**
- Maximum accuracy is critical
- You have sufficient computational resources
- Training time is not a constraint
- You have more data or can use augmentation

**Choose Traditional ML if:**
- You need fast training and inference
- Model interpretability is important
- You prefer simpler models
- XGBoost typically gives best results among ML models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Audio dataset sources and preprocessing techniques
- TensorFlow/Keras for the deep learning framework
- Librosa for audio feature extraction
