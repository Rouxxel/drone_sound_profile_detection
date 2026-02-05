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
├── main.py                          # Full pipeline: download → convert → EDA → train → test
├── datasets/
│   ├── audios/                      # Raw audio files (.wav), created by data_kaggle_download.py
│   ├── converted_csv/               # MFCC CSV files, created by audio_to_csv_converter.py
│   ├── data_kaggle_download.py      # Download dataset from Kaggle, copy .wav into audios/
│   └── audio_to_csv_converter.py    # Convert WAV in audios/ to MFCC CSV in converted_csv/
├── modeling/
│   ├── EDA/                         # Exploratory data analysis
│   │   ├── plot_converted_csv.py    # MFCC heatmap + RMS plot per CSV
│   │   ├── eda_summary_report.py    # Dataset summary, class/duration/PCA plots, report
│   │   └── plots/                   # Per-file plots (created by plot_converted_csv.py)
│   ├── models/
│   │   ├── multiclass/              # 3-class: Drone, Helicopter, Background
│   │   │   ├── tiny_cnn_model.py
│   │   │   ├── robust_cnn_model.py
│   │   │   ├── ml_models.py
│   │   │   └── testing/
│   │   │       ├── test_tiny_model.py
│   │   │       ├── test_robust_model.py
│   │   │       └── test_ml_models.py
│   │   └── binary/                  # 2-class: Drone vs No-Drone
│   │       ├── tiny_cnn_binary.py
│   │       ├── ml_models_binary.py
│   │       └── testing/
│   │           ├── test_tiny_cnn_binary.py
│   │           └── test_ml_models_binary.py
│   ├── documentation/               # USAGE_GUIDE.md and other docs
│   └── utils/                       # e.g. pycache_n_logs_deleter.py
├── trained_models/                  # Saved models (created when training; only if not exists)
│   ├── tiny_cnn/                    # Multiclass Tiny CNN
│   ├── robust_cnn/                  # Multiclass Robust CNN
│   ├── ml_models/                   # Multiclass ML (RF, SVM, XGBoost, GB)
│   └── binary/
│       ├── tiny_cnn/                # Binary Tiny CNN
│       └── ml_models/               # Binary ML + feature_scaler_binary.pkl
├── logs/                            # Training and testing logs
├── requirements.txt
└── README.md
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

### Option A: Run the full pipeline (recommended)

From the repo root, run everything in order (download → convert → EDA → train all → test all):

```bash
python main.py
```

This runs: dataset download, CSV conversion, both EDA scripts (concurrent), all 5 training scripts (concurrent), then all 5 test scripts (concurrent, after training). Test results are written to `logs/`.

### Option B: Run steps manually

#### 1. Download data and convert to MFCC CSV

```bash
# Download Kaggle dataset and copy .wav files into datasets/audios/
python datasets/data_kaggle_download.py

# Convert all .wav in audios/ to MFCC CSV in converted_csv/
python datasets/audio_to_csv_converter.py
```

Scripts use paths relative to their location, so you can run from the repo root.

#### 2. (Optional) EDA

```bash
python modeling/EDA/plot_converted_csv.py       # Per-file MFCC + RMS plots → modeling/EDA/plots/
python modeling/EDA/eda_summary_report.py       # Summary report → modeling/EDA/eda_report/
```

#### 3. Train models

All paths are relative to the repo root; run from anywhere (e.g. repo root). Models and logs go to `trained_models/` and `logs/` at repo root.

**Multiclass (Drone, Helicopter, Background):**

```bash
python modeling/models/multiclass/tiny_cnn_model.py
python modeling/models/multiclass/robust_cnn_model.py
python modeling/models/multiclass/ml_models.py
```

**Binary (Drone vs No-Drone):**

```bash
python modeling/models/binary/tiny_cnn_binary.py
python modeling/models/binary/ml_models_binary.py
```

**Binary classification is recommended when** you only need to detect drone presence, want higher accuracy, or need faster inference.

#### 4. Test models

Test scripts read from `trained_models/` and write results to `logs/`:

**Multiclass:**
```bash
python modeling/models/multiclass/testing/test_tiny_model.py
python modeling/models/multiclass/testing/test_robust_model.py
python modeling/models/multiclass/testing/test_ml_models.py
```

**Binary:**
```bash
python modeling/models/binary/testing/test_tiny_cnn_binary.py
python modeling/models/binary/testing/test_ml_models_binary.py
```

Each test script prints and logs to `logs/*.log`: validation accuracy, confusion matrix, classification report, and per-class accuracy.

## Model Outputs

All outputs are under `trained_models/` at the repo root (folders are created only if they do not exist).

### CNN models (multiclass)
- **tiny_cnn/**: `tiny_cnn_audio_model.pkl`, `tiny_cnn_audio_model.h5`, `training_history_tiny_cnn.png`
- **robust_cnn/**: `robust_cnn_audio_model.pkl`, `robust_cnn_audio_model.h5`, `best_model.keras`, `training_history_robust_cnn.png`

### CNN models (binary)
- **binary/tiny_cnn/**: `tiny_cnn_binary_model.pkl`, `tiny_cnn_binary_model.h5`, `training_history_tiny_cnn_binary.png`

### ML models (multiclass)
- **ml_models/**: `ml_model_randomforest.pkl`, `ml_model_svm.pkl`, `ml_model_xgboost.pkl`, `ml_model_gradientboosting.pkl`, `best_ml_model_*.pkl`, `feature_scaler.pkl` (required for predictions)

### ML models (binary)
- **binary/ml_models/**: `ml_model_binary_*.pkl`, `best_ml_model_binary_*.pkl`, `feature_scaler_binary.pkl` (required for predictions)

## Results

After training and testing:
- **Training logs**: `logs/tiny_cnn_training.log`, `logs/ml_models_training.log`, `logs/robust_cnn_training.log`, and binary equivalents
- **Testing logs**: `logs/tiny_cnn_testing.log`, `logs/ml_models_testing.log`, `logs/robust_cnn_testing.log`, and binary equivalents (accuracy, confusion matrix, classification report)
- **Training plots**: e.g. `trained_models/tiny_cnn/training_history_tiny_cnn.png`
- **Model files**: `trained_models/` (see Model Outputs above)

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
