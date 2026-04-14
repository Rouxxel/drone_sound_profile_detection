# Drone Sound Profile Detection

A comprehensive machine learning project for detecting and classifying audio signatures of drones, helicopters, and background noise using both Deep Learning (CNN) and Traditional ML models.

The models should be able to detect (can be extended with more datasets):
1. FPV quadcopters
2. Fixed‑wing loitering munitions (Lancet, KUB, V2U, etc.)

---

## Target Application: Embedded Drone Detection

The purpose of this project is to **select and optimize the most efficient model** for deployment on an **embedded system**. The target use case is an **add-on for soldier vests** that detects drones by sound: **four microphones** provide **directional information** (where the sound is coming from), and a **lightweight ML model** runs on the device to classify the audio and determine **whether it is a drone or not**. Alerts (detection + direction) are sent to helmet embedded comms.

### Proposed Hardware & Software (Vest Add-On)

| Component | Details |
|-----------|---------|
| **Hardware** | |
| MCU | ESP32-S3 Dev Module (~10 g, €6–15) |
| Audio capture | 4× I²S microphones (front/back) at 16 kHz |
| Microphones | 4× MEMS I²S Microphones (~16 g, €6–15) |
| Power | 2,000 mAh Li-Po Battery (~50 g, €6–15) |
| Other | Enclosure, wiring & mounts (~15–20 g, €5–12), Wind-noise reduction foam (<1 g) |
| **Software / processing** | |
| Buffering | Rolling buffer: 2–4 s of audio for pre-trigger context |
| Calibration | Auto-calibration: per-mic gain normalization at startup |
| DSP | Digital notch filters + wind-noise suppression |
| Inference | Duty-cycled ML: energy detector triggers model only when needed |
| Model | Lightweight audio classifier for drone detection |
| **Output** | Alerts to helmet comms with detection & direction |
| **Total weight** | ~90 g |
| **Total cost** | €25–60 |
| **Runtime** | ~8 hours continuous (extendable) |

### Benefits

- **Low cost** (€25–60 per unit)
- **Lightweight system** (~90–120 g total)
- **Long battery life** (~8 hours continuous, can be extended with )
- **360° acoustic** coverage
- **Reliable drone identification**
- **Low false-alarm** operation
- **Automatic self-calibration**
- **Environmental noise resistance**
- **Efficient** duty-cycled processing
- **Notification** through embedded comms
- **Rugged and modular** hardware
- **Detection range** ~100–900 m

---

## Project Overview

This project implements multiple machine learning approaches for audio classification with two modes:

### Multiclass Classification (3 classes, could be extended)
- **Drone** sounds
- **Helicopter** sounds (use a similar fundamental propulsion system)
- **Background** noise

### Binary Classification (2 classes)
- **DRONE**: Drone sounds
- **NO_DRONE**: Everything else

The project includes:
- **Tiny CNN**: Lightweight model for low-resource environments
- **Robust CNN**: Advanced deep learning model with data augmentation and callbacks, best suited for bigger systems
- **Traditional ML Models**: Random Forest, SVM, XGBoost and Gradient Boosting for classification

Traditional ML models utilize MFCC (Mel-Frequency Cepstral Coefficients) features extracted from audio files for classification, spectral descriptors (centroid, bandwidth, rolloff, flatness), zero-crossing rate, RMS energy and optional HNR (harmonic-to-noise ratio).
CNN models ingest the raw audio files with a Log-Mel Spectrogram (Mel bands vary according to robustness of model) and normalization.

## Project Structure

```
drone_sound_profile_detection/
├── main.py                          # Full pipeline: download → convert → EDA → train → test
├── configuration/
│   ├── config_loader.py      # script to load config.json and import it to any script that needs it
│   └── config.json           # json that contains the configuration of almost all scripts (mainly models)
├── datasets/
│   ├── data_kaggle_download.py      # Download dataset from Kaggle, copy .wav and creates audios/
│   ├── audio_to_csv_converter_simple_mfcc.py # DEPRECATED CONVERTER VERSION
│   ├── aud_csv_converter_trad_ml.py # Convert WAV in audios/ to ingestable csv for trad ML models
│   └── tests/
│      └── trad_ml_converter_test.py # Unit test for audio converter
├── modeling/
│   ├── EDA/                         # Exploratory data analysis
│   │   ├── plot_converted_csv.py    # MFCC heatmap + RMS plot per CSV
│   │   └── eda_summary_report.py    # Dataset summary, class/duration/PCA plots, report
│   ├── models/
│   │   ├── multiclass/              # 3-class: Drone, Helicopter, Background
│   │   │   ├── tiny_cnn_model.py
│   │   │   ├── robust_cnn_model.py
│   │   │   ├── ml_models.py
│   │   │   └── testing/
│   │   │       ├── test_tiny_model.py
│   │   │       ├── test_robust_model.py
│   │   │       └── test_ml_models.py
│   │   ├── binary/                  # 2-class: Drone vs No-Drone
│   │   │   ├── tiny_cnn_binary.py
│   │   │   ├── ml_models_binary.py
│   │   │   └── testing/
│   │   │       ├── test_tiny_cnn_binary.py
│   │   │       └── test_ml_models_binary.py
│   │   └── utils/
│   │       ├── custom_logger.py
│   │       ├── ml_models_binary.py
│   │       ├── cnn_train_methods.py
│   │       ├── ml_train_methods.py
│   │       └── pycache_n_logs_deleter.py
│   ├── documentation/               # USAGE_GUIDE.md and other docs
├── .gitignore
├── exprmntl_insights.md
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

The dataset consists of 90 audio samples (30 per class):
- **DRONE_001.wav** to **DRONE_030.wav**
- **HELICOPTER_001.wav** to **HELICOPTER_030.wav**
- **BACKGROUND_001.wav** to **BACKGROUND_030.wav**

Each CSV file contains frame-level audio features extracted using Librosa and a feature composition (per frame, 21 total features):

- [0–13]&#58; MFCCs (14 coefficients)
- [14]&#58;   Spectral Centroid
- [15]&#58;   Spectral Bandwidth
- [16]&#58;   Spectral Rolloff
- [17]&#58;   Spectral Flatness
- [18]&#58;   Zero Crossing Rate (ZCR)
- [19]&#58;   RMS Energy
- [20]&#58;   HNR (Harmonic-to-Noise Ratio)

Additional details:
- Features are computed per time frame using a shared STFT representation for efficiency.
- MFCCs are derived from the log-power spectrogram.
- HNR is estimated via Harmonic-Percussive Source Separation (HPSS).
- Any NaN or infinite values are replaced with 0 during post-processing.
- Each row in the CSV corresponds to a single time frame, and each column represents a specific feature.

**Data Split:**
- Training: 80% (24 samples per class = 72 total)
- Validation: 20% (6 samples per class = 18 total)

## Model Architectures

### 1. Tiny CNN (Lightweight)
- Input: Log-Mel Spectrograms (configurable Mel bands via `n_mels`)
- 2 Convolutional layers (configurable filters & kernel sizes)
- Batch Normalization after each convolution
- Max Pooling after first convolutional block
- Global Average Pooling layer
- Dense layer with Dropout (configurable rate)
- Final Dense output layer
- **Optimizer**: Adam (configurable learning rate)
- **Loss**: Configurable (typically categorical crossentropy)
- **Epochs**: Configurable (`epoch` in config)
- **Batch Size**: Configurable (`batch_size` in config)
- **Parameters**: Depends on config (typically lightweight, ~tens of thousands)
- **Best for**: Low-resource environments, edge devices, real-time audio classification

### 2. Robust CNN (Advanced)
- Input: Log-Mel Spectrograms (higher resolution via configurable `n_mels`)
- Deep CNN with 3 convolutional blocks (each with 2 Conv2D layers), might be increased
- Batch Normalization after each convolution
- Max Pooling in early blocks for spatial reduction
- Dropout applied throughout (conv blocks + dense layers)
- Global Average Pooling layer
- Multiple Dense layers with Dropout
- Final Dense output layer (softmax for multiclass classification)
- **Classes**: 3 (Drone, Helicopter, Background)
- **Data Augmentation**: Enabled (configurable factor via `aug_factor`)
- **Callbacks**:
  - Early stopping
  - Learning rate scheduling
  - Model checkpointing
- **Optimizer**: Adam (configurable learning rate)
- **Loss**: Configurable (typically categorical crossentropy)
- **Epochs**: Configurable (`epoch` in config, with early stopping)
- **Batch Size**: Configurable (`batch_size` in config)
- **Parameters**: Config-dependent (typically significantly larger than Tiny CNN)
- **Best for**: Higher accuracy, robustness to noise/variability, sufficient computational resources

### 3. Traditional ML Models
ll models operate on handcrafted statistical features derived from MFCC-based representations (and related spectral descriptors).
- **Random Forest**: Ensemble of decision trees (configurable, typically ~200 estimators, depth-limited)
- **SVM**: RBF kernel with regularization (C parameter tuning)
- **XGBoost**: Gradient-boosted decision trees (configurable estimators and depth)
- **Gradient Boosting**: Sequential boosting of weak learners (configurable depth and learning rate)
Common preprocessing:
- Feature extraction from MFCC-based audio representations
- Spectral/statistical feature aggregation per audio sample
- Feature scaling using StandardScaler
- **Best for**: Fast training, strong baseline performance, interpretability, and small-to-medium datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<owner>/drone_sound_profile_detection.git
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

This runs: dataset download, CSV conversion, both EDA scripts (concurrent), all 5 training scripts (concurrent), then all 5 test scripts (concurrent, after training).

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

Test scripts read from `trained_models/`:

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

### CNN models (multiclass)
- **tiny_cnn/**: `tiny_cnn_audio_model.keras`, `tiny_cnn_audio_model.pkl`, `tiny_cnn_audio_model.h5`, `training_history_tiny_cnn.png`
- **robust_cnn/**: `robust_cnn_audio_model.keras`, `robust_cnn_audio_model.pkl`, `robust_cnn_audio_model.h5`, `best_model.keras`, `training_history_robust_cnn.png`

### CNN models (binary)
- **binary/tiny_cnn/**: `robust_cnn_audio_model.keras`,`tiny_cnn_binary_model.pkl`, `tiny_cnn_binary_model.h5`, `training_history_tiny_cnn_binary.png`

### ML models (multiclass)
- **ml_models/**: `ml_model_randomforest.pkl`, `ml_model_svm.pkl`, `ml_model_xgboost.pkl`, `ml_model_gradientboosting.pkl`, `best_ml_model_*.pkl`, `feature_scaler.pkl` (required for predictions)

### ML models (binary)
- **binary/ml_models/**: `ml_model_binary_*.pkl`, `best_ml_model_binary_*.pkl`, `feature_scaler_binary.pkl` (required for predictions)

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Audio dataset sources and preprocessing techniques
- TensorFlow/Keras for the deep learning framework
- Librosa for audio feature extraction
