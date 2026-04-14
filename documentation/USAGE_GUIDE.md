# Comprehensive Usage Guide

## Table of Contents
1. [Quick start (full pipeline)](#quick-start-full-pipeline)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Models](#training-models)
4. [Testing Models](#testing-models)
5. [Using Trained Models](#using-trained-models)
6. [Troubleshooting](#troubleshooting)

---

## Quick start (full pipeline)

From the **repo root**, run the entire workflow (download → convert → EDA → train all → test all):

```bash
python main.py
```

---

## Dataset Preparation

### Step 1: Obtain Audio Files

**Option A – Download from Kaggle (recommended):**

```bash
python datasets/data_kaggle_download.py
```

This downloads the Kaggle dataset, copies all `.wav` files into `datasets/audios/`, and removes the temporary download folder. Requires Kaggle credentials (`kaggle.json`).

**Option B – Manual:** Place your `.wav` files in `datasets/audios/` (create the folder if needed).

### Step 2: Convert Audio to MFCC Features

From the repo root:

```bash
python datasets/audio_to_csv_converter.py
```

**What this does:**
- Reads all `.wav` (and optionally .mp3, .flac, etc.) from `datasets/audios/`
- Extracts 14 MFCC coefficients from each file
- Normalizes the MFCC features
- Saves CSV files in `datasets/converted_csv/` (creates the folder if it does not exist)

**Output format:**
- Each CSV file contains MFCC features
- Rows = time frames
- Columns = 14 MFCC coefficients
- Example: A 3-second audio at 22050 Hz might produce ~130 time frames

### Step 3: Organize CSV Files

Rename your CSV files to follow the naming convention:
```
datasets/converted_csv/
├── DRONE_001.csv
├── DRONE_002.csv
├── ...
├── HELICOPTER_001.csv
├── HELICOPTER_002.csv
├── ...
├── BACKGROUND_001.csv
├── BACKGROUND_002.csv
└── ...
```

**Recommended:** 30 files per class (90 total) for best results.

---

## Training Models
### Option 1: Tiny CNN – Multiclass (Fast & Lightweight)

```bash
python modeling/models/multiclass/tiny_cnn_model.py
```

**Training Details:**
- **Time:** ~5-10 minutes
- **Epochs:** 50
- **Batch Size:** 8
- **Model Size:** ~200KB
- **Best for:** Edge devices, real-time applications

**Output:**
- trained model with .h5, .pkl and .keras extensions
- png of training history plot
- logs of training

### Option 2: Robust CNN – Multiclass (Maximum Accuracy)

```bash
python modeling/models/multiclass/robust_cnn_model.py
```

**Training Details:**
- **Time:** ~15-30 minutes
- **Epochs:** 100 (with early stopping)
- **Batch Size:** 16
- **Data Augmentation:** 3x (time shifting + noise)
- **Model Size:** ~2MB
- **Best for:** Maximum performance

**Output:**
- trained model with .h5, .pkl and .keras extensions
- png of training history plot
- logs of training

### Option 3: Traditional ML Models – Multiclass

```bash
python modeling/models/multiclass/ml_models.py
```

**Training Details:**
- **Time:** ~2-5 minutes
- **Models Trained:** Random Forest, SVM, XGBoost, Gradient Boosting
- **Best for:** Fast training, interpretability

**Output:**
- .pkl file of all models
- .pkl file for feature scaler (required for predictions)
- logs of training

### Option 4: Binary classification (Drone vs No-Drone)

**Tiny CNN Binary:**
```bash
python modeling/models/binary/tiny_cnn_binary.py
```

**ML Models Binary:**
```bash
python modeling/models/binary/ml_models_binary.py
```

---

## Testing Models

After training, run test scripts from the **repo root**. Results are written to `logs/` (e.g. `tiny_cnn_testing.log`, `ml_models_testing.log`).

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

**Test output (in logs and console):**
- Validation accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Per-class accuracy breakdown

---

## Using Trained Models

### CNN Models (Tiny & Robust)
#### Step 1: Load the Model

```python
# Load the trained model
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load pickle model (Tiny CNN / some saved versions)
with open('path/to/model/tiny_cnn_audio_model.pkl', 'rb') as f:
    model = pickle.load(f)

# OR load Keras models (.h5 / .keras)
# model = load_model('path/to/model/robust_cnn_audio_model.keras')
# model = load_model('path/to/model/robust_cnn_audio_model.h5')
```

#### Step 2: Preprocess Audio Data

```python
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
```

#### Step 3: Make Predictions

```python
# Get prediction probabilities
predictions = model.predict(audio_data, verbose=0)

# predictions shape: (1, 3) - probabilities for each class
# Example: [[0.05, 0.15, 0.80]]

# Get predicted class
predicted_class = np.argmax(predictions, axis=1)[0]

# Get confidence score
confidence = predictions[0][predicted_class]

# Class mapping
class_names = {0: "BACKGROUND", 1: "HELICOPTER", 2: "DRONE"}

print(f"Predicted: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
print(f"\nAll probabilities:")
print(f"  BACKGROUND:  {predictions[0][0]:.2%}")
print(f"  HELICOPTER:  {predictions[0][1]:.2%}")
print(f"  DRONE:       {predictions[0][2]:.2%}")
```

**Expected Output:**
```
Predicted: DRONE
Confidence: 80.00%

All probabilities:
  BACKGROUND:  5.00%
  HELICOPTER:  15.00%
  DRONE:       80.00%
```

### ML Models (Random Forest, SVM, XGBoost)
#### Step 1: Load Model and Scaler

```python
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('path/to/model/selected_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the feature scaler (REQUIRED!)
with open('path/to/model/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

#### Step 2: Extract Statistical Features
```python
def extract_statistical_features(mfcc: np.ndarray) -> np.ndarray:
    """
    Extract a fixed-length statistical feature vector from an MFCC matrix. Take 
    MFCC time-series matrix (shape: frames × coefficients) and computes 
    summary statistics across time for each MFCC coefficient.

    Returns
    -------
    np.ndarray
        1D feature vector (float32) containing the following statistics
        for each MFCC coefficient:

        - Mean
        - Standard deviation
        - Minimum value
        - Maximum value
        - Median
        - 25th percentile
        - 75th percentile
        - Mean of first-order delta (temporal derivative)
        - Std of first-order delta

        If the MFCC contains fewer than 2 frames, delta features are
        replaced with zeros.
    """
    # Ensure mfcc is a proper numpy array with float dtype
    mfcc = np.asarray(mfcc, dtype=np.float32)  
    features = []
    
    # Mean, std, min, max for each coefficient
    features.extend(np.mean(mfcc, axis=0))
    features.extend(np.std(mfcc, axis=0))
    features.extend(np.min(mfcc, axis=0))
    features.extend(np.max(mfcc, axis=0))
    
    # Median and percentiles
    features.extend(np.median(mfcc, axis=0))
    features.extend(np.percentile(mfcc, 25, axis=0))
    features.extend(np.percentile(mfcc, 75, axis=0))
    
    # Delta features (first derivative)
    delta = np.diff(mfcc, axis=0)
    if len(delta) > 0:
        features.extend(np.mean(delta, axis=0))
        features.extend(np.std(delta, axis=0))
    else:
        features.extend(np.zeros(mfcc.shape[1]))
        features.extend(np.zeros(mfcc.shape[1]))

    #Convert list to np array for vector operations
    vec = np.array(features, dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0) #nan_to_num before return
    
    return vec
```

#### Step 3: Scale and Predict
```python
# Scale features
features_scaled = scaler.transform(features.reshape(1, -1))

# Make prediction
predicted_class = model.predict(features_scaled)[0]

# Get probability (if model supports it)
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(features_scaled)[0]
else:
    probabilities = None

# Class mapping
class_names = {0: "BACKGROUND", 1: "HELICOPTER", 2: "DRONE"}

print(f"Predicted: {class_names[predicted_class]}")
if probabilities is not None:
    print(f"Confidence: {probabilities[predicted_class]:.2%}")
    print(f"\nAll probabilities:")
    print(f"  BACKGROUND:  {probabilities[0]:.2%}")
    print(f"  HELICOPTER:  {probabilities[1]:.2%}")
    print(f"  DRONE:       {probabilities[2]:.2%}")
```
---

## Input/Output Specifications
### CNN Models Input
**Format:** Numpy array  
**Shape:** `(batch_size, time_frames, n_mels, 1)`  

- `batch_size`: Number of samples (usually 1 for single prediction)  
- `time_frames`: Number of time steps (depends on audio length)  
- `n_mels`: Number of Mel bands (configurable)  
- `1`: Channel dimension  

**Data Type:** `float32`  
**Value Range:** Normalized log-Mel spectrogram values  

### CNN Models Output
**Format:** Numpy array  
**Shape:** `(batch_size, 3)`  
**Example:** `[[0.05, 0.15, 0.80]]`  

**Interpretation:**
- Index 0: BACKGROUND  
- Index 1: HELICOPTER  
- Index 2: DRONE  
- Probabilities sum to 1.0  

---

### ML Models Input
**Format:** Numpy array  
**Shape:** `(n_samples, n_features)`  

- Feature vector derived from MFCC + spectral statistics  
- Must be scaled using the saved `feature_scaler.pkl`  

**Data Type:** `float32`  

### ML Models Output
**Predicted Class:** Integer (e.g., 0, 1, 2)  
**Probabilities:** Array of shape `(n_classes,)` (if `predict_proba` is supported)  
---

## Troubleshooting

### Model file not found
- Ensure you've trained the model first
- Check the correct path: `trained_models/[model_type]/[model_name].pkl`

### Import errors
```bash
pip install -r requirements.txt
```

### Low accuracy
- Even with all optimizations, the dataset is still rather small, if possible to increase, quality should increase too
- Check that CSV files are properly named (DRONE_*, HELICOPTER_*, BACKGROUND_*)
- Verify MFCC extraction was successful

### Scaler not found (ML models)
- Always load `feature_scaler.pkl` before making predictions with ML models
- The scaler is created during training and is required for consistent feature scaling

---

## Quick Reference

### Class Mapping 
#### Multiclass
```python
0 = "BACKGROUND"
1 = "HELICOPTER"  
2 = "DRONE"
```

#### Binary
```python
0 = "NO_DRONE" 
2 = "DRONE"
```

### Minimum Requirements
- **Dataset:** 20-30 samples per class (60-90 total)
- **Audio Format:** WAV files
- **MFCC:** 14 coefficients
- **Python:** 3.7+
