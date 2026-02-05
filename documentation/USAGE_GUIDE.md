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

This runs all dataset, EDA, training, and testing scripts in the correct order. Test results are saved in `logs/`.

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

All training scripts use paths relative to the repo root. Run from the **repo root** (e.g. `python modeling/models/multiclass/tiny_cnn_model.py`). Outputs go to `trained_models/` and `logs/` at the repo root; directories are created only if they do not exist.

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
- `trained_models/tiny_cnn/tiny_cnn_audio_model.pkl`
- `trained_models/tiny_cnn/tiny_cnn_audio_model.h5`
- `trained_models/tiny_cnn/training_history_tiny_cnn.png`
- `logs/tiny_cnn_training.log`

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
- `trained_models/robust_cnn/robust_cnn_audio_model.pkl`
- `trained_models/robust_cnn/robust_cnn_audio_model.h5`
- `trained_models/robust_cnn/best_model.keras` (best checkpoint)
- `trained_models/robust_cnn/training_history_robust_cnn.png`
- `logs/robust_cnn_training.log`

### Option 3: Traditional ML Models – Multiclass

```bash
python modeling/models/multiclass/ml_models.py
```

**Training Details:**
- **Time:** ~2-5 minutes
- **Models Trained:** Random Forest, SVM, XGBoost, Gradient Boosting
- **Best for:** Fast training, interpretability

**Output:**
- `trained_models/ml_models/ml_model_randomforest.pkl`
- `trained_models/ml_models/ml_model_svm.pkl`
- `trained_models/ml_models/ml_model_xgboost.pkl`
- `trained_models/ml_models/ml_model_gradientboosting.pkl`
- `trained_models/ml_models/best_ml_model_*.pkl`
- `trained_models/ml_models/feature_scaler.pkl` (required for predictions)
- `logs/ml_models_training.log`

### Option 4: Binary classification (Drone vs No-Drone)

**Tiny CNN Binary:**
```bash
python modeling/models/binary/tiny_cnn_binary.py
```
Output: `trained_models/binary/tiny_cnn/` (e.g. `tiny_cnn_binary_model.pkl`, `.h5`), `logs/tiny_cnn_binary_training.log`

**ML Models Binary:**
```bash
python modeling/models/binary/ml_models_binary.py
```
Output: `trained_models/binary/ml_models/` (e.g. `ml_model_binary_*.pkl`, `best_ml_model_binary_*.pkl`, `feature_scaler_binary.pkl`), `logs/ml_models_binary_training.log`

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
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('trained_models/tiny_cnn/tiny_cnn_audio_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Or for Robust CNN:
# with open('trained_models/robust_cnn/robust_cnn_audio_model.pkl', 'rb') as f:
#     model = pickle.load(f)
```

#### Step 2: Preprocess Your Audio Data

```python
def preprocess_audio_for_cnn(mfcc_csv_path):
    """
    Preprocess a single MFCC CSV file for CNN prediction
    
    Args:
        mfcc_csv_path: Path to CSV file containing MFCC features
        
    Returns:
        Preprocessed numpy array ready for model input
    """
    # Load MFCC from CSV
    mfcc = pd.read_csv(mfcc_csv_path).values.astype(np.float32)
    
    # Get dimensions
    num_frames = mfcc.shape[0]  # e.g., 130 time frames
    num_coeffs = mfcc.shape[1]  # Should be 14 MFCC coefficients
    
    # Pad to match training data (use max_frames from your training)
    # For this example, let's say max_frames = 862 (from training)
    max_frames = 862
    
    # Create padded array
    padded_mfcc = np.zeros((max_frames, num_coeffs), dtype=np.float32)
    padded_mfcc[:num_frames, :] = mfcc
    
    # Add channel dimension and batch dimension
    # Shape: (1, max_frames, 14, 1)
    preprocessed = np.expand_dims(padded_mfcc, axis=-1)  # Add channel
    preprocessed = np.expand_dims(preprocessed, axis=0)   # Add batch
    
    return preprocessed

# Example usage
audio_data = preprocess_audio_for_cnn('path/to/your/audio.csv')
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

#### Complete Example for CNN

```python
import pickle
import numpy as np
import pandas as pd

def predict_audio_cnn(model_path, audio_csv_path, max_frames=862):
    """Complete prediction pipeline for CNN models"""
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load and preprocess audio
    mfcc = pd.read_csv(audio_csv_path).values.astype(np.float32)
    
    # Pad to max_frames
    num_frames = mfcc.shape[0]
    num_coeffs = mfcc.shape[1]
    padded_mfcc = np.zeros((max_frames, num_coeffs), dtype=np.float32)
    padded_mfcc[:num_frames, :] = mfcc
    
    # Add dimensions
    preprocessed = np.expand_dims(padded_mfcc, axis=-1)
    preprocessed = np.expand_dims(preprocessed, axis=0)
    
    # Predict
    predictions = model.predict(preprocessed, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    
    # Map to class name
    class_names = {0: "BACKGROUND", 1: "HELICOPTER", 2: "DRONE"}
    
    return {
        'class': class_names[predicted_class],
        'class_id': int(predicted_class),
        'confidence': float(confidence),
        'probabilities': {
            'BACKGROUND': float(predictions[0][0]),
            'HELICOPTER': float(predictions[0][1]),
            'DRONE': float(predictions[0][2])
        }
    }

# Usage
result = predict_audio_cnn(
    'trained_models/tiny_cnn/tiny_cnn_audio_model.pkl',
    'path/to/your/audio.csv'
)

print(result)
# Output: {'class': 'DRONE', 'class_id': 2, 'confidence': 0.8, 
#          'probabilities': {'BACKGROUND': 0.05, 'HELICOPTER': 0.15, 'DRONE': 0.8}}
```

---

### ML Models (Random Forest, SVM, XGBoost)

#### Step 1: Load Model and Scaler

```python
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('trained_models/ml_models/best_ml_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the feature scaler (REQUIRED!)
with open('trained_models/ml_models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

#### Step 2: Extract Statistical Features

```python
def extract_statistical_features(mfcc):
    """
    Extract statistical features from MFCC for ML models
    
    Args:
        mfcc: numpy array of shape (num_frames, 14)
        
    Returns:
        Feature vector of shape (126,) - 9 statistics × 14 coefficients
    """
    mfcc = np.asarray(mfcc, dtype=np.float32)
    features = []
    
    # Mean, std, min, max for each coefficient (4 × 14 = 56 features)
    features.extend(np.mean(mfcc, axis=0))
    features.extend(np.std(mfcc, axis=0))
    features.extend(np.min(mfcc, axis=0))
    features.extend(np.max(mfcc, axis=0))
    
    # Median and percentiles (3 × 14 = 42 features)
    features.extend(np.median(mfcc, axis=0))
    features.extend(np.percentile(mfcc, 25, axis=0))
    features.extend(np.percentile(mfcc, 75, axis=0))
    
    # Delta features - first derivative (2 × 14 = 28 features)
    delta = np.diff(mfcc, axis=0)
    if len(delta) > 0:
        features.extend(np.mean(delta, axis=0))
        features.extend(np.std(delta, axis=0))
    else:
        features.extend(np.zeros(mfcc.shape[1]))
        features.extend(np.zeros(mfcc.shape[1]))
    
    return np.array(features, dtype=np.float32)

# Example usage
mfcc = pd.read_csv('path/to/your/audio.csv').values.astype(np.float32)
features = extract_statistical_features(mfcc)
# features shape: (126,)
```

#### Step 3: Scale and Predict

```python
# Scale features (IMPORTANT!)
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

#### Complete Example for ML Models

```python
import pickle
import numpy as np
import pandas as pd

def predict_audio_ml(model_path, scaler_path, audio_csv_path):
    """Complete prediction pipeline for ML models"""
    
    # Load model and scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load MFCC
    mfcc = pd.read_csv(audio_csv_path).values.astype(np.float32)
    
    # Extract statistical features
    features = []
    features.extend(np.mean(mfcc, axis=0))
    features.extend(np.std(mfcc, axis=0))
    features.extend(np.min(mfcc, axis=0))
    features.extend(np.max(mfcc, axis=0))
    features.extend(np.median(mfcc, axis=0))
    features.extend(np.percentile(mfcc, 25, axis=0))
    features.extend(np.percentile(mfcc, 75, axis=0))
    
    delta = np.diff(mfcc, axis=0)
    if len(delta) > 0:
        features.extend(np.mean(delta, axis=0))
        features.extend(np.std(delta, axis=0))
    else:
        features.extend(np.zeros(mfcc.shape[1]))
        features.extend(np.zeros(mfcc.shape[1]))
    
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    predicted_class = model.predict(features_scaled)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
    else:
        probabilities = None
    
    # Map to class name
    class_names = {0: "BACKGROUND", 1: "HELICOPTER", 2: "DRONE"}
    
    result = {
        'class': class_names[predicted_class],
        'class_id': int(predicted_class),
    }
    
    if probabilities is not None:
        result['confidence'] = float(probabilities[predicted_class])
        result['probabilities'] = {
            'BACKGROUND': float(probabilities[0]),
            'HELICOPTER': float(probabilities[1]),
            'DRONE': float(probabilities[2])
        }
    
    return result

# Usage
result = predict_audio_ml(
    'trained_models/ml_models/best_ml_model_xgboost.pkl',
    'trained_models/ml_models/feature_scaler.pkl',
    'path/to/your/audio.csv'
)

print(result)
```

---

## Input/Output Specifications

### CNN Models Input

**Format:** Numpy array
**Shape:** `(batch_size, max_frames, 14, 1)`
- `batch_size`: Number of samples (usually 1 for single prediction)
- `max_frames`: Maximum time frames (862 for this project)
- `14`: Number of MFCC coefficients
- `1`: Channel dimension

**Data Type:** `float32`
**Value Range:** Normalized MFCC values (typically -3 to +3)

### CNN Models Output

**Format:** Numpy array
**Shape:** `(batch_size, 3)`
**Example:** `[[0.05, 0.15, 0.80]]`

**Interpretation:**
- Index 0: Probability of BACKGROUND class
- Index 1: Probability of HELICOPTER class
- Index 2: Probability of DRONE class
- Sum of probabilities = 1.0

### ML Models Input

**Format:** Numpy array
**Shape:** `(1, 126)`
- 126 statistical features extracted from MFCC
- Must be scaled using the saved `feature_scaler.pkl`

**Data Type:** `float32`

### ML Models Output

**Predicted Class:** Integer (0, 1, or 2)
**Probabilities:** Array of shape `(3,)` (if model supports `predict_proba`)

---

## Troubleshooting

### Model file not found
- Ensure you've trained the model first
- Check the correct path: `trained_models/[model_type]/[model_name].pkl`

### Shape mismatch errors
- **CNN:** Ensure your MFCC has 14 coefficients
- **CNN:** Pad to max_frames (862) before prediction
- **ML:** Ensure you extract exactly 126 features

### Import errors
```bash
pip install -r requirements.txt
```

### Low accuracy
- Ensure you have at least 20-30 samples per class
- Check that CSV files are properly named (DRONE_*, HELICOPTER_*, BACKGROUND_*)
- Verify MFCC extraction was successful

### Scaler not found (ML models)
- Always load `feature_scaler.pkl` before making predictions with ML models
- The scaler is created during training and is required for consistent feature scaling

---

## Quick Reference

### Class Mapping
```python
0 = "BACKGROUND"
1 = "HELICOPTER"  
2 = "DRONE"
```

### Model Paths
```
trained_models/
├── tiny_cnn/          # Multiclass Tiny CNN
├── robust_cnn/        # Multiclass Robust CNN
├── ml_models/         # Multiclass ML + feature_scaler.pkl (required!)
└── binary/
    ├── tiny_cnn/      # Binary Tiny CNN
    └── ml_models/     # Binary ML + feature_scaler_binary.pkl (required!)
```

### Minimum Requirements
- **Dataset:** 20-30 samples per class (60-90 total)
- **Audio Format:** WAV files
- **MFCC:** 14 coefficients
- **Python:** 3.7+
