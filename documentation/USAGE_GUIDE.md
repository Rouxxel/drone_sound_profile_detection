# Quick Usage Guide

## Training the Model

1. Navigate to the model directory:
```bash
cd model
```

2. Run the training script:
```bash
python tiny_cnn_model.py
```

**What happens:**
- Loads 90 CSV files (30 per class) from `../datasets/converted_csv/`
- Splits data: 80% training (72 samples), 20% validation (18 samples)
- Trains for 50 epochs with batch size 8
- Saves model to `trained_model/` as both `.pkl` and `.h5`
- Generates training plot: `training_history_tiny_cnn.png`
- Logs everything to `logs/tiny_cnn_training.log`

## Testing the Model

1. After training, run the test script:
```bash
python test_model.py
```

**What happens:**
- Loads the trained model from `trained_model/tiny_cnn_audio_model.pkl`
- Evaluates on the validation set (18 samples)
- Displays:
  - Overall accuracy
  - Confusion matrix
  - Classification report (precision, recall, F1-score per class)
  - Per-class accuracy breakdown
- Logs everything to `logs/tiny_cnn_testing.log`

## Expected Output Structure

After running both scripts:
```
drone_sound_profile_detection/
├── model/
│   ├── tiny_cnn_model.py
│   ├── test_model.py
│   └── training_history_tiny_cnn.png  ← Generated plot
├── trained_model/                      ← Created automatically
│   ├── tiny_cnn_audio_model.pkl       ← Pickle format
│   └── tiny_cnn_audio_model.h5        ← Keras H5 format
└── logs/                               ← Created automatically
    ├── tiny_cnn_training.log          ← Training logs
    └── tiny_cnn_testing.log           ← Testing logs
```

## Loading the Model in Your Own Code

### Using Pickle (Recommended)
```python
import pickle

with open('trained_model/tiny_cnn_audio_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Use the model
predictions = model.predict(your_data)
```

### Using Keras H5
```python
from keras.models import load_model

model = load_model('trained_model/tiny_cnn_audio_model.h5')

# Use the model
predictions = model.predict(your_data)
```

## Data Format

Input data must be preprocessed MFCC features:
- Shape: `(batch_size, max_frames, 14, 1)`
- Where `max_frames` is the maximum number of time frames in your dataset
- 14 MFCC coefficients per frame
- Values should be normalized

## Class Labels

- 0: BACKGROUND
- 1: HELICOPTER
- 2: DRONE

## Troubleshooting

**Model file not found:**
- Make sure you've run `tiny_cnn_model.py` first to train and save the model

**Import errors:**
- Install all dependencies: `pip install -r requirements.txt`

**Path errors:**
- Run scripts from the `model/` directory
- Dataset should be in `../datasets/converted_csv/`
