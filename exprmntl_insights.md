# Insights based on results and testing
## Unified Model Ranking (All Models — Best → Worst)

1. **RandomForest (Binary)** — 88.89%  
   - Most stable across tiny datasets  
   - Excellent balance between DRONE and NO_DRONE  
   - Low overfitting risk, robust, lightweight

2. **XGBoost (Binary)** — 88.89%  
   - Matches RandomForest in accuracy  
   - Slightly more sensitive to parameter tuning  
   - Strong choice for embedded inference

3. **SVM (Binary)** — 83.33%  
   - Very balanced per-class performance  
   - Strong with tiny datasets  
   - Fast inference, small model size

4. **Tiny CNN (Binary)** — 83.33%  
   - Best CNN overall  
   - Performs surprisingly well given low amountsdata  
   - Suitable for embedded deployment, but still trails top ML models in stability

5. **GradientBoosting (Binary)** — 77.78%  
   - Noticeably weaker than top ML models  
   - Struggles with DRONE recall  
   - Not recommended for deployment

6. **Tiny CNN (Multiclass)** — 55.56%  
   - Moderate performance  
   - Fails to detect BACKGROUND reliably  
   - Dataset too small for 3-class CNN training

7. **Robust CNN (Multiclass)** — 33.33%  
   - Model collapse: predicts DRONE for every input  
   - Severe overfitting due to dataset size  
   - Unusable with current training data

## Notes & Practical Insights

- **More complex models fail due to extremely limited data.**  
  CNNs—especially the Robust CNN—require hundreds or thousands of audio samples to generalize well. With only ~90 samples (30 per class), deep networks collapse or overfit, while traditional ML models remain stable and reliable. Theoretically they should perform better if more data can be provided.

- **Traditional ML models outperform CNNs with small datasets.**  
  RandomForest, XGBoost and SVM extract statistical structure from MFCCs efficiently and are far less data-hungry, making them ideal for early prototypes and low-data field environments.

- **Binary classification dramatically improves performance.**  
  Removing the HELICOPTER vs BACKGROUND distinction simplifies the problem, enabling higher accuracy and more robust models for real-time detection on constrained hardware given that the purpose is exclusively for detecting drones.

- **Tiny CNN (Binary) is the only CNN viable at the current dataset size.**  
  Its lightweight architecture avoids overfitting and provides competitive performance—but still trails behind the best ML models in stability and reliability.

- **Multiclass CNNs (both tiny and robust) are not usable yet.**  
  They fail to learn BACKGROUND and HELICOPTER patterns consistently due to insufficient data, producing class collapse and unreliable predictions.

- **High-capacity models consume more power and compute—bad for field deployment.**  
  The ESP32-S3 has limited RAM, flash, and compute capability.  
  - Robust CNN (~500k params) is too large, slow, and energy-demanding  
  - Tiny CNN (~50k params) is feasible  
  - Traditional ML models (RF, XGBoost, SVM) are *extremely lightweight* and fast

- **Battery life and thermal limits strongly favor traditional ML models.**  
  RandomForest, XGBoost and SVM require minimal CPU cycles for inference, making them ideal for an 8-hour mission runtime on a 2,000 mAh Li-Po battery.

- **Feature extraction cost matters more than model cost.**  
  MFCC computation often consumes more power than model inference itself.  
  Traditional ML + MFCC offers the best balance between accuracy and MCU feasibility.

- **For the soldier-worn vest system, reliability > sophistication.**  
  False negatives (missing a drone) are the most dangerous failure mode.  
  The highest, most consistent DRONE recall currently comes from:  
  - RandomForest (Binary)  
  - XGBoost (Binary)  
  - Tiny CNN (Binary)

- **Field noise conditions favor simpler models.**  
  Wind, footsteps, explosions, voices, vehicle noise, and weapon fire add acoustic chaos.  
  Traditional ML methods are more resilient to unpredictable noise with small datasets.

- **Scaling the dataset will completely change the ranking.**  
  - With *hundreds of new drone recordings*, CNNs—especially the robust architecture—would eventually outperform ML models.  
  - For now, with the data you have, traditional ML is the correct strategic choice.

- **Binary models align best with mission needs.**  
  The soldier does not need to know *what* kind of drone it is—only **“drone detected: yes/no + direction”**.  
  Binary classification simplifies computation and improves reliability in combat conditions.

- **MFCC was invented for human speech**
    It removes pitch, harmonic detail, fine spectral structure and Compresses everything into 12–20 coefficients, other methods for converting audio to a format digestible for models might be

## 🔧 Future Improvements to the Audio Processing Pipeline

To further increase overall model accuracy and robustness—especially as the system moves toward real embedded deployment—the following improvements are recommended.

---

## Recommended Processing Pipelines changes (Per Model Type)**

### **For Traditional ML Models (RandomForest, XGBoost, SVM)**
**Pipeline:**
```
Audio → STFT → MFCC + Spectral Statistics → RF / SVM / XGBoost
```
- MFCC + spectral features work extremely well with small datasets  
- Produces low-dimensional, stable features  
- Ideal for embedded inference and classical ML algorithms  

---

### **For Tiny CNN (Binary Drone Detection)**
**Pipeline:**
```
Audio → Log-Mel Spectrogram (32–64 bands) → Tiny CNN
```
- CNNs learn patterns better from 2D spectrograms than from MFCC  
- Log-Mel preserves harmonic and frequency structure relevant to drone sounds  
- Works well even with limited data

---

### **For Multiclass CNN (Once dataset size increases significantly)**
**Pipeline:**
```
Audio → Log-Mel Spectrogram → Medium-size CNN
```
- A larger dataset enables CNNs to outperform ML models  
- Multiclass tasks require richer features  
- Log-Mel spectrograms preserve full acoustic detail needed for distinguishing drone types vs helicopter vs background  

---

## Harmonic Information Matters for Drone Detection

Drone acoustics contain distinct **harmonic comb patterns**, including:

- **Propeller RPM** → strong fundamental tone  
- **Blade count** → evenly spaced harmonics  
- **Motor & frame resonance** → sidebands and modulations  

These harmonic structures are essential for reliable drone detection.

**MFCC suppresses or removes most of these harmonics.**  
**Log-Mel Spectrograms preserve them.**

This is why academic drone-detection systems overwhelmingly use:

- **Mel-Spectrograms**  
- **STFT Spectrograms**  
- **CQT (Constant-Q Transform)**  

---
