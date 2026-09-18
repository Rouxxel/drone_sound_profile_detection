# Future Improvements

Roadmap ideas for **drone_sound_profile_detection**, from closing current gaps to new software features and hardware that better match the soldier-vest embedded target (ESP32-S3, 4× I²S mics, duty-cycled ML, ~8 h runtime).

Priorities below are guided by `README.md`, `documentation/USAGE_GUIDE.md`, and `exprmntl_insights.md`:

- **Binary detection** (drone vs not) fits the mission better than multiclass today.
- **Traditional ML (RF / XGBoost / SVM)** currently beats CNNs on ~90 samples; CNNs need more data.
- **Feature cost matters**: MFCC/DSP often costs more MCU time than inference.
- **Harmonics** (RPM, blade count, motor resonance) matter; MFCC alone suppresses much of that structure.

---

## 1. Fill Gaps & Harden What Exists

### Documentation & repo consistency
- Align `USAGE_GUIDE.md` with the current converters (`aud_csv_converter_trad_ml.py`, config-driven CNN paths) so examples match `README.md` / `configuration/config.json`.
- Document skip logic in `main.py` (when download / convert / EDA / train / test are skipped).
- Keep a short “recommended deploy model” note (today: **binary RandomForest or XGBoost**; Tiny CNN binary as CNN fallback).
- Link `exprmntl_insights.md` and this file from the root README.

### Evaluation & MLOps hygiene
- Replace single 80/20 split with **stratified k-fold** (or repeated hold-out) so rankings are less noisy on 6 samples/class.
- Track **precision / recall / F1 for DRONE**, not only overall accuracy (false negatives are the critical failure mode).
- Add a frozen **test set** never used for model selection; log metrics to a small results table (CSV/JSON) under `logs/` or `results/`.
- Unit tests for converters and feature extractors (extend `datasets/tests/`); smoke tests that load each saved model and run one prediction.
- Version datasets and models (hash of audio set + config snapshot written next to each `trained_models/` run).

### Data quality
- Enforce naming and class balance checks in the pipeline (fail early if a class is missing or too small).
- Record sample metadata (device, distance, environment, SNR) when collecting new audio.
- Deduplicate / near-duplicate detection so similar clips don’t leak across train/val.

---

## 2. Dataset Growth (Highest Impact)

Current bottleneck: **~30 files × 3 classes (~15 minutes of audio)**. Insights already show multiclass CNNs collapsing and Robust CNN being unusable.

| Goal | Why |
|------|-----|
| Hundreds of **DRONE** clips (FPV, fixed-wing loitering munitions: Lancet, KUB, V2U, etc.) | Improve DRONE recall; reduce field false negatives |
| Diverse **NO_DRONE** (wind, voices, vehicles, gunfire-like bursts, helicopters, generators) | Cut false alarms in combat-like noise |
| Distance / SNR labels (e.g. 100 m … 900 m) | Match claimed detection range; train/evaluate by distance |
| Multiple mics / channels | Prepare for 4-mic vest geometry and direction finding |
| On-device or field recordings at **16 kHz** | Match ESP32 capture rate; avoid train/deploy mismatch |

**Augmentation (until more real data exists):**
- Time shift, gain, mild noise, wind simulation, mic dropouts.
- SpecAugment / time–frequency masking for Log-Mel CNN paths.
- Mix background with quiet drone (SNR curriculum).

**Sources:** expand beyond the single Kaggle set; controlled range recordings; partner/shared military-open acoustic datasets where licenses allow.

---

## 3. Audio Features & Processing Pipeline

MFCC was designed for speech and compresses away pitch/harmonic detail that drones need. Insights recommend model-specific pipelines:

### Traditional ML (keep + enrich)
```
Audio → STFT → MFCC + spectral stats (+ HNR) → scaler → RF / SVM / XGBoost
```
- Keep current 21-dim frame features; add **pitch / F0**, **harmonic peak spacing**, **spectral flux**, **crest factor**.
- Optional **CQT** or harmonic–percussive stats tailored to propeller combs.
- Profile MFCC cost on MCU; consider lighter / fixed-point feature stacks for duty-cycled wake.

### Tiny CNN / future CNNs
```
Audio → Log-Mel Spectrogram (32–64+ bands) → Tiny CNN
```
- Prefer Log-Mel (or STFT/CQT) over MFCC-as-image so RPM harmonics remain visible.
- Align sample rate, hop, and window with the embedded DSP (16 kHz, same frame length as rolling buffer).

### Shared DSP (matches vest design)
- Digital notch filters + wind-noise suppression before features.
- Rolling **2–4 s** buffer for pre-trigger context.
- **Energy / VAD-like detector** → only then run full feature extract + ML (duty cycle).
- Per-mic **auto-calibration** (gain normalization) at startup; store gains for session.

---

## 4. Models & Training

### Near term
- Standardize on **binary** as the primary product path; keep multiclass as research once data grows.
- Hyperparameter search for RF / XGBoost / SVM with nested CV; export **one** “production” binary model + scaler.
- Class-weight / focal-style emphasis on **DRONE** recall.
- Quantization-aware training or post-training **int8** for Tiny CNN; tree pruning / leaf limits for RF size on flash.

### Medium term (more data)
- Revisit Robust CNN and deeper Log-Mel CNNs; transfer learning from pretrained audio CNNs (e.g. YAMNet-style or similar embeddings) then fine-tune.
- Lightweight architectures aimed at TinyML: depthwise separable CNNs, MobileNet-style audio backbones, or distilled students from a larger teacher.
- Ensemble only if latency/power budget allows (e.g. RF + Tiny CNN vote on ESP32 may be too heavy—prefer on a Pi gateway).

### Export for edge
- **TFLite** / **TensorFlow Lite Micro** for Tiny CNN.
- **ONNX** or **sklearn → C** / **micromlgen** / **emlearn** for trees/SVM where possible.
- Benchmark: latency (ms), RAM peak, flash size, mA draw per inference on target silicon.

---

## 5. Software Features (Product / Pipeline)

| Feature | Description |
|---------|-------------|
| Real-time inference CLI / daemon | Stream mic or WAV → features → model → JSON alert (label, confidence, timestamp) |
| Confidence + hysteresis | Require N consecutive positive windows; cooldown to reduce chatter |
| Direction estimate API | From 4-mic TDOA / beam energy → sector or angle for helmet alert payload |
| Config-driven pipeline | Already partly in `configuration/`; extend to DSP thresholds, duty-cycle, alert format |
| Alert adapter | Mock “helmet comms” sink (UART/BLE/MQTT) with schema: `{drone: bool, confidence, direction, ts}` |
| Field logging mode | Save anonymized clips + decisions for later retraining (opt-in) |
| Dashboard (optional) | Simple web/UI on a companion Pi for lab demos (live spectrogram + direction) |
| Continuous integration | Run converter unit tests + tiny train smoke on push |

---

## 6. Embedded Firmware & Duty-Cycled Inference

Target from README: **ESP32-S3**, 4× I²S MEMS @ 16 kHz, 2000 mAh Li-Po, ~90 g, €25–60.

Suggested firmware milestones:
1. **I²S multi-mic capture** + circular buffer (2–4 s).
2. **Energy detector** wake → feature extract → **binary ML** → sleep.
3. Wind foam + software wind/notch filters; startup mic gain calibration.
4. Output to helmet link (UART/BLE); direction from mic array.
5. Power profiling toward **~8 h** continuous; extend with deeper sleep, lower sample rate when idle, or secondary coin-cell RTC wake.

**Companion / lab platforms (not vest mass budget):**
- **Raspberry Pi 4/5 or Zero 2 W**: full Python pipeline, live debug, larger CNNs, MQTT gateway, UI.
- **NVIDIA Jetson / Coral** (optional): only if future multi-class / multi-sensor fusion needs GPU/TPU—overkill for binary RF on the vest.

---

## 7. Hardware & Modules (Better Results in the Field)

### Stay on the vest path (optimize current BOM)
- Higher-SNR MEMS I²S mics; matched front/back pairs; foam/windscreens per mic.
- Rigid mic mounts with known geometry (baseline for TDOA).
- Better enclosure: vibration isolation from vest fabric / body movement.
- Battery options: larger Li-Po or dual-cell for longer missions; power-path / charging module.
- Optional **IMU**: reject motion-induced noise or gate inference when sprinting.

### Array & ranging
- Calibrated 4-mic (or 6–8 mic) array for **360°** and finer bearing.
- Optional ultrasonic / RF cue as a *secondary* trigger (still acoustic-primary for stealth/cost).

### Lab & validation hardware
- Raspberry Pi + USB/I²S mic array for dataset collection and algorithm prototyping.
- Calibrated speaker / drone audio playback at known distances (100–900 m simulations where legal/safe).
- Environmental chamber / outdoor wind tests for false-alarm rates.

### Future modular “kit”
| Module | Role |
|--------|------|
| Acoustic core (ESP32-S3 + 4 mics) | Detect + direction + alert |
| Pi HAT / USB bridge | Training data capture, firmware update, field diagnostics |
| Helmet comms adapter | Protocol glue to existing soldier radios |
| Spare mic / foam packs | Maintainability in the field |

---

## 8. Suggested Priority Order

1. **More labeled field-like audio** (especially DRONE + hard negatives) + better metrics (DRONE recall).
2. **Binary production model** + MCU-friendly export + duty-cycled energy gate.
3. **Feature pipeline** aligned to harmonics (Log-Mel for CNN; enriched spectral/harmonic features for ML) and **16 kHz** match to hardware.
4. **4-mic direction** + alert schema; firmware on ESP32-S3.
5. **Docs/tests/config** cleanup so the repo matches what actually ships.
6. Scale data → revive deeper CNNs / transfer learning; optional Pi companion for demos and collection.

---

## 9. Success Criteria (Useful North Stars)

- DRONE recall and false-alarm rate characterized on **held-out field noise**, not only the Kaggle-style set.
- End-to-end latency from acoustic onset to helmet alert within a defined budget (e.g. &lt; 1–2 s including buffer context).
- Continuous operation **≥ 8 h** on the reference 2000 mAh pack with duty cycling.
- Detection useful in the **100–900 m** band under documented SNR conditions.
- Single documented “ship” artifact: binary model + scaler/DSP params + firmware build that fits flash/RAM.

---

*This document is a living roadmap, not a commitment. Prefer changes that improve reliability and power on the vest over complexity that only helps on large GPUs.*
