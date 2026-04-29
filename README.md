# AI-BASED MULTIMODAL PHYSIOLOGICAL SIGNAL ANALYTICS FOR STRESS AND RELAXATION STATE DETECTION

A machine learning pipeline for binary stress detection using ECG and respiration signals from the **WESAD** dataset. The pipeline is designed to be leakage-aware, using subject-wise cross-validation to ensure genuine generalization to unseen individuals.
---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Evaluation Protocol](#evaluation-protocol)
- [Results](#results)
- [How to Run](#how-to-run)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project detects physiological stress by processing raw biosignals and extracting handcrafted HRV and respiration features. A 1D CNN deep learning baseline is also included for comparison. The core task is binary classification: **Stress (label 2)** vs. **Non-Stress (labels 1 & 3)**.

**Best interpretable model:** SVM with RBF kernel — F1 `0.5885 ± 0.2063`, ROC-AUC `0.8355 ± 0.1033`  
**Best overall model:** 1D CNN — F1 `0.699 ± 0.115`, ROC-AUC `0.854 ± 0.087`

---

## Dataset

**Source:** [WESAD (Wearable Stress and Affect Detection)](https://archive.ics.uci.edu/ml/datasets/WESAD)

| Property | Value |
|---|---|
| Subjects | 15 (S2–S17, excluding S12) |
| Signals used | Chest ECG, Chest Respiration |
| Sampling rate | 700 Hz |
| Window size | 60 seconds (42,000 samples/window) |
| Total windows | 599 |
| Labels | 1 = Baseline, 2 = Stress, 3 = Relaxed |

**Binary label mapping:**

| Original Label | Binary Label |
|---|---|
| 2 (Stress) | 1 — Stress |
| 1, 3 (Baseline / Relaxed) | 0 — Non-Stress |

**Class distribution:** 182 stress windows (30.4%) · 417 non-stress windows (69.6%)

> Raw `.pkl` files are expected at `data/raw/WESAD/S*/S*.pkl`

---

## Project Structure

```
.
├── data/
│   ├── raw/WESAD/               # Raw WESAD .pkl files
│   └── processed/
│       ├── windows_features.csv             # 6 base features
│       └── windows_features_engineered.csv  # 9 engineered features + metadata
├── src/
│   ├── preprocessing.py         # Windowing and label assignment
│   ├── feature_extraction.py    # ECG/HRV and respiration feature extraction
│   ├── models.py                # Classical ML model definitions
│   └── utils.py                 # StandardScaler normalization utilities
├── notebooks/
│   ├── 01–04                    # Initial multiclass experiments (6 features)
│   ├── 11–14                    # Final binary pipeline (9 engineered features)
│   ├── 14_subject_wise_5fold_cv_with_fi.ipynb  # CV + feature importance
│   ├── 15_*.ipynb               # 1D CNN deep learning baseline
│   ├── 16_*.ipynb               # ECG-only / RESP-only / multimodal comparison
│   └── 17_*.ipynb               # Learning curves
└── README.md
```

---

## Feature Engineering

### Base ECG / HRV Features

| Feature | Description |
|---|---|
| `Mean_RR` | Mean RR interval (seconds) from detected R-peaks |
| `SDNN` | Standard deviation of RR intervals |
| `RMSSD` | Root mean square of successive RR differences |
| `Mean_HR` | Mean heart rate in BPM (`60 / Mean_RR`) |

### Base Respiration Features

| Feature | Description |
|---|---|
| `Resp_Rate` | Breathing rate (breaths/min) via zero-crossing |
| `Resp_Variability` | Standard deviation of the respiration signal |

### Engineered Autonomic Features

| Feature | Formula |
|---|---|
| `HRV_HR_Ratio` | `RMSSD / Mean_HR` |
| `Resp_Regularity` | `1 / (Resp_Variability + 1e-6)` |
| `Autonomic_Index` | `RMSSD + SDNN - Mean_HR` |

**Implementation notes:**
- ECG is z-score normalized before R-peak detection.
- R-peaks detected via `scipy.signal.find_peaks` with minimum distance `0.6 × fs` (0.6 s).
- Features scaled with `StandardScaler` fitted **on training subjects only** to prevent data leakage.

---

## Models

### Classical ML (Handcrafted Features)

| Model | Key Parameters |
|---|---|
| Logistic Regression | `max_iter=1000, random_state=42` |
| SVM (RBF kernel) | `kernel='rbf', probability=True, random_state=42` |
| Random Forest | `n_estimators=300, random_state=42` |

### Deep Learning Baseline — 1D CNN

- **Input shape:** `(2, 42000)` — raw ECG + Respiration
- **Architecture:** Three Conv1D blocks (2→32→64→128 channels), BatchNorm + ReLU + MaxPool, `AdaptiveAvgPool1d(16)`, FC layers `2048→128→2`
- **Training:** 25 epochs · batch size 16 · lr `1e-3` · dropout `0.4` · Adam + cosine annealing · weight decay `1e-4`
- **Total parameters:** 321,570

---

## Evaluation Protocol

Final evaluation uses **subject-wise 5-fold cross-validation** to prevent subject leakage.

| Property | Value |
|---|---|
| Splitter | `KFold(n_splits=5, shuffle=True, random_state=42)` |
| Split unit | Subjects (not windows) |
| Train windows/fold | ~478–480 |
| Test windows/fold | ~119–121 |

**Fold assignments:**

| Fold | Test Subjects |
|---|---|
| 1 | S10, S4, S6 |
| 2 | S16, S3, S8 |
| 3 | S11, S13, S9 |
| 4 | S15, S2, S5 |
| 5 | S14, S17, S7 |

**Metrics reported:** Accuracy · Precision · Recall · F1 · ROC-AUC · Balanced Accuracy · Specificity

> F1 and balanced accuracy are emphasized due to class imbalance (stress = 30.4%).

---

## Results

### Handcrafted Feature Models (Subject-wise 5-Fold CV)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Balanced Acc | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7627 ± 0.0681 | 0.6301 ± 0.1063 | 0.5748 ± 0.1930 | 0.5830 ± 0.1310 | 0.8336 ± 0.0764 | 0.7096 ± 0.0932 | 0.8443 ± 0.0800 |
| **SVM RBF** | **0.7875 ± 0.0707** | **0.6535 ± 0.0851** | 0.5805 ± 0.2696 | **0.5885 ± 0.2063** | **0.8355 ± 0.1033** | **0.7291 ± 0.1250** | **0.8778 ± 0.0424** |
| Random Forest | 0.7609 ± 0.0876 | 0.5948 ± 0.1412 | 0.5854 ± 0.2423 | 0.5764 ± 0.1944 | 0.8175 ± 0.1086 | 0.7113 ± 0.1240 | 0.8372 ± 0.0795 |

### 1D CNN Baseline

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0.793 ± 0.084 | 0.653 ± 0.156 | 0.779 ± 0.132 | 0.699 ± 0.115 | 0.854 ± 0.087 |

### Modality Comparison (Best F1)

| Modality | Best F1 |
|---|---|
| ECG + Respiration | 0.589 ± 0.206 |
| ECG only | 0.576 ± 0.193 |
| Respiration only | 0.495 ± 0.089 |

### Feature Importance (Random Forest, averaged across 5 folds)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `RMSSD` | 0.1834 |
| 2 | `Resp_Regularity` | 0.1412 |
| 3 | `Resp_Variability` | 0.1362 |
| 4 | `HRV_HR_Ratio` | 0.1257 |
| 5 | `SDNN` | 0.1210 |

> `RMSSD` is the most influential feature across all models — physiologically meaningful as stress directly impacts autonomic nervous system activity and short-term heart-rate variability.

---

## How to Run

**1. Install dependencies**
```bash
pip install numpy pandas scipy scikit-learn torch matplotlib seaborn
```

**2. Prepare data**

Place WESAD `.pkl` files in:
```
data/raw/WESAD/S2/S2.pkl
data/raw/WESAD/S3/S3.pkl
...
```

**3. Run preprocessing and feature extraction**
```bash
python src/preprocessing.py
python src/feature_extraction.py
```

**4. Run the final evaluation notebook**
```
notebooks/14_subject_wise_5fold_cv_with_fi.ipynb
```

**5. (Optional) Run the 1D CNN baseline**
```
notebooks/15_*.ipynb
```

---

## Limitations & Future Work

- **Small dataset:** Only 15 subjects — results may not generalize broadly.
- **Signals:** Only ECG and respiration are used; EDA, EMG, accelerometer, and temperature signals from WESAD were excluded.
- **No hyperparameter tuning:** Classical models use default/fixed hyperparameters.
- **Future directions:**
  - Validate on larger, more diverse cohorts.
  - Incorporate additional modalities (EDA, Temp).
  - Systematic hyperparameter search (grid/random/Bayesian).
  - Explore attention-based or transformer architectures for the deep learning baseline.

---

## Citation

> Philip Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection", ACM ICMI 2018.
