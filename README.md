# AI-Driven Real-Time EEG Analysis for Epileptic Seizure Prediction

A complete, modular, production-ready Python system for EEG-based epileptic seizure
prediction using Machine Learning, lightweight Deep Learning embeddings, and a
real-time GUI dashboard.

---

## Architecture

```
EEG Dataset → Preprocessing → Augmentation → Feature Extraction
      → EMBO Optimization → ML Models → Model Comparison
      → Deep Learning Embeddings → HDQL → GUI Real-Time Prediction
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd epileptic_seizure_prediction
pip install -r requirements.txt
```

> **Note:** `tkinter` is required for the GUI. On Linux: `sudo apt-get install python3-tk`
> On macOS it's included with Python. On Windows it's built-in.

### 2. Run the Demo (no dataset needed)

```bash
python main.py --cli demo
```

Generates synthetic EEG data and runs the full pipeline end-to-end.

### 3. Launch the GUI

```bash
python main.py
```

### 4. Train on Real Data

```bash
# Bonn dataset
python train.py --dataset bonn --model all --embo

# CHB-MIT (single patient)
python train.py --dataset chbmit --patient chb01 --model all

# With deep embeddings and augmentation
python train.py --dataset bonn --model all --embo --deep --augment
```

### 5. Predict on a New File

```bash
python predict.py --input data/Bonn/S/S001.txt --model svm --dataset bonn
python predict.py --input data/CHBMIT/chb01/chb01_01.edf --model xgboost --dataset chbmit
python predict.py --evaluate --dataset bonn --model all
```

---

## Dataset Setup

Place dataset files in the `data/` directory:

```
data/
├── Bonn/
│   ├── Z/    (100 .txt files, normal - eyes open)
│   ├── O/    (100 .txt files, normal - eyes closed)
│   ├── N/    (100 .txt files, interictal - hippocampus)
│   ├── F/    (100 .txt files, interictal - focus area)
│   └── S/    (100 .txt files, ictal/SEIZURE)
├── CHBMIT/
│   ├── chb01/
│   │   ├── chb01_01.edf
│   │   ├── chb01-summary.txt
│   │   └── ...
│   └── chb02/ ...
└── TUH/
    └── <patient_id>/
        ├── *.edf
        └── *.tse  (annotation files)
```

**Sources:**
- Bonn: https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/
- CHB-MIT: https://physionet.org/content/chbmit/1.0.0/
- TUH: https://isip.piconepress.com/projects/tuh_eeg/

---

## Project Structure

```
epileptic_seizure_prediction/
├── data/                        # Dataset storage
├── datasets/
│   ├── chbmit_loader.py         # CHB-MIT EDF loader with seizure annotations
│   ├── bonn_loader.py           # Bonn University 5-set loader
│   └── tuh_loader.py            # TUH EEG corpus loader
├── preprocessing/
│   ├── filtering.py             # Bandpass (0.5–45 Hz) + Notch (50/60 Hz)
│   ├── artifact_removal.py      # ICA + wavelet denoising
│   ├── normalization.py         # Z-score / minmax / robust
│   └── segmentation.py          # 1-second windows, 50% overlap
├── augmentation/
│   └── augmentation.py          # Gaussian noise, time shift, scaling, SMOTE
├── features/
│   ├── time_features.py         # Hjorth, ZCR, RMS, skew, kurtosis
│   ├── frequency_features.py    # Band powers, spectral entropy, peak freq
│   ├── wavelet_features.py      # DWT energy/entropy (db4, level 4)
│   ├── nonlinear_features.py    # ApEn, SampEn, PermEn, Hurst, Lyapunov
│   ├── hybrid_features.py       # Combined hybrid feature vector
│   └── statt_embeddings.py      # Spatial-Temporal Attention embeddings
├── optimization/
│   └── embo.py                  # Enhanced Multi-Objective Bean Optimization
├── models/
│   ├── svm_model.py             # SVM (RBF, C=10)
│   ├── random_forest.py         # Random Forest (300 trees)
│   ├── xgboost_model.py         # XGBoost (500 estimators) + SHAP
│   ├── cnn_embeddings.py        # Lightweight CNN embedder (64-dim)
│   ├── lstm_embeddings.py       # Lightweight BiLSTM embedder (64-dim)
│   ├── cnn_bilstm_statt.py      # CNN + BiLSTM + STAtt combined model
│   └── hdql.py                  # Hierarchical Dual Q-Learning controller
├── evaluation/
│   ├── metrics.py               # Accuracy, F1, ROC-AUC, MCC, Dice, etc.
│   ├── plots.py                 # Confusion matrix, ROC, PR, SHAP plots
│   └── comparison.py            # Multi-model comparison + ranking
├── gui/
│   ├── dashboard.py             # Main tkinter application window
│   ├── eeg_viewer.py            # Interactive EEG waveform viewer
│   ├── prediction_panel.py      # Real-time probability display
│   └── alerts.py                # Seizure alert log panel
├── utils/
│   ├── config.py                # All hyperparameters and paths
│   ├── logger.py                # Structured logging
│   └── helpers.py               # Segmentation, normalization, I/O utilities
├── train.py                     # Full training pipeline
├── predict.py                   # Inference and batch evaluation
├── main.py                      # GUI launcher + CLI router
├── requirements.txt
└── README.md
```

---

## Feature Engineering

| Category | Features |
|----------|----------|
| Time Domain | Mean, Variance, Std, RMS, Skewness, Kurtosis, Hjorth Activity/Mobility/Complexity, ZCR |
| Frequency | Delta/Theta/Alpha/Beta/Gamma band powers, Spectral Entropy, Peak Freq, Mean Freq |
| Wavelet | DWT Energy, DWT Entropy, Coefficient Mean/Std/Max (db4, L4) |
| Nonlinear | Approximate Entropy, Sample Entropy, Permutation Entropy, Petrosian FD, Hurst, Lyapunov |
| Deep | CNN-64, BiLSTM-64, STAtt-64 embeddings |

---

## Machine Learning Models

| Model | Key Config |
|-------|-----------|
| SVM | kernel=RBF, C=10, probability=True |
| Random Forest | n_estimators=300, max_depth=20 |
| XGBoost | n_estimators=500, lr=0.05, max_depth=8, SHAP |

---

## EMBO — Feature Selection

Enhanced Multi-Objective Bean Optimization selects the optimal feature subset by maximizing:

```
Fitness = α·Relevance - β·Redundancy + γ·Separability - δ·Cost
```

Default: α=0.4, β=0.3, γ=0.2, δ=0.1

---

## HDQL — Reinforcement Learning Controller

Hierarchical Dual Q-Learning reduces false alarms with two decision levels:

- **Level 1:** Normal / Pre-ictal → Continue / Alert
- **Level 2:** Pre-ictal / Ictal → Non-Seizure / Seizure

Q-update: `Q(s,a) = Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]`

---

## Performance Metrics

Accuracy, Sensitivity, Specificity, Precision, Recall, F1, ROC-AUC, PR-AUC,
False Prediction Rate, Detection Delay, MCC, Dice Score, Inference Time

---

## System Requirements

- Python 3.9+
- RAM: 8–16 GB (sufficient for all three datasets)
- GPU: Optional (PyTorch CPU mode works fine for lightweight embeddings)
- OS: Linux / macOS / Windows
