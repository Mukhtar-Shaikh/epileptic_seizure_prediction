"""
train.py — Full training pipeline for EEG Seizure Prediction.

Usage:
    python train.py --dataset bonn --model all
    python train.py --dataset chbmit --patient chb01 --model all --embo
    python train.py --dataset chbmit --patient chb01 --model all --embo --max_files 10
    python train.py --dataset bonn --model xgboost --embo --deep
"""

import argparse
import sys
import os
import gc
import time
import warnings
import numpy as np

# Suppress slow/noisy warnings from nolds and sklearn R²
warnings.filterwarnings("ignore", module="nolds")
warnings.filterwarnings("ignore", message="R\\^2 score is not well-defined")
warnings.filterwarnings("ignore", message="signal has very low mean frequency")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from utils.helpers import train_val_test_split, class_counts
from utils.config import RANDOM_STATE, TEST_SIZE, VAL_SIZE

logger = get_logger("Train")


# ─────────────────────────────────────────────────────────
# STREAMING FEATURE EXTRACTION  (memory-safe for CHB-MIT)
# ─────────────────────────────────────────────────────────

def extract_features_streaming_chbmit(
    patient: str,
    fs: float,
    window_sec: float = 1.0,
    overlap: float = 0.5,
    max_files: int = None,
) -> tuple:
    """
    Process CHB-MIT one EDF file at a time.
    Filters → normalises → extracts features → frees raw signal.
    Returns (X_feat, y) where X_feat is the full feature matrix.
    Peak RAM ≈ one file's worth of data at a time.
    """
    from datasets.chbmit_loader import CHBMITLoader
    from preprocessing.filtering import apply_filters
    from preprocessing.normalization import normalize
    from features.hybrid_features import compute_hybrid_features, replace_nan_inf

    loader = CHBMITLoader()
    patients = [patient] if patient else None

    X_parts, y_parts = [], []
    total_windows = 0

    for segs, seg_labels in loader.iter_segments(patients, window_sec, overlap, max_files):
        # segs: (n_windows, n_channels, n_samples)  float32
        logger.info(f"  Processing {len(segs)} windows …")

        # Preprocess each window in-place (filter + normalize)
        processed = []
        for seg in segs:
            f = apply_filters(seg.astype(np.float64), fs)
            n = normalize(f).astype(np.float32)
            processed.append(n)
        processed = np.array(processed, dtype=np.float32)
        del segs
        gc.collect()

        # Extract features (result is small: ~212 floats per window)
        feats = compute_hybrid_features(processed, fs)
        feats = replace_nan_inf(feats).astype(np.float32)
        del processed
        gc.collect()

        X_parts.append(feats)
        y_parts.append(seg_labels)
        total_windows += len(feats)
        logger.info(f"  Running total: {total_windows} windows, seizure={seg_labels.sum()}")

    if not X_parts:
        raise RuntimeError(
            "No data loaded. Check data/CHBMIT/<patient>/ contains .edf files "
            "and a summary .txt file."
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    logger.info(f"Feature matrix: {X.shape}  labels: {class_counts(y)}")
    return X, y


# ─────────────────────────────────────────────────────────
# STANDARD (in-memory) PIPELINE — Bonn / TUH
# ─────────────────────────────────────────────────────────

def load_dataset(dataset_name: str, patient: str = None,
                 window_sec: float = 1.0, overlap: float = 0.5,
                 max_files: int = None):
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_name == "bonn":
        from datasets.bonn_loader import BonnLoader
        loader = BonnLoader()
        X_win, y_win = loader.build_windowed_dataset(window_sec, overlap)
        if X_win.ndim == 1:
            raise RuntimeError("Bonn dataset not found. Place files in data/Bonn/{Z,O,N,F,S}/")
        return X_win, y_win

    elif dataset_name == "tuh":
        from datasets.tuh_loader import TUHLoader
        loader = TUHLoader()
        X, y = loader.build_dataset(window_sec, overlap,
                                    max_files=max_files or 50)
        if X.ndim == 1:
            raise RuntimeError("TUH dataset not found. Place EDF+TSE files in data/TUH/")
        return X, y

    else:
        raise ValueError(f"Unknown dataset for in-memory load: {dataset_name}")


def preprocess(X: np.ndarray, fs: float) -> np.ndarray:
    from preprocessing.filtering import apply_filters
    from preprocessing.normalization import normalize
    logger.info(f"Preprocessing {len(X)} segments…")
    out = []
    for seg in X:
        filtered = apply_filters(seg.astype(np.float64), fs)
        out.append(normalize(filtered).astype(np.float32))
    return np.array(out, dtype=np.float32)


def augment(X: np.ndarray, y: np.ndarray, fs: float) -> tuple:
    from augmentation.augmentation import augment_batch, apply_smote
    logger.info("Augmenting dataset…")
    X_aug, y_aug = augment_batch(X, y, fs=fs)
    X_aug, y_aug = apply_smote(X_aug, y_aug)
    logger.info(f"After augmentation: {class_counts(y_aug)}")
    return X_aug, y_aug


def extract_features(X: np.ndarray, fs: float,
                     use_deep: bool = False,
                     y_train: np.ndarray = None) -> tuple:
    from features.hybrid_features import compute_hybrid_features, replace_nan_inf
    logger.info("Extracting hybrid features…")
    deep_emb, deep_model = None, None

    if use_deep and y_train is not None:
        logger.info("Training STAtt deep embedding model…")
        from features.statt_embeddings import train_statt, extract_statt_embeddings
        deep_model = train_statt(X, y_train)
        deep_emb = extract_statt_embeddings(deep_model, X)
        logger.info(f"STAtt embeddings: {deep_emb.shape}")

    features = compute_hybrid_features(X, fs, include_deep=deep_emb is not None,
                                       deep_embeddings=deep_emb)
    features = replace_nan_inf(features).astype(np.float32)
    logger.info(f"Feature matrix: {features.shape}")
    return features, deep_model


def run_embo(X: np.ndarray, y: np.ndarray) -> tuple:
    from optimization.embo import EMBO
    logger.info("Running EMBO feature optimisation…")
    embo = EMBO()
    X_opt = embo.fit_transform(X, y)
    logger.info(f"EMBO: {X_opt.shape[1]} / {X.shape[1]} features selected")
    return X_opt, embo


def train_models(X_train, y_train, X_val, y_val, model_names):
    trained = {}

    if "svm" in model_names:
        from models.svm_model import SVMModel
        logger.info("Training SVM…")
        t0 = time.time()
        svm = SVMModel()
        svm.fit(X_train, y_train)
        logger.info(f"SVM done in {time.time()-t0:.1f}s")
        svm.save()
        trained["SVM"] = svm

    if "rf" in model_names:
        from models.random_forest import RandomForestModel
        logger.info("Training Random Forest…")
        t0 = time.time()
        rf = RandomForestModel()
        rf.fit(X_train, y_train)
        logger.info(f"RF done in {time.time()-t0:.1f}s")
        rf.save()
        trained["Random Forest"] = rf

    if "xgboost" in model_names:
        from models.xgboost_model import XGBoostModel
        logger.info("Training XGBoost…")
        t0 = time.time()
        xgb = XGBoostModel()
        xgb.fit(X_train, y_train, X_val, y_val)
        logger.info(f"XGBoost done in {time.time()-t0:.1f}s")
        xgb.save()
        trained["XGBoost"] = xgb

    return trained


def evaluate_and_compare(models: dict, X_test: np.ndarray, y_test: np.ndarray):
    from evaluation.comparison import compare_models, rank_models
    from evaluation.plots import (plot_confusion_matrix, plot_roc_curve,
                                  plot_pr_curve, plot_feature_importance, plot_shap)

    logger.info("Running model comparison…")
    df_results, _ = compare_models(models, X_test, y_test)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(df_results.to_string())
    print("=" * 70)
    print("\nModel Rankings:")
    print(rank_models(df_results).to_string())

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        plot_confusion_matrix(y_test, y_pred, name)
        if y_proba is not None:
            plot_roc_curve(y_test, y_proba, name)
            plot_pr_curve(y_test, y_proba, name)
        if hasattr(model, "get_feature_importance"):
            plot_feature_importance(model.get_feature_importance(), model_name=name)
        if name == "XGBoost" and hasattr(model, "get_shap_values"):
            sv = model.get_shap_values(X_test[:100])
            if sv is not None and len(sv) > 0:
                plot_shap(sv, X_test[:100], model_name=name)

    logger.info("Plots saved to results/plots/")
    return df_results


def train_hdql(models: dict, X_test: np.ndarray, y_test: np.ndarray):
    from models.hdql import HDQL
    from evaluation.metrics import compute_all_metrics, print_metrics
    logger.info("Training HDQL controller…")
    best_model = list(models.values())[0]
    proba = best_model.predict_proba(X_test)
    hdql = HDQL()
    hdql.train(proba, y_test)
    final_preds = hdql.get_final_predictions(proba)
    print_metrics(compute_all_metrics(y_test, final_preds), "HDQL Final")
    return hdql


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EEG Seizure Prediction Training Pipeline")
    parser.add_argument("--dataset",   choices=["bonn", "chbmit", "tuh"], default="bonn")
    parser.add_argument("--patient",   type=str,   default=None,
                        help="Patient ID for CHB-MIT (e.g. chb01)")
    parser.add_argument("--model",     default="all",
                        help="svm, rf, xgboost, or all")
    parser.add_argument("--embo",      action="store_true",
                        help="Apply EMBO feature selection")
    parser.add_argument("--deep",      action="store_true",
                        help="Include deep learning embeddings")
    parser.add_argument("--augment",   action="store_true",
                        help="Apply data augmentation")
    parser.add_argument("--window",    type=float, default=1.0,
                        help="Window size in seconds (default 1.0)")
    parser.add_argument("--overlap",   type=float, default=0.5,
                        help="Window overlap fraction (default 0.5)")
    parser.add_argument("--max_files", type=int,   default=None,
                        help="Max EDF files to use (CHB-MIT/TUH). "
                             "Recommended: 10 on 8 GB RAM, 20 on 16 GB RAM")
    args = parser.parse_args()

    fs_map = {"bonn": 173.61, "chbmit": 256.0, "tuh": 250.0}
    fs = fs_map[args.dataset]
    model_names = ["svm", "rf", "xgboost"] if args.model == "all" else [args.model]

    logger.info(f"Pipeline: dataset={args.dataset} models={model_names} "
                f"embo={args.embo} deep={args.deep} max_files={args.max_files}")

    # ── Load & feature-extract ────────────────────────────
    if args.dataset == "chbmit":
        # Streaming path — one file at a time, RAM-safe
        logger.info("Using streaming pipeline for CHB-MIT (memory-safe)")
        X_feat, y = extract_features_streaming_chbmit(
            patient=args.patient,
            fs=fs,
            window_sec=args.window,
            overlap=args.overlap,
            max_files=args.max_files,
        )
    else:
        # Standard in-memory path for Bonn / TUH
        X_raw, y = load_dataset(args.dataset, args.patient,
                                args.window, args.overlap, args.max_files)
        logger.info(f"Dataset: {X_raw.shape}  labels: {class_counts(y)}")
        X_proc = preprocess(X_raw, fs)
        del X_raw; gc.collect()

        if args.augment:
            X_train_r, _, X_test_r, y_train_r, _, y_test_raw = train_val_test_split(
                X_proc, y, TEST_SIZE, VAL_SIZE, RANDOM_STATE)
            X_train_r, y_train_r = augment(X_train_r, y_train_r, fs)
            X_proc = np.concatenate([X_train_r, X_test_r], axis=0)
            y = np.concatenate([y_train_r, y_test_raw], axis=0)

        X_feat, _ = extract_features(X_proc, fs, args.deep,
                                     y if args.deep else None)
        del X_proc; gc.collect()

    # ── Train / val / test split ──────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X_feat, y, TEST_SIZE, VAL_SIZE, RANDOM_STATE)
    logger.info(f"Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"  Train labels: {class_counts(y_train)}")
    logger.info(f"  Test  labels: {class_counts(y_test)}")

    # ── EMBO feature selection ────────────────────────────
    if args.embo:
        X_train, embo = run_embo(X_train, y_train)
        X_val  = embo.transform(X_val)
        X_test = embo.transform(X_test)

    # ── Model training ────────────────────────────────────
    models = train_models(X_train, y_train, X_val, y_val, model_names)

    # ── Evaluation ────────────────────────────────────────
    evaluate_and_compare(models, X_test, y_test)
    train_hdql(models, X_test, y_test)

    logger.info("Training pipeline complete!")
    logger.info("Trained models saved to: saved_models/")
    logger.info("Plots saved to:          results/plots/")


if __name__ == "__main__":
    main()
