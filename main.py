"""
main.py — Entry point for the EEG Seizure Prediction GUI and CLI.

Usage:
    python main.py                        # Launch GUI dashboard
    python main.py --cli train --dataset bonn --model all
    python main.py --cli predict --input data/Bonn/S/S001.txt
    python main.py --cli demo             # Run demo with synthetic data
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger

logger = get_logger("Main")


def run_gui():
    try:
        import tkinter as tk
        tk.Tk().destroy()
    except Exception as e:
        logger.error(f"Tkinter not available: {e}")
        print("ERROR: GUI requires tkinter. Run: sudo apt-get install python3-tk")
        sys.exit(1)
    from gui.dashboard import run_dashboard
    logger.info("Starting GUI dashboard...")
    run_dashboard()


def run_demo():
    """Demo with synthetic EEG data to verify the pipeline works end-to-end."""
    print("\n" + "=" * 60)
    print("EEG SEIZURE PREDICTION — DEMO MODE")
    print("=" * 60)
    print("Generating synthetic EEG data...")

    np.random.seed(42)
    fs = 256.0
    n_channels = 4
    n_windows = 200
    window_samples = int(fs)

    X = np.random.randn(n_windows, n_channels, window_samples).astype(np.float32)
    X[:100, :, 50:150] += 5 * np.sin(np.linspace(0, 10 * np.pi, 100))
    y = np.array([1] * 100 + [0] * 100)

    print(f"Generated: {X.shape} segments, {y.sum()} seizure, {(y==0).sum()} normal")
    print("\nExtracting hybrid features...")

    from features.hybrid_features import compute_hybrid_features, replace_nan_inf
    X_feat = compute_hybrid_features(X, fs)
    X_feat = replace_nan_inf(X_feat)
    print(f"Feature matrix: {X_feat.shape}")

    print("\nRunning EMBO feature selection...")
    from optimization.embo import EMBO
    embo = EMBO(n_population=10, n_iterations=10)
    X_opt = embo.fit_transform(X_feat, y)
    print(f"EMBO selected: {X_opt.shape[1]} features")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    print("\nTraining models...")
    models = {}

    from models.svm_model import SVMModel
    svm = SVMModel()
    svm.fit(X_train, y_train)
    models["SVM"] = svm
    print("  ✓ SVM trained")

    from models.random_forest import RandomForestModel
    rf = RandomForestModel(params={"n_estimators": 50, "max_depth": 10, "random_state": 42})
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    print("  ✓ Random Forest trained")

    try:
        from models.xgboost_model import XGBoostModel
        xgb = XGBoostModel(params={"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4,
                                    "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42})
        xgb.fit(X_train, y_train)
        models["XGBoost"] = xgb
        print("  ✓ XGBoost trained")
    except ImportError:
        print("  ⚠ XGBoost not installed, skipping")

    print("\nEvaluating models...")
    from evaluation.metrics import compute_all_metrics, print_metrics
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        m = compute_all_metrics(y_test, y_pred, y_proba)
        print_metrics(m, name)

    print("\nTraining HDQL controller...")
    best_model = list(models.values())[0]
    proba = best_model.predict_proba(X_test)
    from models.hdql import HDQL
    hdql = HDQL(episodes=20)
    hdql.train(proba, y_test)
    hdql_preds = hdql.get_final_predictions(proba)
    m_hdql = compute_all_metrics(y_test, hdql_preds)
    print_metrics(m_hdql, "HDQL")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE — Pipeline verified successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Place dataset files in data/Bonn/, data/CHBMIT/, or data/TUH/")
    print("  2. Train: python train.py --dataset bonn --model all --embo")
    print("  3. Predict: python predict.py --input <file> --model svm")
    print("  4. GUI: python main.py")


def main():
    parser = argparse.ArgumentParser(description="EEG Seizure Prediction System")
    parser.add_argument("--cli", choices=["train", "predict", "demo"], default=None,
                        help="Run in CLI mode instead of launching GUI")
    parser.add_argument("--dataset", choices=["bonn", "chbmit", "tuh"], default="bonn")
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--embo", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    if args.cli is None:
        run_gui()
    elif args.cli == "demo":
        run_demo()
    elif args.cli == "train":
        train_args = ["--dataset", args.dataset, "--model", args.model]
        if args.patient:
            train_args += ["--patient", args.patient]
        if args.embo:
            train_args.append("--embo")
        if args.deep:
            train_args.append("--deep")
        if args.augment:
            train_args.append("--augment")
        sys.argv = ["train.py"] + train_args
        from train import main as train_main
        train_main()
    elif args.cli == "predict":
        predict_args = ["--dataset", args.dataset, "--model", args.model]
        if args.input:
            predict_args += ["--input", args.input]
        else:
            predict_args.append("--evaluate")
        sys.argv = ["predict.py"] + predict_args
        from predict import main as predict_main
        predict_main()


if __name__ == "__main__":
    main()
