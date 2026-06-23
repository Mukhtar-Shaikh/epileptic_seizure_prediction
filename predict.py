"""
predict.py — Inference and evaluation on new EEG data.

Usage:
    python predict.py --input data/Bonn/S/S001.txt --model svm --dataset bonn
    python predict.py --input data/CHBMIT/chb01/chb01_01.edf --model xgboost --dataset chbmit
    python predict.py --evaluate --dataset bonn --model all
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from utils.config import MODELS_DIR

logger = get_logger("Predict")


def load_signal(path: str, dataset: str = "bonn") -> tuple:
    fs_map = {"bonn": 173.61, "chbmit": 256.0, "tuh": 250.0}
    fs = fs_map.get(dataset, 256.0)

    if path.endswith(".edf"):
        try:
            import mne
            mne.set_log_level("WARNING")
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            return raw.get_data(), raw.info["sfreq"]
        except ImportError:
            raise ImportError("mne required for EDF files: pip install mne")

    elif path.endswith(".npy"):
        data = np.load(path)
        return data if data.ndim > 1 else data[np.newaxis, :], fs

    elif path.endswith(".txt") or path.endswith(".dat"):
        data = np.loadtxt(path)
        return data[np.newaxis, :] if data.ndim == 1 else data, fs

    else:
        raise ValueError(f"Unsupported file format: {path}")


def preprocess_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    from preprocessing.filtering import apply_filters
    from preprocessing.normalization import normalize
    filtered = apply_filters(signal, fs)
    return normalize(filtered)


def extract_features_from_windows(signal: np.ndarray, fs: float) -> np.ndarray:
    from utils.helpers import segment_signal
    from features.hybrid_features import compute_hybrid_features, replace_nan_inf

    window_samples = int(1.0 * fs)
    step = int(window_samples * 0.5)
    n = signal.shape[-1]
    segments = []
    start = 0
    while start + window_samples <= n:
        seg = signal[..., start: start + window_samples]
        segments.append(seg)
        start += step

    if not segments:
        raise RuntimeError("Signal too short to segment.")

    X = np.array(segments)
    features = compute_hybrid_features(X, fs)
    return replace_nan_inf(features)


def load_model(model_name: str):
    import joblib
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Train it first with train.py")
    obj = joblib.load(path)
    if hasattr(obj, "predict"):
        return obj
    if isinstance(obj, dict):
        class WrappedModel:
            def __init__(self, d):
                self._m = d.get("model") or d.get("rf") or d.get("svm")
                self._s = d.get("scaler")
            def predict(self, X):
                Xs = self._s.transform(X) if self._s else X
                return self._m.predict(Xs)
            def predict_proba(self, X):
                Xs = self._s.transform(X) if self._s else X
                return self._m.predict_proba(Xs)
        return WrappedModel(obj)
    raise RuntimeError(f"Cannot load model from {path}")


def predict_file(signal_path: str, model_name: str, dataset: str = "bonn") -> dict:
    signal, fs = load_signal(signal_path, dataset)
    signal = preprocess_signal(signal, fs)
    features = extract_features_from_windows(signal, fs)

    model = load_model(model_name)
    y_pred = model.predict(features)
    y_proba = model.predict_proba(features) if hasattr(model, "predict_proba") else None

    seizure_proba = y_proba[:, 1] if y_proba is not None and y_proba.ndim == 2 else (y_proba or [0.0] * len(y_pred))
    seizure_windows = int(y_pred.sum())
    total_windows = len(y_pred)
    seizure_prob_mean = float(np.mean(seizure_proba))
    is_seizure = seizure_prob_mean >= 0.5

    result = {
        "file": os.path.basename(signal_path),
        "model": model_name,
        "total_windows": total_windows,
        "seizure_windows": seizure_windows,
        "seizure_rate": round(seizure_windows / max(total_windows, 1), 3),
        "mean_seizure_probability": round(seizure_prob_mean, 4),
        "prediction": "SEIZURE" if is_seizure else "NORMAL",
        "probabilities": seizure_proba.tolist() if hasattr(seizure_proba, 'tolist') else seizure_proba,
    }
    return result


def evaluate_all_models(dataset: str = "bonn", model_names: list = None):
    from train import load_dataset, preprocess
    from features.hybrid_features import compute_hybrid_features, replace_nan_inf
    from utils.helpers import train_val_test_split
    from utils.config import TEST_SIZE, VAL_SIZE, RANDOM_STATE
    from evaluation.comparison import compare_models, rank_models

    fs_map = {"bonn": 173.61, "chbmit": 256.0, "tuh": 250.0}
    fs = fs_map[dataset]

    X, y = load_dataset(dataset)
    X = preprocess(X, fs)
    _, _, X_test, _, _, y_test = train_val_test_split(X, y, TEST_SIZE, VAL_SIZE, RANDOM_STATE)
    X_feat = replace_nan_inf(compute_hybrid_features(X_test, fs))

    names = model_names or ["svm", "random_forest", "xgboost"]
    models = {}
    for name in names:
        try:
            models[name.upper()] = load_model(name)
        except FileNotFoundError as e:
            logger.warning(str(e))

    if not models:
        raise RuntimeError("No trained models found. Run train.py first.")

    df, _ = compare_models(models, X_feat, y_test)
    print(df.to_string())
    return df


def main():
    parser = argparse.ArgumentParser(description="EEG Seizure Prediction Inference")
    parser.add_argument("--input", type=str, default=None, help="Path to EEG file")
    parser.add_argument("--model", type=str, default="svm", help="Model name: svm, random_forest, xgboost")
    parser.add_argument("--dataset", choices=["bonn", "chbmit", "tuh"], default="bonn")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation on test set")
    args = parser.parse_args()

    if args.evaluate:
        model_names = ["svm", "random_forest", "xgboost"]
        evaluate_all_models(args.dataset, model_names)
    elif args.input:
        result = predict_file(args.input, args.model, args.dataset)
        print("\n=== Prediction Result ===")
        for k, v in result.items():
            if k != "probabilities":
                print(f"  {k:35s}: {v}")
        print(f"  {'Prediction':35s}: *** {result['prediction']} ***")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
