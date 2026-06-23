import numpy as np
import os
import joblib
from typing import Tuple, List
from utils.config import MODELS_DIR


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_model(model, name: str) -> str:
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    return path


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True) + 1e-8
    return (signal - mean) / std


def segment_signal(
    signal: np.ndarray,
    fs: float,
    window_sec: float = 1.0,
    overlap: float = 0.5,
) -> np.ndarray:
    window_samples = int(window_sec * fs)
    step = int(window_samples * (1 - overlap))
    segments = []
    start = 0
    while start + window_samples <= signal.shape[-1]:
        seg = signal[..., start : start + window_samples]
        segments.append(seg)
        start += step
    return np.array(segments) if segments else np.empty((0, window_samples))


def flatten_features(feature_dict: dict) -> np.ndarray:
    parts = []
    for v in feature_dict.values():
        arr = np.asarray(v).flatten()
        parts.append(arr)
    return np.concatenate(parts) if parts else np.array([])


def stack_feature_matrix(feature_list: List[np.ndarray]) -> np.ndarray:
    return np.vstack(feature_list)


def safe_log(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def class_counts(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, ...]:
    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
