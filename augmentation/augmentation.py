import numpy as np
from typing import Tuple
from utils.logger import get_logger

logger = get_logger("Augmentation")


def add_gaussian_noise(data: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    noise = np.random.normal(0, sigma * data.std(axis=-1, keepdims=True), data.shape)
    return data + noise


def time_shift(data: np.ndarray, max_shift: int = 100) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(data, shift, axis=-1)


def scale_signal(data: np.ndarray, low: float = 0.9, high: float = 1.1) -> np.ndarray:
    factor = np.random.uniform(low, high)
    return data * factor


def window_crop(data: np.ndarray, crop_fraction: float = 0.1) -> np.ndarray:
    n = data.shape[-1]
    crop = int(n * crop_fraction)
    start = np.random.randint(0, crop + 1)
    end = n - np.random.randint(0, crop + 1)
    cropped = data[..., start:end]
    return np.pad(cropped, [(0, 0)] * (data.ndim - 1) + [(0, n - cropped.shape[-1])], mode="edge")


def frequency_perturbation(data: np.ndarray, fs: float, max_shift_hz: float = 1.0) -> np.ndarray:
    fft = np.fft.rfft(data, axis=-1)
    freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / fs)
    shift_samples = int(max_shift_hz / (fs / data.shape[-1]))
    fft_shifted = np.roll(fft, np.random.randint(-shift_samples, shift_samples + 1), axis=-1)
    return np.fft.irfft(fft_shifted, n=data.shape[-1], axis=-1)


def augment_batch(
    X: np.ndarray,
    y: np.ndarray,
    fs: float = 256.0,
    methods: Tuple[str, ...] = ("noise", "shift", "scale", "crop"),
) -> Tuple[np.ndarray, np.ndarray]:
    augmented_X, augmented_y = [X.copy()], [y.copy()]
    method_map = {
        "noise": lambda d: add_gaussian_noise(d),
        "shift": lambda d: time_shift(d),
        "scale": lambda d: scale_signal(d),
        "crop": lambda d: window_crop(d),
        "freq": lambda d: frequency_perturbation(d, fs),
    }
    for method in methods:
        fn = method_map.get(method)
        if fn is None:
            logger.warning(f"Unknown augmentation method: {method}")
            continue
        X_aug = np.array([fn(x) for x in X])
        augmented_X.append(X_aug)
        augmented_y.append(y.copy())
        logger.debug(f"Augmented with {method}: {X_aug.shape}")

    return np.concatenate(augmented_X, axis=0), np.concatenate(augmented_y, axis=0)


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("imbalanced-learn not installed, skipping SMOTE.")
        return X, y

    original_shape = X.shape
    X_2d = X.reshape(len(X), -1)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        logger.warning("SMOTE requires at least 2 classes.")
        return X, y
    min_count = counts.min()
    k = max(1, min(5, min_count - 1))
    sm = SMOTE(random_state=random_state, k_neighbors=k)
    try:
        X_res, y_res = sm.fit_resample(X_2d, y)
        if len(original_shape) > 2:
            X_res = X_res.reshape((-1,) + original_shape[1:])
        logger.info(f"SMOTE: {original_shape[0]} -> {len(X_res)} samples")
        return X_res, y_res
    except Exception as e:
        logger.warning(f"SMOTE failed: {e}")
        return X, y
