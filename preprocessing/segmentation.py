import numpy as np
from typing import Tuple, List
from utils.config import WINDOW_SIZE_SEC, OVERLAP
from utils.logger import get_logger

logger = get_logger("Segmentation")


def segment_eeg(
    data: np.ndarray,
    labels: np.ndarray,
    fs: float,
    window_sec: float = WINDOW_SIZE_SEC,
    overlap: float = OVERLAP,
    label_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    window_samples = int(window_sec * fs)
    step = int(window_samples * (1 - overlap))
    n_samples = data.shape[-1]

    X_segs, y_segs = [], []
    start = 0
    while start + window_samples <= n_samples:
        end = start + window_samples
        seg = data[..., start:end]
        seg_labels = labels[start:end] if labels is not None else np.zeros(window_samples)
        y = 1 if seg_labels.mean() >= label_threshold else 0
        X_segs.append(seg)
        y_segs.append(y)
        start += step

    if not X_segs:
        empty_shape = data.shape[:-1] + (window_samples,) if data.ndim > 1 else (window_samples,)
        return np.empty((0,) + empty_shape), np.empty((0,), dtype=np.int32)

    logger.debug(f"Segmentation: {len(X_segs)} segments, window={window_samples}samples, step={step}samples")
    return np.array(X_segs), np.array(y_segs, dtype=np.int32)


def segment_batch(
    signals: List[np.ndarray],
    labels_list: List[np.ndarray],
    fs: float,
    window_sec: float = WINDOW_SIZE_SEC,
    overlap: float = OVERLAP,
) -> Tuple[np.ndarray, np.ndarray]:
    X_all, y_all = [], []
    for sig, lab in zip(signals, labels_list):
        X_segs, y_segs = segment_eeg(sig, lab, fs, window_sec, overlap)
        if X_segs.shape[0] > 0:
            X_all.append(X_segs)
            y_all.append(y_segs)
    if not X_all:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)
