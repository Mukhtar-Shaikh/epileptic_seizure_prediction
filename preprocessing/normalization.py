import numpy as np
from utils.logger import get_logger

logger = get_logger("Normalization")


def zscore_normalize(data: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    mean = data.mean(axis=axis, keepdims=True)
    std = data.std(axis=axis, keepdims=True) + eps
    return (data - mean) / std


def minmax_normalize(data: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    dmin = data.min(axis=axis, keepdims=True)
    dmax = data.max(axis=axis, keepdims=True)
    return (data - dmin) / (dmax - dmin + eps)


def robust_normalize(data: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    median = np.median(data, axis=axis, keepdims=True)
    q75 = np.percentile(data, 75, axis=axis, keepdims=True)
    q25 = np.percentile(data, 25, axis=axis, keepdims=True)
    iqr = q75 - q25 + eps
    return (data - median) / iqr


def normalize(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    methods = {"zscore": zscore_normalize, "minmax": minmax_normalize, "robust": robust_normalize}
    if method not in methods:
        raise ValueError(f"Unknown normalization method: {method}. Choose from {list(methods.keys())}")
    result = methods[method](data)
    logger.debug(f"Normalization ({method}) complete: shape={result.shape}")
    return result
