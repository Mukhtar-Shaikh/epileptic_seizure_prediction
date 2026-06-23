import numpy as np
from features.time_features import compute_time_features_batch
from features.frequency_features import compute_frequency_features_batch
from features.wavelet_features import compute_wavelet_features_batch
from features.nonlinear_features import compute_nonlinear_features_batch
from utils.logger import get_logger

logger = get_logger("HybridFeatures")


def compute_hybrid_features(
    X: np.ndarray,
    fs: float,
    include_deep: bool = False,
    deep_embeddings: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the full hybrid feature vector for each segment.

    X: (n_segments, n_channels, n_samples) or (n_segments, n_samples)
    Returns: (n_segments, n_features)
    """
    logger.info("Computing time-domain features...")
    X_time = compute_time_features_batch(X)

    logger.info("Computing frequency-domain features...")
    X_freq = compute_frequency_features_batch(X, fs)

    logger.info("Computing wavelet features...")
    X_wav = compute_wavelet_features_batch(X)

    logger.info("Computing nonlinear features...")
    X_nonlin = compute_nonlinear_features_batch(X)

    parts = [X_time, X_freq, X_wav, X_nonlin]

    if include_deep and deep_embeddings is not None:
        if deep_embeddings.shape[0] == X.shape[0]:
            parts.append(deep_embeddings)
            logger.info(f"Appending deep embeddings: shape={deep_embeddings.shape}")
        else:
            logger.warning("Deep embeddings shape mismatch; skipping.")

    hybrid = np.concatenate(parts, axis=1)
    logger.info(f"Hybrid feature matrix: {hybrid.shape}")
    return hybrid.astype(np.float32)


def replace_nan_inf(X: np.ndarray) -> np.ndarray:
    X = np.where(np.isnan(X), 0.0, X)
    X = np.where(np.isinf(X), 0.0, X)
    return X
