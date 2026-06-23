"""
time_features.py — Vectorized time-domain EEG features.

compute_time_features_batch() now operates on the full (N, C, T) array
at once using numpy axis operations — no Python per-window loop, ~50× faster.
"""
import numpy as np
from utils.logger import get_logger

logger = get_logger("TimeFeatures")


# ─────────────────────────────────────────────────────────
# Single-window helper (kept for predict.py / ad-hoc use)
# ─────────────────────────────────────────────────────────

def compute_time_features(segment: np.ndarray) -> np.ndarray:
    """segment: (n_channels, n_samples) or (n_samples,)"""
    if segment.ndim == 1:
        segment = segment[np.newaxis, :]
    # delegate to batch, then squeeze
    return compute_time_features_batch(segment[np.newaxis])[0]


# ─────────────────────────────────────────────────────────
# VECTORIZED BATCH  — processes all windows at once
# ─────────────────────────────────────────────────────────

def compute_time_features_batch(X: np.ndarray) -> np.ndarray:
    """
    X: (N, C, T) or (N, T)
    Returns: (N, C*12) float32  — 12 features per channel

    Features per channel:
      mean, variance, std, RMS, skewness, kurtosis,
      Hjorth activity, mobility, complexity,
      zero-crossing rate, peak-to-peak range, IQR
    """
    if X.ndim == 2:
        X = X[:, np.newaxis, :]          # (N, 1, T)

    X = X.astype(np.float32)
    N, C, T = X.shape

    # ── basic moments ─────────────────────────────────────
    mean_  = X.mean(axis=2)                         # (N,C)
    var_   = X.var(axis=2)                          # (N,C)
    std_   = np.sqrt(var_ + 1e-10)                  # (N,C)
    rms_   = np.sqrt((X ** 2).mean(axis=2))         # (N,C)

    # ── skewness & kurtosis (scipy-free, vectorised) ──────
    x_c    = X - mean_[:, :, np.newaxis]            # (N,C,T) centred
    std3   = (std_ ** 3)[:, :, np.newaxis] + 1e-10
    std4   = (std_ ** 4)[:, :, np.newaxis] + 1e-10
    skew_  = (x_c ** 3).mean(axis=2) / std3[:, :, 0]   # (N,C)
    kurt_  = (x_c ** 4).mean(axis=2) / std4[:, :, 0] - 3.0  # excess

    # ── Hjorth parameters ─────────────────────────────────
    dX     = np.diff(X, axis=2)                     # (N,C,T-1)
    ddX    = np.diff(dX, axis=2)                    # (N,C,T-2)
    act_   = var_                                    # activity = variance
    mob_   = np.sqrt(dX.var(axis=2) / (var_ + 1e-10))  # (N,C)
    mob_d  = np.sqrt(ddX.var(axis=2) / (dX.var(axis=2) + 1e-10))
    comp_  = mob_d / (mob_ + 1e-10)                 # complexity

    # ── zero-crossing rate ────────────────────────────────
    signs  = np.sign(X)
    zcr_   = (np.diff(signs, axis=2) != 0).sum(axis=2) / T  # (N,C)

    # ── peak-to-peak & IQR ───────────────────────────────
    ptp_   = X.max(axis=2) - X.min(axis=2)
    q75    = np.percentile(X, 75, axis=2)
    q25    = np.percentile(X, 25, axis=2)
    iqr_   = q75 - q25

    # ── stack: (N, C, 12) → (N, C*12) ───────────────────
    parts  = np.stack(
        [mean_, var_, std_, rms_, skew_, kurt_,
         act_, mob_, comp_, zcr_, ptp_, iqr_],
        axis=2,
    )                                                # (N, C, 12)

    out = parts.reshape(N, C * 12).astype(np.float32)
    # replace any nan/inf from degenerate segments
    out = np.where(np.isfinite(out), out, 0.0)
    return out
