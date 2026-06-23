"""
frequency_features.py — Vectorized frequency-domain EEG features.

Uses numpy FFT on the full (N, C, T) batch at once instead of calling
scipy.signal.welch per window in a Python loop. ~40× faster.
"""
import numpy as np
from utils.config import BANDS
from utils.logger import get_logger

logger = get_logger("FreqFeatures")


def compute_frequency_features_batch(X: np.ndarray, fs: float) -> np.ndarray:
    """
    X: (N, C, T) or (N, T)
    Returns: (N, C * n_feats) float32

    Features per channel (n_feats = len(BANDS) + 5):
      band powers (normalised), spectral entropy,
      peak frequency, total power, mean frequency, spectral edge freq.
    """
    if X.ndim == 2:
        X = X[:, np.newaxis, :]          # (N, 1, T)

    X = X.astype(np.float32)
    N, C, T = X.shape

    # ── FFT over time axis ─────────────────────────────────
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(T).astype(np.float32)
    Xw     = X * window[np.newaxis, np.newaxis, :]   # (N, C, T)
    fft_   = np.fft.rfft(Xw, axis=2)                 # (N, C, T//2+1)
    freqs  = np.fft.rfftfreq(T, d=1.0 / fs)          # (F,)

    # PSD: |FFT|² / (fs * T)  — one-sided so ×2 for non-DC/Nyquist
    psd = (np.abs(fft_) ** 2) / (fs * T)             # (N, C, F)
    psd[:, :, 1:-1] *= 2.0

    # df for trapezoid integration
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    # total power per window/channel  (N, C)
    total_power = psd.sum(axis=2) * df + 1e-10

    # ── band powers (normalised) ───────────────────────────
    band_feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs <= hi)
        bp   = psd[:, :, mask].sum(axis=2) * df      # (N, C)
        band_feats.append(bp / total_power)           # normalised

    # ── spectral entropy ───────────────────────────────────
    psd_norm = psd / total_power[:, :, np.newaxis]   # (N, C, F)
    sp_ent   = -(psd_norm * np.log(psd_norm + 1e-10)).sum(axis=2)  # (N, C)

    # ── peak frequency ─────────────────────────────────────
    peak_f   = freqs[psd.argmax(axis=2)]             # (N, C)

    # ── mean frequency ─────────────────────────────────────
    mean_f   = (freqs[np.newaxis, np.newaxis, :] * psd).sum(axis=2) / \
               (psd.sum(axis=2) + 1e-10)             # (N, C)

    # ── spectral edge frequency (95 % cumulative power) ───
    cum_psd  = np.cumsum(psd, axis=2)                # (N, C, F)
    thresh   = 0.95 * cum_psd[:, :, -1:]            # (N, C, 1)
    # first freq index where cumulative power ≥ threshold
    edge_idx = (cum_psd >= thresh).argmax(axis=2)    # (N, C)
    edge_f   = freqs[edge_idx]                       # (N, C)

    # ── assemble: (N, C, n_feats) ─────────────────────────
    parts = band_feats + [sp_ent, peak_f, total_power, mean_f, edge_f]
    stacked = np.stack(parts, axis=2)                # (N, C, n_feats)
    out     = stacked.reshape(N, C * stacked.shape[2]).astype(np.float32)
    out     = np.where(np.isfinite(out), out, 0.0)
    return out


def compute_frequency_features(segment: np.ndarray, fs: float) -> np.ndarray:
    """Single-window convenience wrapper."""
    if segment.ndim == 1:
        segment = segment[np.newaxis, :]
    return compute_frequency_features_batch(segment[np.newaxis], fs)[0]
